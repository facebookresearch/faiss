# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""SSD scans into shuffled host windows and asynchronous CUDA batches.

Adapts the window scans and chunk-group permutation from the reference
ssd_prefetch.py and data.py. Each rank reads its own contiguous row shard.
"""

from concurrent.futures import ThreadPoolExecutor
import errno
import math
import mmap
import os
import time

import numpy as np
import torch


class SSDRowSource:
    """A rank's range in fixed-width uint8 files, optionally with row headers.

    ``paths`` are ordered corpus chunks. O_DIRECT uses page-aligned buffers
    and reads; filesystems that refuse it fall back to buffered preadv.
    """

    def __init__(self, paths, d, begin, end, header_bytes=0, direct=True):
        if d < 1 or header_bytes < 0 or not 0 <= begin < end:
            raise ValueError("invalid SSD row layout or range")
        self.d, self.header_bytes = int(d), int(header_bytes)
        self.row_bytes = self.d + self.header_bytes
        self.begin, self.n = int(begin), int(end - begin)
        self.fds = []
        self.direct = []
        lengths = []
        try:
            for path in paths:
                size = os.path.getsize(path)
                if size % self.row_bytes:
                    raise ValueError(f"incomplete row in {path}")
                lengths.append(size // self.row_bytes)
                try:
                    fd = os.open(
                        path, os.O_RDONLY | (os.O_DIRECT if direct else 0)
                    )
                    is_direct = bool(direct)
                except OSError as error:
                    if error.errno not in (errno.EINVAL, errno.EOPNOTSUPP):
                        raise
                    fd, is_direct = os.open(path, os.O_RDONLY), False
                self.fds.append(fd)
                self.direct.append(is_direct)
            self.ends = np.cumsum(lengths)
            if not lengths or end > self.ends[-1]:
                raise ValueError("SSD shard exceeds file rows")
        except BaseException:
            self.close()
            raise

    def close(self):
        for fd in self.fds:
            os.close(fd)
        self.fds.clear()

    def read(self, begin, end):
        """Read a sequential local range into an owned compact uint8 array."""
        if not 0 <= begin <= end <= self.n:
            raise IndexError("SSD row range out of bounds")
        out = np.empty((end - begin, self.d), dtype=np.uint8)
        a, b, at = self.begin + begin, self.begin + end, 0
        for fd, hi, lo in zip(self.fds, self.ends, np.r_[0, self.ends[:-1]]):
            start, stop = max(a, int(lo)), min(b, int(hi))
            if start >= stop:
                continue
            offset = (start - lo) * self.row_bytes
            length = (stop - start) * self.row_bytes
            aligned = int(offset) // mmap.PAGESIZE * mmap.PAGESIZE
            slack = int(offset) - aligned
            size = math.ceil((slack + length) / mmap.PAGESIZE) * mmap.PAGESIZE
            with mmap.mmap(-1, size) as buffer:
                got = os.preadv(fd, [buffer], aligned)
                if got < slack + length:
                    raise OSError("short SSD read")
                rows = np.ndarray(
                    (stop - start, self.row_bytes),
                    dtype=np.uint8,
                    buffer=buffer,
                    offset=slack,
                )
                out[at : at + stop - start] = rows[:, self.header_bytes :]
                del rows
            at += stop - start
        return out

    def gather(self, request, out, depth=256):
        """Queue small aligned reads of random rows into a host array.

        Rows sharing a starting block use one read. Payload offsets include
        bvecs headers, and reads span two blocks when a payload straddles one.
        Buffers and the ring are bounded by depth; output order is preserved.
        """
        if __package__:
            from . import ctypes_liburing as uring
        else:
            import ctypes_liburing as uring

        request = np.asarray(request, dtype=np.int64)
        if request.ndim != 1 or (
            len(request) and (request.min() < 0 or request.max() >= self.n)
        ):
            raise IndexError("random request outside SSD shard")
        if depth < 1 or depth & (depth - 1):
            raise ValueError("I/O depth must be a positive power of two")
        destination = out.numpy() if isinstance(out, torch.Tensor) else out
        if destination.shape != (len(request), self.d):
            raise ValueError("random read output has the wrong shape")
        stats = {"bytes_read": 0, "requests": 0}
        if not len(request):
            return stats
        block = 4096
        slot_bytes = math.ceil((self.d + block - 1) / block) * block
        ring = uring.io_uring()
        buffers, views = [], []
        try:
            uring.io_uring_queue_init(depth, ring, 0)
            buffers = [mmap.mmap(-1, slot_bytes) for _ in range(depth)]
            views = [
                np.frombuffer(buffer, dtype=np.uint8) for buffer in buffers
            ]
            cqe = uring.io_uring_cqe()
            columns = np.arange(self.d)
            absolute = self.begin + request
            for fd, hi, lo in zip(
                self.fds, self.ends, np.r_[0, self.ends[:-1]]
            ):
                ids = np.flatnonzero((absolute >= lo) & (absolute < hi))
                if not len(ids):
                    continue
                starts = (
                    absolute[ids] - lo
                ) * self.row_bytes + self.header_bytes
                perm = np.argsort(starts, kind="stable")
                ids, starts = ids[perm], starts[perm]
                blocks = starts // block
                bounds = np.r_[
                    0, np.flatnonzero(np.diff(blocks)) + 1, len(ids)
                ]
                for first in range(0, len(bounds) - 1, depth):
                    pending = []
                    for j in range(first, min(first + depth, len(bounds) - 1)):
                        a, b = int(bounds[j]), int(bounds[j + 1])
                        offset = int(blocks[a]) * block
                        needed = int(starts[b - 1]) + self.d - offset
                        length = math.ceil(needed / block) * block
                        slot = len(pending)
                        sqe = uring.io_uring_get_sqe(ring)
                        uring.io_uring_prep_read(
                            sqe, fd, buffers[slot], length, offset
                        )
                        sqe.user_data = slot
                        pending.append((a, b, offset, needed))
                    uring.io_uring_submit_and_wait(ring, len(pending))
                    error = None
                    for _ in pending:
                        if uring.io_uring_peek_cqe(ring, cqe):
                            raise OSError("missing random read completion")
                        a, b, offset, needed = pending[cqe.user_data]
                        if cqe.res < 0:
                            error = OSError(-cqe.res, "random SSD read failed")
                        elif cqe.res < needed:
                            error = OSError("short random SSD read")
                        else:
                            destination[ids[a:b]] = views[cqe.user_data][
                                (starts[a:b] - offset)[:, None] + columns
                            ]
                            stats["bytes_read"] += cqe.res
                            stats["requests"] += 1
                        uring.io_uring_cqe_seen(ring, cqe)
                    if error is not None:
                        raise error
        finally:
            uring.io_uring_queue_exit(ring)
            views.clear()
            for buffer in buffers:
                buffer.close()
        return stats


def chunk_grouped_order(n, rows_per_chunk, groups, window_rows, seed):
    """Permute all rows once, grouping full windows by random scan chunks.

    Each full window draws from one of ``groups`` random subsets of scan
    chunks. Full windows are shuffled across groups; short group tails are
    placed last. Groups=1 gives a uniform row permutation.
    """
    if min(n, rows_per_chunk, groups, window_rows) < 1:
        raise ValueError("permutation dimensions must be positive")
    rng = np.random.RandomState(seed)
    if groups == 1:
        return rng.permutation(n)
    count = math.ceil(n / rows_per_chunk)
    pieces = np.array_split(rng.permutation(count), min(groups, count))
    layout, full, partial = [], [], []
    for j, chunks in enumerate(pieces):
        last = chunks == count - 1
        chunks = np.r_[chunks[~last], chunks[last]]
        size = len(chunks) * rows_per_chunk - (
            count * rows_per_chunk - n if last.any() else 0
        )
        layout.append((chunks, size))
        full.extend((j, k) for k in range(size // window_rows))
        if size % window_rows:
            partial.append((j, size // window_rows))
    positions, at = {}, 0
    for items in (full, partial):
        for index in rng.permutation(len(items)):
            j, k = items[index]
            length = min(window_rows, layout[j][1] - k * window_rows)
            positions[j, k] = (at, length)
            at += length
    output = np.empty(n, dtype=np.int64)
    for j, (chunks, size) in enumerate(layout):
        perm = rng.permutation(size)
        for k in range(math.ceil(size / window_rows)):
            dest, length = positions[j, k]
            for start in range(0, length, 1 << 20):
                q = perm[
                    k * window_rows
                    + start : k * window_rows
                    + min(length, start + (1 << 20))
                ]
                output[dest + start : dest + start + len(q)] = (
                    chunks[q // rows_per_chunk] * rows_per_chunk
                    + q % rows_per_chunk
                )
    return output


class SSDPrefetch:
    """Two host windows, chunk scans, and one CUDA batch copied ahead.

    Random requests retain their exact order, including duplicates. Reads
    within each window are issued in increasing chunk order; chunks with no
    requested row are skipped. A window is reused only after its asynchronous
    copies complete. Close the iterator before closing its source.
    """

    def __init__(
        self,
        source,
        device,
        batch_size,
        *,
        n_prefetch="auto",
        chunk_groups=4,
        chunk_bytes=64 << 20,
        ram_budget_bytes=16 << 30,
        read_threads=1,
        extract_threads=4,
    ):
        if (
            min(
                batch_size,
                chunk_groups,
                chunk_bytes,
                ram_budget_bytes,
                read_threads,
                extract_threads,
            )
            < 1
        ):
            raise ValueError(
                "SSD prefetch sizes and thread counts must be positive"
            )
        self.source, self.device = source, torch.device(device)
        if self.device.type != "cuda":
            raise ValueError("SSD prefetch requires CUDA")
        self.batch_size, self.chunk_groups = batch_size, chunk_groups
        self.rows_per_chunk = max(1, chunk_bytes // source.row_bytes)
        steps = math.ceil(source.n / batch_size)
        capacity = ram_budget_bytes // (2 * batch_size * source.d)
        if capacity < 1:
            raise ValueError("RAM budget must fit two host batches")
        self.n_prefetch = (
            min(steps, capacity) if n_prefetch == "auto" else int(n_prefetch)
        )
        if not 1 <= self.n_prefetch <= capacity:
            raise ValueError("prefetch windows exceed RAM budget")
        self.window_rows = self.n_prefetch * batch_size
        self.read_threads, self.extract_threads = read_threads, extract_threads
        self.copy_stream = torch.cuda.Stream(self.device)
        self.stats = {"bytes_read": 0, "wait_s": 0.0, "windows": 0}

    def order(self, seed):
        return chunk_grouped_order(
            self.source.n,
            self.rows_per_chunk,
            self.chunk_groups,
            self.window_rows,
            seed,
        )

    def fill_window(self, request, out):
        request = np.asarray(request, dtype=np.int64)
        if len(request) and (
            request.min() < 0 or request.max() >= self.source.n
        ):
            raise IndexError("prefetch request outside SSD shard")
        order = np.argsort(request, kind="stable")
        rows = request[order]
        chunk_ids = rows // self.rows_per_chunk
        bounds = np.r_[0, np.flatnonzero(np.diff(chunk_ids)) + 1, len(rows)]
        if not len(rows):
            return
        destination = out.numpy()

        def read_chunk(lo, hi):
            chunk = int(chunk_ids[lo])
            begin = chunk * self.rows_per_chunk
            end = min(self.source.n, begin + self.rows_per_chunk)
            return lo, hi, begin, self.source.read(begin, end)

        def extract(item):
            lo, hi, begin, data = item
            destination[order[lo:hi]] = data[rows[lo:hi] - begin]

        with ThreadPoolExecutor(
            self.read_threads
        ) as readers, ThreadPoolExecutor(self.extract_threads) as extractors:
            # Keep read and extraction queues bounded to avoid retaining a scan.
            pending_reads, pending_extracts = [], []
            for lo, hi in zip(bounds[:-1], bounds[1:]):
                pending_reads.append(
                    readers.submit(read_chunk, int(lo), int(hi))
                )
                if len(pending_reads) >= 2 * self.read_threads:
                    item = pending_reads.pop(0).result()
                    self.stats["bytes_read"] += (
                        item[3].shape[0] * self.source.row_bytes
                    )
                    pending_extracts.append(extractors.submit(extract, item))
                if len(pending_extracts) >= 2 * self.extract_threads:
                    pending_extracts.pop(0).result()
            for future in pending_reads:
                item = future.result()
                self.stats["bytes_read"] += (
                    item[3].shape[0] * self.source.row_bytes
                )
                pending_extracts.append(extractors.submit(extract, item))
            for future in pending_extracts:
                future.result()
        self.stats["windows"] += 1

    def fill_sequential(self, start, end, out):
        def read(begin):
            stop = min(end, begin + self.rows_per_chunk)
            return begin, stop, self.source.read(begin, stop)

        with ThreadPoolExecutor(self.read_threads) as readers:
            pending = []
            for begin in range(start, end, self.rows_per_chunk):
                pending.append(readers.submit(read, begin))
                if len(pending) >= 2 * self.read_threads:
                    begin, stop, data = pending.pop(0).result()
                    out[begin - start : stop - start].copy_(
                        torch.from_numpy(data)
                    )
            for future in pending:
                begin, stop, data = future.result()
                out[begin - start : stop - start].copy_(torch.from_numpy(data))
        self.stats["bytes_read"] += (end - start) * self.source.row_bytes
        self.stats["windows"] += 1

    def batches(self, order=None, *, sequential=False, mean=None, scale=None):
        """Yield rank-local FP32 CUDA batches, optionally centered.

        Sequential mode performs one corpus scan without permutation or
        repeated sparse window scans, for centering and frozen closing means.
        """
        self.copy_stream.wait_stream(torch.cuda.current_stream(self.device))
        if not sequential and (order is None or len(order) != self.source.n):
            raise ValueError("random mode needs one request per source row")
        slots = [
            torch.empty(
                (self.window_rows, self.source.d),
                dtype=torch.uint8,
                pin_memory=True,
            )
            for _ in range(2)
        ]
        events = [None, None]
        pool = ThreadPoolExecutor(1)

        def fill(start, slot):
            if events[slot] is not None:
                events[slot].synchronize()
            end = min(self.source.n, start + self.window_rows)
            if sequential:
                self.fill_sequential(start, end, slots[slot][: end - start])
            else:
                self.fill_window(order[start:end], slots[slot][: end - start])
            return end - start

        def copies():
            future = pool.submit(fill, 0, 0)
            for window, start in enumerate(
                range(0, self.source.n, self.window_rows)
            ):
                slot = window % 2
                waited = time.perf_counter()
                length = future.result()
                self.stats["wait_s"] += time.perf_counter() - waited
                if start + self.window_rows < self.source.n:
                    future = pool.submit(
                        fill, start + self.window_rows, 1 - slot
                    )
                for a in range(0, length, self.batch_size):
                    with torch.cuda.stream(self.copy_stream):
                        batch = (
                            slots[slot][a : min(length, a + self.batch_size)]
                            .to(self.device, non_blocking=True)
                            .float()
                        )
                        if mean is not None:
                            batch.sub_(mean)
                        if scale is not None:
                            batch.div_(scale)
                        done = torch.cuda.Event()
                        done.record(self.copy_stream)
                    events[slot] = done
                    yield batch, done

        iterator = copies()
        pending = None
        try:
            for batch, ready in iterator:
                if pending is not None:
                    previous, event = pending
                    torch.cuda.current_stream(self.device).wait_event(event)
                    previous.record_stream(
                        torch.cuda.current_stream(self.device)
                    )
                    yield previous
                pending = batch, ready
            if pending is not None:
                batch, ready = pending
                torch.cuda.current_stream(self.device).wait_event(ready)
                batch.record_stream(torch.cuda.current_stream(self.device))
                yield batch
        finally:
            iterator.close()
            pool.shutdown(wait=True, cancel_futures=True)
            self.copy_stream.synchronize()
