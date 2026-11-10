# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import numpy as np
import os
import operator

"""
I/O functions in fvecs, bvecs, ivecs formats
definition of the formats here: http://corpus-texmex.irisa.fr/
"""


def ivecs_read(fname):
    a = np.fromfile(fname, dtype="int32")
    if sys.byteorder == "big":
        a.byteswap(inplace=True)
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view("float32")


def ivecs_mmap(fname):
    assert sys.byteorder != "big"
    a = np.memmap(fname, dtype="int32", mode="r")
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:]


def fvecs_mmap(fname):
    return ivecs_mmap(fname).view("float32")


def bvecs_mmap(fname):
    x = np.memmap(fname, dtype="uint8", mode="r")
    if sys.byteorder == "big":
        da = x[:4][::-1].copy()
        d = da.view("int32")[0]
    else:
        d = x[:4].view("int32")[0]
    return x.reshape(-1, d + 4)[:, 4:]


def ivecs_write(fname, m):
    n, d = m.shape
    m1 = np.empty((n, d + 1), dtype="int32")
    m1[:, 0] = d
    m1[:, 1:] = m
    if sys.byteorder == "big":
        m1.byteswap(inplace=True)
    m1.tofile(fname)


def fvecs_write(fname, m):
    m = m.astype("float32")
    ivecs_write(fname, m.view("int32"))


def bvecs_iter(filepath, batch_size=100_000):
    """
    Memory-mapped iterator - only loads requested slices into RAM
    """

    batch_size = operator.index(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    file_size = os.path.getsize(filepath)
    with open(filepath, "rb") as f:
        dim = np.frombuffer(f.read(4), dtype="<i4")[0]

    bytes_per_vec = 4 + dim
    n_vectors = file_size // bytes_per_vec

    mm = np.memmap(filepath, mode="r", dtype=np.uint8)
    records = mm.reshape(n_vectors, bytes_per_vec)

    for start in range(0, n_vectors, batch_size):
        end = min(start + batch_size, n_vectors)
        yield records[start:end, 4:]


def bvecs_iter_chunked(chunk_folder, batch_size=100_000):
    """
    Memory-mapped iterator over chunked .bvecs files.
    Iterates through all chunk files in order (chunk_0000.bvecs,
    chunk_0001.bvecs, etc.) and yields batches of vectors, handling cases
    where batches span multiple files.

    Args:
        chunk_folder: path to folder containing chunk_XXXX.bvecs files
        batch_size: number of vectors to yield per batch

    Yields:
        numpy array of shape (batch_size, d) or smaller for last batch

    Raises:
        ValueError: if there are gaps in the chunk sequence
    """

    batch_size = operator.index(batch_size)
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")

    # Find all chunk files and sort by number, including unpadded names and
    # chunk numbers wider than the usual four-digit padding.
    chunk_files = []
    with os.scandir(chunk_folder) as entries:
        for entry in entries:
            if (
                entry.is_file()
                and entry.name.startswith("chunk_")
                and entry.name.endswith(".bvecs")
            ):
                try:
                    number = int(entry.name[len("chunk_") : -len(".bvecs")])
                except ValueError:
                    raise ValueError(
                        f"Invalid chunk filename format: {entry.name}"
                    ) from None
                chunk_files.append((number, entry.path))
    chunk_files.sort()

    if not chunk_files:
        raise ValueError(f"No chunk files found in {chunk_folder}")

    # Extract chunk numbers and verify no gaps
    chunk_numbers = [number for number, path in chunk_files]

    # Check for gaps in sequence
    expected_chunks = list(range(len(chunk_numbers)))
    if sorted(chunk_numbers) != expected_chunks:
        missing = set(expected_chunks) - set(chunk_numbers)
        raise ValueError(
            f"Gap detected in chunk sequence! Missing chunks: "
            f"{sorted(missing)}\n"
            f"Found chunks: {sorted(chunk_numbers)}\n"
            f"Expected continuous sequence from 0 to {len(chunk_numbers)-1}"
        )

    # Get dimension from first chunk
    with open(chunk_files[0][1], "rb") as f:
        dim = np.frombuffer(f.read(4), dtype="<i4")[0]

    bytes_per_vec = 4 + dim
    chunk_sizes = [
        (path, os.path.getsize(path) // bytes_per_vec)
        for _, path in chunk_files
    ]
    buffer_capacity = min(batch_size, sum(size for path, size in chunk_sizes))

    # Buffer to accumulate vectors across chunk boundaries
    buffer = None
    buffer_size = 0

    # Iterate through each chunk file
    for chunk_path, n_vectors in chunk_sizes:
        # Memory-map this chunk
        mm = np.memmap(chunk_path, mode="r", dtype=np.uint8)
        records = mm.reshape(n_vectors, bytes_per_vec)
        vectors = records[:, 4:]  # Skip dimension prefix

        start = 0

        # First, handle any buffered data from previous chunk
        if buffer is not None:
            needed = min(batch_size - buffer_size, n_vectors)
            buffer[buffer_size : buffer_size + needed] = vectors[:needed]
            buffer_size += needed
            if buffer_size == batch_size:
                yield buffer
                buffer = None
                buffer_size = 0
                start = needed
            else:
                continue

        # Now process complete batches from this chunk
        while start + batch_size <= n_vectors:
            yield vectors[start : start + batch_size]
            start += batch_size

        remainder = n_vectors - start
        if remainder > 0:
            # Each vector spanning a chunk boundary is copied once, even
            # when a batch spans many small chunks.
            buffer = np.empty((buffer_capacity, dim), dtype="uint8")
            buffer[:remainder] = vectors[start:]
            buffer_size = remainder

    if buffer is not None:
        yield buffer[:buffer_size]
