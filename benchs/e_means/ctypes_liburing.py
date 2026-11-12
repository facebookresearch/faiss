# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""The reference ctypes io_uring subset: aligned READ requests without wheels.

Submission uses membarrier before publishing SQEs and after waiting for CQEs,
including on ARM. Completion heads are published at the next submission.
The caller must keep writable buffers alive until their requests complete.
"""
from __future__ import annotations

import ctypes
import errno
import mmap
import os
import platform
import struct


# Linux userspace ABI: fixed per architecture, forever. io_uring numbers
# are shared; only membarrier differs.
_SYSCALLS = {
    # nr_setup, nr_enter, nr_membarrier
    "x86_64": (425, 426, 324),
    "aarch64": (425, 426, 283),
    "arm64": (425, 426, 283),  # Darwin spelling; harmless if unused
}

# membarrier commands (linux/membarrier.h).
_MEMBARRIER_REGISTER_PRIVATE_EXPEDITED = 0x10
_MEMBARRIER_PRIVATE_EXPEDITED = 0x08

_IORING_OP_READ = 22
_IORING_ENTER_GETEVENTS = 1
_IORING_OFF_SQ_RING = 0
_IORING_OFF_CQ_RING = 0x8000000
_IORING_OFF_SQES = 0x10000000
_SQE_BYTES = 64
_CQE_BYTES = 16


_libc = ctypes.CDLL(None, use_errno=True)
# libc syscall returns -1 and sets errno. Full-width arguments preserve
# pointers on 64-bit platforms.
_libc.syscall.argtypes = [ctypes.c_long] * 7
_libc.syscall.restype = ctypes.c_long


def _syscall(nr, a1=0, a2=0, a3=0, a4=0, a5=0, a6=0):
    result = _libc.syscall(nr, a1, a2, a3, a4, a5, a6)
    return -ctypes.get_errno() if result == -1 else result


class _SqOffsets(ctypes.Structure):
    _fields_ = [
        ("head", ctypes.c_uint32),
        ("tail", ctypes.c_uint32),
        ("ring_mask", ctypes.c_uint32),
        ("ring_entries", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("dropped", ctypes.c_uint32),
        ("array", ctypes.c_uint32),
        ("resv1", ctypes.c_uint32),
        ("resv2", ctypes.c_uint64),
    ]


class _CqOffsets(ctypes.Structure):
    _fields_ = [
        ("head", ctypes.c_uint32),
        ("tail", ctypes.c_uint32),
        ("ring_mask", ctypes.c_uint32),
        ("ring_entries", ctypes.c_uint32),
        ("overflow", ctypes.c_uint32),
        ("cqes", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("resv1", ctypes.c_uint32),
        ("resv2", ctypes.c_uint64),
    ]


class _Params(ctypes.Structure):
    _fields_ = [
        ("sq_entries", ctypes.c_uint32),
        ("cq_entries", ctypes.c_uint32),
        ("flags", ctypes.c_uint32),
        ("sq_thread_cpu", ctypes.c_uint32),
        ("sq_thread_idle", ctypes.c_uint32),
        ("features", ctypes.c_uint32),
        ("wq_fd", ctypes.c_uint32),
        ("resv", ctypes.c_uint32 * 3),
        ("sq_off", _SqOffsets),
        ("cq_off", _CqOffsets),
    ]


assert ctypes.sizeof(_Params) == 120, ctypes.sizeof(_Params)


_registered = False


def _fence():
    """A full memory barrier both CPUs honor. Raises, never degrades."""
    global _registered
    nr = _numbers()[2]
    if not _registered:
        rc = _syscall(nr, _MEMBARRIER_REGISTER_PRIVATE_EXPEDITED, 0, 0)
        if rc != 0:
            raise OSError(
                -rc if rc < 0 else rc, "membarrier registration failed"
            )
        _registered = True
    rc = _syscall(nr, _MEMBARRIER_PRIVATE_EXPEDITED, 0, 0)
    if rc != 0:
        raise OSError(-rc if rc < 0 else rc, "membarrier fence failed")


def _numbers():
    """(setup, enter, membarrier) syscall numbers for this CPU."""
    try:
        return _SYSCALLS[platform.machine()]
    except KeyError:
        raise OSError(
            errno.ENOSYS,
            "no syscall numbers for architecture %r (have: %s)"
            % (platform.machine(), sorted(_SYSCALLS)),
        )


def _u32(view, offset):
    return struct.unpack_from("I", view, offset)[0]


def _put_u32(view, offset, value):
    struct.pack_into("I", view, offset, value & 0xFFFFFFFF)


class _Sqe:
    """A reserved submission slot; fields land in the shared ring."""

    def __init__(self, ring, index):
        self._ring = ring
        self._off = index * _SQE_BYTES

    @property
    def user_data(self):
        return struct.unpack_from("Q", self._ring._sqes, self._off + 32)[0]

    @user_data.setter
    def user_data(self, value):
        struct.pack_into("Q", self._ring._sqes, self._off + 32, value)


class _Ring:
    def __init__(self):
        self.fd = -1
        self.entries = 0
        self._sq = None
        self._cq = None
        self._sqes = None
        self._sq_mask = 0
        self._cq_mask = 0
        self._sq_tail = 0
        self._cq_head = 0
        self._pending = 0
        self._closed = False


class _Cqe:
    def __init__(self):
        self.res = 0
        self.user_data = 0


def io_uring():
    """An uninitialized ring; pass it to io_uring_queue_init next."""
    return _Ring()


def io_uring_cqe():
    """A completion holder; peek_cqe fills its res/user_data."""
    return _Cqe()


def io_uring_queue_init(entries, ring, flags):
    """Create the ring: setup syscall plus the three shared mappings."""
    if flags != 0:
        raise ValueError(
            "only flags=0 (plain syscall-driven ring); got %r" % (flags,)
        )
    if entries <= 0 or entries & (entries - 1):
        raise ValueError(
            "entries must be a positive power of two, not %r" % (entries,)
        )
    nr_setup = _numbers()[0]
    params = _Params()
    params.sq_entries = entries
    fd = _syscall(nr_setup, int(entries), ctypes.addressof(params))
    if fd < 0:
        err = -fd
        if err == errno.ENOSYS:
            raise OSError(
                err,
                "io_uring_setup missing: pre-5.1 kernel, "
                "seccomp-blocked io_uring, or wrong syscall "
                "numbers for %s" % platform.machine(),
            )
        raise OSError(err, "io_uring_setup failed")
    if params.cq_entries < entries:
        os.close(fd)
        raise OSError(
            "CQ ring holds %d entries for a depth-%d queue; "
            "cannot bound completions" % (params.cq_entries, entries)
        )
    sq_size = params.sq_off.array + entries * 4
    cq_size = params.cq_off.cqes + params.cq_entries * _CQE_BYTES
    ring.fd = fd
    try:
        sq = ring._sq = mmap.mmap(
            fd, sq_size, offset=_IORING_OFF_SQ_RING, access=mmap.ACCESS_WRITE
        )
        cq = ring._cq = mmap.mmap(
            fd, cq_size, offset=_IORING_OFF_CQ_RING, access=mmap.ACCESS_WRITE
        )
        sqes = ring._sqes = mmap.mmap(
            fd,
            entries * _SQE_BYTES,
            offset=_IORING_OFF_SQES,
            access=mmap.ACCESS_WRITE,
        )
    except Exception:
        io_uring_queue_exit(ring)
        raise
    # Identity submission map: slot i consumes SQE i. Written once; the
    # kernel never touches it again.
    for i in range(entries):
        _put_u32(sq, params.sq_off.array + i * 4, i)
    ring.fd = fd
    ring.entries = entries
    ring._sq = sq
    ring._cq = cq
    ring._sqes = sqes
    ring._sq_mask = _u32(sq, params.sq_off.ring_mask)
    ring._cq_mask = _u32(cq, params.cq_off.ring_mask)
    ring._sq_off = params.sq_off
    ring._cq_off = params.cq_off
    return 0


def io_uring_queue_exit(ring):
    """Unmap the rings and close the fd. Idempotent."""
    if ring._closed:
        return
    ring._closed = True
    try:
        for mapping in (ring._sq, ring._cq, ring._sqes):
            if mapping is not None:
                mapping.close()
    finally:
        if ring.fd >= 0:
            os.close(ring.fd)
            ring.fd = -1


def io_uring_get_sqe(ring):
    """Reserve a submission slot, or None when the queue is full."""
    if ring._pending >= ring.entries:
        return None
    slot = (ring._sq_tail + ring._pending) & ring._sq_mask
    ring._pending += 1
    return _Sqe(ring, slot)


def _buf_address(buf):
    try:
        return ctypes.addressof(ctypes.c_char.from_buffer(buf))
    except (TypeError, BufferError) as exc:
        raise TypeError(
            "prep_read needs a writable contiguous buffer "
            "(mmap, bytearray); got %s: %s" % (type(buf).__name__, exc)
        )


def io_uring_prep_read(sqe, fd, buf, nbytes, offset):
    """Point a reserved slot at an O_DIRECT read. Mirrors the C call."""
    addr = _buf_address(buf)
    ring = sqe._ring
    off = sqe._off
    ring._sqes[off : off + _SQE_BYTES] = b"\0" * _SQE_BYTES
    view = ring._sqes
    struct.pack_into("B", view, off, _IORING_OP_READ)
    struct.pack_into("i", view, off + 4, int(fd))
    struct.pack_into("Q", view, off + 8, int(offset))
    struct.pack_into("Q", view, off + 16, addr)
    struct.pack_into("I", view, off + 24, int(nbytes))


def io_uring_submit_and_wait(ring, wait_nr):
    """Publish new submissions and wait for at least wait_nr completions.

    The fence orders the SQE fills (and the lazy head publish) before
    the tail the kernel is about to read; the second fence orders the
    completion reads below after the kernel's publish. Returns the
    enter count; raises rather than returning a negative errno, because
    the caller loops on completions and a silent -errno would spin.
    """
    n = ring._pending
    _fence()
    _put_u32(ring._sq, ring._sq_off.tail, ring._sq_tail + n)
    _put_u32(ring._cq, ring._cq_off.head, ring._cq_head)
    ring._sq_tail += n
    ring._pending = 0
    nr_enter = _numbers()[1]
    ret = _syscall(
        nr_enter, ring.fd, n, int(wait_nr), _IORING_ENTER_GETEVENTS, 0, 0
    )
    if ret < 0:
        raise OSError(-ret, "io_uring_enter failed")
    _fence()
    return ret


def io_uring_peek_cqe(ring, cqe):
    """Fill cqe with the next completion; -EAGAIN when none is ready."""
    tail = _u32(ring._cq, ring._cq_off.tail)
    if ring._cq_head == tail:
        return -errno.EAGAIN
    off = (ring._cq_head & ring._cq_mask) * _CQE_BYTES + ring._cq_off.cqes
    view = ring._cq
    cqe.user_data = struct.unpack_from("Q", view, off)[0]
    cqe.res = struct.unpack_from("i", view, off + 8)[0]
    return 0


def io_uring_cqe_seen(ring, cqe):
    """Consume one completion; the real head publishes at next submit."""
    ring._cq_head += 1
