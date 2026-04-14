# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import sys
import numpy as np
import os

"""
I/O functions in fvecs, bvecs, ivecs formats
definition of the formats here: http://corpus-texmex.irisa.fr/
"""


def ivecs_read(fname):
    a = np.fromfile(fname, dtype='int32')
    if sys.byteorder == 'big':
        a.byteswap(inplace=True)
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    return ivecs_read(fname).view('float32')


def ivecs_mmap(fname):
    assert sys.byteorder != 'big'
    a = np.memmap(fname, dtype='int32', mode='r')
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:]


def fvecs_mmap(fname):
    return ivecs_mmap(fname).view('float32')


def bvecs_mmap(fname):
    x = np.memmap(fname, dtype='uint8', mode='r')
    if sys.byteorder == 'big':
        da = x[:4][::-1].copy()
        d = da.view('int32')[0]
    else:
        d = x[:4].view('int32')[0]
    return x.reshape(-1, d + 4)[:, 4:]


def ivecs_write(fname, m):
    n, d = m.shape
    m1 = np.empty((n, d + 1), dtype='int32')
    m1[:, 0] = d
    m1[:, 1:] = m
    if sys.byteorder == 'big':
        m1.byteswap(inplace=True)
    m1.tofile(fname)


def fvecs_write(fname, m):
    m = m.astype('float32')
    ivecs_write(fname, m.view('int32'))

def bvecs_iter(filepath, batch_size=100_000):
    """
    Memory-mapped iterator - only loads requested slices into RAM
    """

    file_size = os.path.getsize(filepath)
    with open(filepath, 'rb') as f:
        dim = np.frombuffer(f.read(4), dtype='<i4')[0]

    bytes_per_vec = 4 + dim
    n_vectors = file_size // bytes_per_vec

    mm = np.memmap(filepath, mode='r', dtype=np.uint8)
    records = mm.reshape(n_vectors, bytes_per_vec)

    for start in range(0, n_vectors, batch_size):
        end = np.min([start + batch_size, n_vectors])
        yield records[start:end, 4:]

def bvecs_iter_chunked(chunk_folder, batch_size=100_000):
    """
    Memory-mapped iterator over chunked .bvecs files.
    Iterates through all chunk files in order (chunk_0000.bvecs, chunk_0001.bvecs, etc.)
    and yields batches of vectors, handling cases where batches span multiple files.

    Args:
        chunk_folder: path to folder containing chunk_XXXX.bvecs files
        batch_size: number of vectors to yield per batch

    Yields:
        numpy array of shape (batch_size, d) or smaller for last batch

    Raises:
        ValueError: if there are gaps in the chunk sequence
    """

    # Find all chunk files and sort them
    chunk_files = []
    for entry in os.scandir(chunk_folder):
        if entry.is_file() and entry.name.startswith("chunk_") and entry.name.endswith(".bvecs"):
            chunk_files.append(entry.path)
    chunk_files.sort()

    if not chunk_files:
        raise ValueError(f"No chunk files found in {chunk_folder}")

    # Extract chunk numbers and verify no gaps
    chunk_numbers = []
    for path in chunk_files:
        basename = os.path.basename(path)
        try:
            num_str = basename.split('_')[1].split('.')[0]
            chunk_numbers.append(int(num_str))
        except (IndexError, ValueError):
            raise ValueError(f"Invalid chunk filename format: {basename}")

    # Check for gaps in sequence
    expected_chunks = list(range(len(chunk_numbers)))
    if sorted(chunk_numbers) != expected_chunks:
        missing = set(expected_chunks) - set(chunk_numbers)
        raise ValueError(
            f"Gap detected in chunk sequence! Missing chunks: {sorted(missing)}\n"
            f"Found chunks: {sorted(chunk_numbers)}\n"
            f"Expected continuous sequence from 0 to {len(chunk_numbers)-1}"
        )

    # Get dimension from first chunk
    with open(chunk_files[0], 'rb') as f:
        dim = np.frombuffer(f.read(4), dtype='<i4')[0]

    bytes_per_vec = 4 + dim

    # Buffer to accumulate vectors across chunk boundaries
    buffer = None
    buffer_size = 0

    # Iterate through each chunk file
    for chunk_path in chunk_files:
        file_size = os.path.getsize(chunk_path)
        n_vectors = file_size // bytes_per_vec

        # Memory-map this chunk
        mm = np.memmap(chunk_path, mode='r', dtype=np.uint8)
        records = mm.reshape(n_vectors, bytes_per_vec)
        vectors = records[:, 4:]  # Skip dimension prefix

        start = 0

        # First, handle any buffered data from previous chunk
        if buffer is not None:
            needed = batch_size - buffer_size
            if needed <= n_vectors:
                # Can complete the buffered batch with this chunk
                batch = np.vstack([buffer, vectors[:needed]])
                yield batch
                buffer = None
                buffer_size = 0
                start = needed
            else:
                # Still not enough, accumulate and continue to next chunk
                buffer = np.vstack([buffer, vectors])
                buffer_size += n_vectors
                continue

        # Now process complete batches from this chunk
        while start + batch_size <= n_vectors:
            yield vectors[start:start + batch_size]
            start += batch_size

        remainder = n_vectors - start
        if remainder > 0:
            buffer = vectors[start:].copy()
            buffer_size = remainder

    if buffer is not None:
        yield buffer
