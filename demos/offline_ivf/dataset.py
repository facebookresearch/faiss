# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import numpy as np
import faiss
from typing import List
import random
import logging
from functools import lru_cache


def create_dataset_from_oivf_config(cfg, ds_name):
    normalise = cfg["normalise"] if "normalise" in cfg else False
    return MultiFileVectorDataset(
        cfg["datasets"][ds_name]["root"],
        [
            FileDescriptor(
                f["name"], f["format"], np.dtype(f["dtype"]), f["size"]
            )
            for f in cfg["datasets"][ds_name]["files"]
        ],
        cfg["d"],
        normalise,
        cfg["datasets"][ds_name]["size"],
    )


@lru_cache(maxsize=100)
def _memmap_vecs(
    file_name: str, format: str, dtype: np.dtype, size: int, d: int
) -> np.array:
    """
    If the file is in raw format, the file size will
    be divisible by the dimensionality and by the size
    of the data type.
    Otherwise,the file contains a header and we assume
    it is of .npy type. It the returns the memmapped file.
    """

    assert os.path.exists(file_name), f"file does not exist {file_name}"
    if format == "raw":
        fl = os.path.getsize(file_name)
        nb = fl // d // dtype.itemsize
        assert nb == size, f"{nb} is different than config's {size}"
        assert fl == d * dtype.itemsize * nb  # no header
        return np.memmap(file_name, shape=(nb, d), dtype=dtype, mode="r")
    elif format == "npy":
        vecs = np.load(file_name, mmap_mode="r")
        assert vecs.shape[0] == size, f"size:{size},shape {vecs.shape[0]}"
        assert vecs.shape[1] == d
        assert vecs.dtype == dtype
        return vecs
    else:
        ValueError("The file cannot be loaded in the current format.")


class FileDescriptor:
    def __init__(self, name: str, format: str, dtype: np.dtype, size: int):
        self.name = name
        self.format = format
        self.dtype = dtype
        self.size = size


class MultiFileVectorDataset:
    def __init__(
        self,
        root: str,
        file_descriptors: List[FileDescriptor],
        d: int,
        normalize: bool,
        size: int,
    ):
        assert os.path.exists(root)
        self.root = root
        self.file_descriptors = file_descriptors
        self.d = d
        self.normalize = normalize
        self.size = size
        self.file_offsets = [0]
        t = 0
        for f in self.file_descriptors:
            xb = _memmap_vecs(
                f"{self.root}/{f.name}", f.format, f.dtype, f.size, self.d
            )
            t += xb.shape[0]
            self.file_offsets.append(t)
        assert (
            t == self.size
        ), "the sum of num of embeddings per file!=total num of embeddings"

    def iterate(self, start: int, batch_size: int, dt: np.dtype):
        buffer = np.empty(shape=(batch_size, self.d), dtype=dt)
        rem = 0
        for f in self.file_descriptors:
            if start >= f.size:
                start -= f.size
                continue
            logging.info(f"processing: {f.name}...")
            xb = _memmap_vecs(
                f"{self.root}/{f.name}",
                f.format,
                f.dtype,
                f.size,
                self.d,
            )
            if start > 0:
                xb = xb[start:]
                start = 0
            req = min(batch_size - rem, xb.shape[0])
            buffer[rem:rem + req] = xb[:req]
            rem += req
            if rem == batch_size:
                if self.normalize:
                    faiss.normalize_L2(buffer)
                yield buffer.copy()
                rem = 0
            for i in range(req, xb.shape[0], batch_size):
                j = i + batch_size
                if j <= xb.shape[0]:
                    tmp = xb[i:j].astype(dt)
                    if self.normalize:
                        faiss.normalize_L2(tmp)
                    yield tmp
                else:
                    rem = xb.shape[0] - i
                    buffer[:rem] = xb[i:j]
        if rem > 0:
            tmp = buffer[:rem]
            if self.normalize:
                faiss.normalize_L2(tmp)
            yield tmp

    def get(self, idx: List[int]):
        n = len(idx)
        fidx = np.searchsorted(self.file_offsets, idx, "right")
        res = np.empty(shape=(len(idx), self.d), dtype=np.float32)
        for r, id, fid in zip(range(n), idx, fidx):
            assert fid > 0 and fid <= len(self.file_descriptors), f"{fid}"
            f = self.file_descriptors[fid - 1]
            # deferring normalization until after reading the vec
            vecs = _memmap_vecs(
                f"{self.root}/{f.name}", f.format, f.dtype, f.size, self.d
            )
            i = id - self.file_offsets[fid - 1]
            assert i >= 0 and i < vecs.shape[0]
            res[r, :] = vecs[i]  # TODO: find a faster way
        if self.normalize:
            faiss.normalize_L2(res)
        return res

    def sample(self, n, idx_fn, vecs_fn):
        if vecs_fn and os.path.exists(vecs_fn):
            vecs = np.load(vecs_fn)
            assert vecs.shape == (n, self.d)
            return vecs
        if idx_fn and os.path.exists(idx_fn):
            idx = np.load(idx_fn)
            assert idx.size == n
        else:
            idx = np.array(sorted(random.sample(range(self.size), n)))
            if idx_fn:
                np.save(idx_fn, idx)
        vecs = self.get(idx)
        if vecs_fn:
            np.save(vecs_fn, vecs)
        return vecs

    def get_first_n(self, n, dt):
        assert n <= self.size
        return next(self.iterate(0, n, dt))
