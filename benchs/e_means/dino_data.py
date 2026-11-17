# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Bounded reads of the DINO bvecs corpus."""

from pathlib import Path

import numpy as np


class DinoQueries:
    """The complete held-out DINO10B query set, read in bounded batches."""

    def __init__(self, root, d, n=100_000):
        path = Path(root) / "queries_clean.bvecs"
        if not path.exists():
            raise FileNotFoundError(
                f"download the DINO10B test queries to {path}"
            )
        if path.stat().st_size != n * (d + 4):
            raise ValueError(f"expected {n} {d}-dimensional queries in {path}")
        self.mapping = np.memmap(
            path, dtype=np.uint8, mode="r", shape=(n, d + 4)
        )
        headers = self.mapping[:, :4].copy().view("<i4").ravel()
        if not np.all(headers == d):
            raise ValueError(f"invalid bvecs dimension headers in {path}")
        self.n = n

    def batches(self, mean, scale, batch_size):
        import torch

        if batch_size < 1:
            raise ValueError("query batch size must be positive")
        for begin in range(0, self.n, batch_size):
            rows = torch.from_numpy(
                self.mapping[begin : begin + batch_size, 4:].copy()
            ).to(device=mean.device, dtype=torch.float32)
            yield (rows - mean) / scale


def query_mse(model, batches):
    """Exact MSE over identical query batches on every centroid owner.

    Evaluate current epoch centroids or finalized means without updating the
    trainer. IVF training still uses exhaustive FP32 query assignment.
    """
    import torch
    import torch.distributed as dist

    from faiss.contrib.e_means import Emeans

    with torch.no_grad():
        centroids = (
            model.centroids if model.centroids is not None else model._centroids
        )
        evaluator = Emeans(
            model.d,
            model.k,
            assignment_backend=model.assignment_backend,
            centroid_chunk_size=model.centroid_chunk_size,
        )
        norms = centroids.square().sum(1)
        total = centroids.new_zeros((), dtype=torch.float64)
        n = 0
        for rows in batches:
            _, labels = evaluator._assign_rows(rows, centroids, norms)
            distances = (rows - centroids[labels.long()]).square().sum(1)
            dist.all_reduce(distances, op=dist.ReduceOp.MIN, group=model._group)
            total += distances.sum(dtype=torch.float64)
            n += len(rows)
        if not n:
            raise ValueError("held-out evaluation requires queries")
        return total.item() / n


class DinoBvecs:
    def __init__(self, root, n):
        directory = Path(root) / "chunked_base_10B"
        self.paths = sorted(
            p for p in directory.iterdir() if p.suffix == ".bvecs"
        )
        if not self.paths:
            raise ValueError(f"no bvecs chunks in {directory}")
        with self.paths[0].open("rb") as stream:
            self.d = int(np.frombuffer(stream.read(4), dtype="<i4")[0])
        if self.d <= 0:
            raise ValueError("invalid bvecs dimension")
        self.row_bytes = self.d + 4
        lengths = []
        self.maps = []
        for index, path in enumerate(self.paths):
            if sum(lengths) < n:
                if (
                    self.paths[0].match("chunk_[0-9][0-9][0-9][0-9].bvecs")
                    and path.name != f"chunk_{index:04d}.bvecs"
                ):
                    raise ValueError(f"missing chunk in required DINO prefix: {path}")
                if Path(str(path) + ".parts.json").exists():
                    raise ValueError(f"unfinished DINO download: {path}")
            size = path.stat().st_size
            if size % self.row_bytes:
                raise ValueError(f"incomplete bvecs chunk: {path}")
            lengths.append(size // self.row_bytes)
            self.maps.append(
                np.memmap(
                    path,
                    dtype=np.uint8,
                    mode="r",
                    shape=(lengths[-1], self.row_bytes),
                )
            )
        self.ends = np.cumsum(lengths)
        if not 0 < n <= self.ends[-1]:
            raise ValueError(
                f"requested {n} rows; corpus contains {self.ends[-1]}"
            )
        self.n = n

    def read(self, begin, end):
        if not 0 <= begin <= end <= self.n:
            raise IndexError("DINO row range out of bounds")
        output = np.empty((end - begin, self.d), dtype=np.uint8)
        at = 0
        for mapping, hi, lo in zip(
            self.maps, self.ends, np.r_[0, self.ends[:-1]]
        ):
            a, b = max(begin, int(lo)), min(end, int(hi))
            if a < b:
                output[at : at + b - a] = mapping[a - lo : b - lo, 4:]
                at += b - a
        return output

    def resident(self, begin, end, device):
        import torch

        output = torch.empty((end - begin, self.d), device=device)
        for a in range(begin, end, 65536):
            b = min(end, a + 65536)
            output[a - begin : b - begin] = torch.from_numpy(
                self.read(a, b)
            ).to(device)
        return output
