# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import torch
import torch.distributed

import faiss

import faiss.contrib.torch_utils
from faiss.contrib.torch import clustering
from faiss.contrib import datasets


class DatasetAssignDistributedGPU(clustering.DatasetAssign):
    """
    There is one instance per worker, each worker has a dataset shard.
    The non-master workers do not run through the k-means function, so some
    code has run it to keep the workers in sync.
    """

    def __init__(self, res, x, rank, nproc):
        clustering.DatasetAssign.__init__(self, x)
        self.res = res
        self.rank = rank
        self.nproc = nproc
        self.device = x.device

        n = len(x)
        sizes = torch.zeros(nproc, device=self.device, dtype=torch.int64)
        sizes[rank] = n
        torch.distributed.all_gather(
            [sizes[i:i + 1] for i in range(nproc)], sizes[rank:rank + 1])
        self.sizes = sizes.cpu().numpy()

        # begin & end of each shard
        self.cs = np.zeros(nproc + 1, dtype='int64')
        self.cs[1:] = np.cumsum(self.sizes)

    def count(self):
        return int(self.sizes.sum())

    def int_to_slaves(self, i):
        " broadcast an int to all workers "
        rank = self.rank
        tab = torch.zeros(1, device=self.device, dtype=torch.int64)
        if rank == 0:
            tab[0] = i
        else:
            assert i is None
        torch.distributed.broadcast(tab, 0)
        return tab.item()

    def get_subset(self, indices):
        rank = self.rank
        assert rank == 0 or indices is None

        len_indices = self.int_to_slaves(len(indices) if rank == 0 else None)

        if rank == 0:
            indices = torch.from_numpy(indices).to(self.device)
        else:
            indices = torch.zeros(
                len_indices, dtype=torch.int64, device=self.device)
        torch.distributed.broadcast(indices, 0)

        # select subset of indices

        i0, i1 = self.cs[rank], self.cs[rank + 1]

        mask = torch.logical_and(indices < i1, indices >= i0)
        output = torch.zeros(
            len_indices, self.x.shape[1],
            dtype=self.x.dtype, device=self.device)
        output[mask] = self.x[indices[mask] - i0]
        torch.distributed.reduce(output, 0)  # sum
        if rank == 0:
            return output
        else:
            return None

    def perform_search(self, centroids):
        assert False, "shoudl not be called"

    def assign_to(self, centroids, weights=None):
        assert weights is None

        rank, nproc = self.rank, self.nproc
        assert rank == 0 or centroids is None
        nc = self.int_to_slaves(len(centroids) if rank == 0 else None)

        if rank != 0:
            centroids = torch.zeros(
                nc, self.x.shape[1], dtype=self.x.dtype, device=self.device)
        torch.distributed.broadcast(centroids, 0)

        # perform search
        D, I = faiss.knn_gpu(
            self.res, self.x, centroids, 1, device=self.device.index)

        I = I.ravel()
        D = D.ravel()

        sum_per_centroid = torch.zeros_like(centroids)
        if weights is None:
            sum_per_centroid.index_add_(0, I, self.x)
        else:
            sum_per_centroid.index_add_(0, I, self.x * weights[:, None])

        torch.distributed.reduce(sum_per_centroid, 0)

        if rank == 0:
            # gather deos not support tensors of different sizes
            # should be implemented with point-to-point communication
            assert np.all(self.sizes == self.sizes[0])
            device = self.device
            all_I = torch.zeros(self.count(), dtype=I.dtype, device=device)
            all_D = torch.zeros(self.count(), dtype=D.dtype, device=device)
            torch.distributed.gather(
                I, [all_I[self.cs[r]:self.cs[r + 1]] for r in range(nproc)],
                dst=0,
            )
            torch.distributed.gather(
                D, [all_D[self.cs[r]:self.cs[r + 1]] for r in range(nproc)],
                dst=0,
            )
            return all_I.cpu().numpy(), all_D, sum_per_centroid
        else:
            torch.distributed.gather(I, None, dst=0)
            torch.distributed.gather(D, None, dst=0)
            return None


if __name__ == "__main__":

    torch.distributed.init_process_group(
        backend="nccl",
    )
    rank = torch.distributed.get_rank()
    nproc = torch.distributed.get_world_size()

    # current version does only support shards of the same size
    ds = datasets.SyntheticDataset(32, 10000, 0, 0, seed=1234 + rank)
    x = ds.get_train()

    device = torch.device(f"cuda:{rank}")

    torch.cuda.set_device(device)
    x = torch.from_numpy(x).to(device)
    res = faiss.StandardGpuResources()

    da = DatasetAssignDistributedGPU(res, x, rank, nproc)

    k = 1000
    niter = 25

    if rank == 0:
        print(f"sizes = {da.sizes}")
        centroids, iteration_stats = clustering.kmeans(
            k, da, niter=niter, return_stats=True)
        print("clusters:", centroids.cpu().numpy())
    else:
        # make sure the iterations are aligned with master
        da.get_subset(None)

        for _ in range(niter):
            da.assign_to(None)

    torch.distributed.barrier()
    print("Done")
