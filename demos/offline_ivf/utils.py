# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import os
from typing import Dict
import yaml
import faiss
from faiss.contrib.datasets import SyntheticDataset


def load_config(config):
    assert os.path.exists(config)
    with open(config, "r") as f:
        return yaml.safe_load(f)


def faiss_sanity_check():
    ds = SyntheticDataset(256, 0, 100, 100)
    xq = ds.get_queries()
    xb = ds.get_database()
    index_cpu = faiss.IndexFlat(ds.d)
    index_gpu = faiss.index_cpu_to_all_gpus(index_cpu)
    index_cpu.add(xb)
    index_gpu.add(xb)
    D_cpu, I_cpu = index_cpu.search(xq, 10)
    D_gpu, I_gpu = index_gpu.search(xq, 10)
    assert np.all(I_cpu == I_gpu), "faiss sanity check failed"
    assert np.all(np.isclose(D_cpu, D_gpu)), "faiss sanity check failed"


def margin(sample, idx_a, idx_b, D_a_b, D_a, D_b, k, k_extract, threshold):
    """
    two datasets: xa, xb; n = number of pairs
    idx_a - (np,) - query vector ids in xa
    idx_b - (np,) - query vector ids in xb
    D_a_b - (np,) - pairwise distances between xa[idx_a] and xb[idx_b]
    D_a - (np, k) - distances between vectors xa[idx_a] and corresponding nearest neighbours in xb
    D_b - (np, k) - distances between vectors xb[idx_b] and corresponding nearest neighbours in xa
    k - k nearest neighbours used for margin
    k_extract - number of nearest neighbours of each query in xb we consider for margin calculation and filtering
    threshold - margin threshold
    """

    n = sample
    nk = n * k_extract
    assert idx_a.shape == (n,)
    idx_a_k = idx_a.repeat(k_extract)
    assert idx_a_k.shape == (nk,)
    assert idx_b.shape == (nk,)
    assert D_a_b.shape == (nk,)
    assert D_a.shape == (n, k)
    assert D_b.shape == (nk, k)
    mean_a = np.mean(D_a, axis=1)
    assert mean_a.shape == (n,)
    mean_a_k = mean_a.repeat(k_extract)
    assert mean_a_k.shape == (nk,)
    mean_b = np.mean(D_b, axis=1)
    assert mean_b.shape == (nk,)
    margin = 2 * D_a_b / (mean_a_k + mean_b)
    above_threshold = margin > threshold
    print(np.count_nonzero(above_threshold))
    print(idx_a_k[above_threshold])
    print(idx_b[above_threshold])
    print(margin[above_threshold])
    return margin


def add_group_args(group, *args, **kwargs):
    return group.add_argument(*args, **kwargs)


def get_intersection_cardinality_frequencies(
    I: np.ndarray, I_gt: np.ndarray
) -> Dict[int, int]:
    """
    Computes the frequencies for the cardinalities of the intersection of neighbour indices.
    """
    nq = I.shape[0]
    res = []
    for ell in range(nq):
        res.append(len(np.intersect1d(I[ell, :], I_gt[ell, :])))
    values, counts = np.unique(res, return_counts=True)
    return dict(zip(values, counts))


def is_pretransform_index(index):
    if index.__class__ == faiss.IndexPreTransform:
        assert hasattr(index, "chain")
        return True
    else:
        assert not hasattr(index, "chain")
        return False
