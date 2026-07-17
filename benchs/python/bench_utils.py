# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Shared helpers for the pytest-based faiss benchmark suite.

Benchmark functions declare their parameter sweeps with the `params`
decorator; conftest.py turns each sweep into a pytest parametrization and
lets the user override any of them from the command line with a
comma-separated list (e.g. ``--d=128,256``), mirroring the gflags interface
of the C++ suite in ../cpp.
"""

import functools
import os

import numpy as np

# Attribute name read by conftest.pytest_generate_tests.
SWEEPS_ATTR = "_bench_sweeps"


def params(**sweeps):
    """Declare default parameter sweeps for a benchmark function.

    Example:
        @params(d=[128, 256], nq=[1, 10, 100])
        def bench_flat_search(benchmark, d, nq): ...

    Each keyword becomes a pytest parameter whose default values are the
    given list; the user can override every parameter globally with
    ``--<name>=v1,v2,...`` on the pytest command line.
    """

    def deco(fn):
        setattr(fn, SWEEPS_ATTR, dict(sweeps))
        return fn

    return deco


@functools.lru_cache(maxsize=8)
def rand_mat(n, d, seed=12345):
    """Cached (n, d) float32 matrix of uniform [0, 1) values.

    Callers must treat the returned array as read-only — it is shared
    between benchmarks.
    """
    rng = np.random.default_rng(seed)
    return rng.random((n, d), dtype=np.float32)


@functools.lru_cache(maxsize=8)
def rand_codes(n, code_size, seed=12345):
    """Cached (n, code_size) uint8 matrix (for binary indexes / codes)."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(n, code_size), dtype=np.uint8)


@functools.lru_cache(maxsize=4)
def built_index(factory, d, nb, metric="L2"):
    """Cached trained+populated index for search benchmarks.

    Keyed by (factory string, d, nb, metric) so parameter sweeps over
    nq/k/nprobe/efSearch reuse one build. Search-only — callers must not
    mutate the returned index. Build/add benchmarks construct their own
    fresh indexes instead of using this.
    """
    import faiss

    metric_type = (
            faiss.METRIC_INNER_PRODUCT if metric == "IP" else faiss.METRIC_L2)
    index = faiss.index_factory(d, factory, metric_type)
    xb = rand_mat(nb, d)
    if not index.is_trained:
        index.train(xb)
    index.add(xb)
    return index


def ivecs_read(fname):
    """Read an .ivecs file into an (n, d) int32 array."""
    a = np.fromfile(fname, dtype="int32")
    d = a[0]
    return a.reshape(-1, d + 1)[:, 1:].copy()


def fvecs_read(fname):
    """Read an .fvecs file into an (n, d) float32 array."""
    return ivecs_read(fname).view("float32")


@functools.lru_cache(maxsize=2)
def load_dataset(data_dir):
    """Load a SIFT1M-layout dataset directory (see cpp/bench_dataset_utils.h).

    Expects sift_learn.fvecs, sift_base.fvecs, sift_query.fvecs and
    sift_groundtruth.ivecs under ``data_dir`` (the --data_dir pytest option).
    Returns a dict with float32 arrays xt/xb/xq and int32 array gt, or None
    when the directory or any file is missing, in which case callers skip
    their dataset benchmarks (the synthetic sweeps are unaffected).
    """
    files = {
        "xt": "sift_learn.fvecs",
        "xb": "sift_base.fvecs",
        "xq": "sift_query.fvecs",
        "gt": "sift_groundtruth.ivecs",
    }
    paths = {k: os.path.join(data_dir, v) for k, v in files.items()}
    if not all(os.path.isfile(p) for p in paths.values()):
        return None
    return {
        k: (ivecs_read(p) if k == "gt" else fvecs_read(p))
        for k, p in paths.items()
    }


def dataset_or_skip(data_dir):
    """load_dataset(), skipping the current benchmark when unavailable."""
    import pytest

    ds = load_dataset(data_dir)
    if ds is None:
        pytest.skip(f"dataset not found in {data_dir!r}")
    return ds


@functools.lru_cache(maxsize=4)
def built_dataset_index(factory, data_dir, metric="L2"):
    """Cached index trained on and populated with the on-disk dataset.

    Same contract as built_index: search-only, callers must not mutate the
    returned index. Callers are responsible for checking dataset
    availability first (dataset_or_skip).
    """
    import faiss

    ds = load_dataset(data_dir)
    metric_type = (
            faiss.METRIC_INNER_PRODUCT if metric == "IP" else faiss.METRIC_L2)
    index = faiss.index_factory(ds["xb"].shape[1], factory, metric_type)
    if not index.is_trained:
        index.train(ds["xt"])
    index.add(ds["xb"])
    return index


def require_attr(faiss_module, name):
    """Skip the current test if this faiss build lacks `name`."""
    import pytest

    if not hasattr(faiss_module, name):
        pytest.skip(f"faiss build has no {name}")


@functools.lru_cache(maxsize=16)
def ground_truth(d, nb, nq, k, metric="L2", xb_seed=12345, xq_seed=54321):
    """Compute exact k-NN ground truth (cached).

    Returns an (nq, k) int64 array of neighbor IDs.
    """
    import faiss

    metric_type = (
            faiss.METRIC_INNER_PRODUCT if metric == "IP" else faiss.METRIC_L2)
    index = faiss.IndexFlat(d, metric_type)
    index.add(rand_mat(nb, d, seed=xb_seed))
    _, gt_I = index.search(rand_mat(nq, d, seed=xq_seed), k)
    return gt_I


def compute_recall(labels, gt_labels):
    """Compute recall@k: average fraction of true top-k found in results.

    Parameters
    ----------
    labels : ndarray of shape (nq, k)
        Search result IDs from the approximate index.
    gt_labels : ndarray of shape (nq, k') where k' >= k
        Ground truth IDs from exact search.

    Returns
    -------
    float in [0, 1]
    """
    k = labels.shape[1]
    gt_k = gt_labels.shape[1]
    eval_k = min(k, gt_k)
    # For each query, count how many of the top-eval_k GT neighbors appear
    # in the result set of size k.
    n_found = 0
    for i in range(labels.shape[0]):
        result_set = set(labels[i].tolist())
        for j in range(eval_k):
            if gt_labels[i, j] in result_set:
                n_found += 1
    return n_found / (labels.shape[0] * eval_k)

