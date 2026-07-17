# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Python counterpart of cpp/bench_index_flat.cpp — IndexFlat add and search
# over d/nb/nq/k/metric sweeps.
#
# Timing is handled by the pytest-benchmark harness (warmup included
# automatically).

import faiss
import pytest

from bench_utils import built_index, params, rand_mat
from bench_utils import built_dataset_index, compute_recall, dataset_or_skip

METRICS = {"L2": faiss.METRIC_L2, "IP": faiss.METRIC_INNER_PRODUCT}


# Force the reservoir result handler for small k by setting
# faiss.cvar.distance_compute_min_k_reservoir = 5. Set-and-restore so the
# value does not leak into other benchmark modules in the same pytest session.
@pytest.fixture(autouse=True, scope="module")
def _min_k_reservoir():
    old = faiss.cvar.distance_compute_min_k_reservoir
    faiss.cvar.distance_compute_min_k_reservoir = 5
    yield
    faiss.cvar.distance_compute_min_k_reservoir = old


def metric_type(name):
    if name not in METRICS:
        pytest.fail(f"unknown metric {name!r}; choose from {sorted(METRICS)}")
    return METRICS[name]


@params(metric=["L2", "IP"], d=[16, 32, 64, 128, 256, 768], nb=[10000, 100000])
def bench_flat_add(benchmark, metric, d, nb):
    xb = rand_mat(nb, d)

    def fresh_index():
        return (faiss.IndexFlat(d, metric_type(metric)),), {}

    benchmark.pedantic(
            lambda index: index.add(xb), setup=fresh_index, rounds=3)


@params(
        metric=["L2", "IP"],
        d=[16, 32, 64, 128, 256, 768],
        nb=[10000, 100000],
        nq=[1, 10, 100, 1000, 10000],
        k=[1, 10, 100])
def bench_flat_search(benchmark, metric, d, nb, nq, k):
    index = built_index("Flat", d, nb, metric)
    xq = rand_mat(nq, d, seed=54321)
    benchmark(index.search, xq, k)


# SIFT1M search (skipped when --data-dir is missing), mirroring the
# "sift1m/search" group of cpp/bench_index_flat.cpp (L2 only).
@params(nq=[1, 10, 100], k=[1, 10, 100])
def bench_flat_search_sift1m(benchmark, data_dir, nq, k):
    ds = dataset_or_skip(data_dir)
    index = built_dataset_index("Flat", data_dir)
    _, labels = benchmark(index.search, ds["xq"][:nq], k)

    # Report recall against the dataset ground truth
    benchmark.extra_info["recall"] = compute_recall(labels, ds["gt"][:nq])
