# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

# Python counterpart of cpp/bench_index_ivf_factory.cpp — the generic index_factory
# train / add / search harness. Builds an arbitrary index_factory index, times
# train and add, sweeps search parameters reporting recall@1/10/100, ndis and
# ms/query, and optionally runs a ParameterSpace autotune exploration. This
# mirrors that:
#   * bench_factory_train / bench_factory_add time building any factory key
#   * bench_factory_search times search over an nprobe sweep and reports
#     recall_at_1 / recall_at_10 / recall_at_100 against exact ground truth
#   * bench_factory_autotune runs ParameterSpace.explore and records the best
#     operating point
#
# CLI-overridable options:
#   * metric axis (L2 default, IP available). Threaded through index_factory
#     and ground-truth computation.
#   * search reports recall_at_1 / recall_at_10 / recall_at_100.
#   * INTER selects IntersectionCriterion instead of OneRecallAtRCriterion
#     in autotune. Default keeps OneRecallAtRCriterion.
#   * autotune defaults: ps.n_experiments = N_AUTOTUNE = 500,
#     ps.min_test_duration = MIN_TEST_DURATION = 3.0.
#   * construction knobs applied to the built index where the type supports
#     them (each guarded so unsupported keys are left unchanged):
#     BY_RESIDUAL, NO_PRECOMPUTED_TABLES, RQ_BEAM_SIZE (max_beam_size),
#     LSQ_ENCODE_ILS_ITERS (encode_ils_iters), and ADD_BS (batched add).
#
# Design notes:
#   * search threads default to 1 (single-threaded timing).
#   * knobs not exposed: --RQ_use_beam_LUT, --RQ_train_default,
#     --clustering_niter, --autotune_max / --autotune_range, arbitrary
#     --searchparams lists, and add-time quantizer tuning (IndexRefine
#     k_factor and quantizer nprobe/efSearch boosts).

import functools

import faiss

from bench_utils import (
    built_dataset_index,
    compute_recall,
    dataset_or_skip,
    ground_truth,
    params,
    rand_mat,
    require_attr,
)

D = 64
NB = 100_000
NT = 65_536  # 256 * 256
NQ = 1_000
K = 100

# Autotune defaults.
N_AUTOTUNE = 500
MIN_TEST_DURATION = 3.0

# Autotune criterion: False -> OneRecallAtRCriterion (default), True ->
# IntersectionCriterion.
INTER = False

# Construction knobs applied to the built index before train/add. Set to a
# non-default value to enable; each is applied only where the index type
# supports it (guarded by hasattr / downcast).
BY_RESIDUAL = -1  # 0/1 to force IVF by_residual; -1 leaves default
NO_PRECOMPUTED_TABLES = False  # disable IVFPQ precomputed tables
RQ_BEAM_SIZE = -1  # RQ/AQ max_beam_size; -1 leaves default
LSQ_ENCODE_ILS_ITERS = -1  # LSQ encode_ils_iters; -1 leaves default
ADD_BS = -1  # add in batches of this size; -1 adds all at once


def _metric_type(metric):
    return faiss.METRIC_INNER_PRODUCT if metric == "IP" else faiss.METRIC_L2


def _apply_construction_options(index):
    """Apply construction knobs where the index type supports them.

    Each knob is guarded so factory keys that do not expose it are unchanged.
    """
    ivf = faiss.try_extract_index_ivf(index)
    if BY_RESIDUAL != -1 and ivf is not None and hasattr(ivf, "by_residual"):
        ivf.by_residual = BY_RESIDUAL == 1
    if NO_PRECOMPUTED_TABLES and ivf is not None and hasattr(
            ivf, "use_precomputed_table"):
        try:
            ivf.use_precomputed_table = 0
        except Exception:
            pass
    if RQ_BEAM_SIZE != -1 and hasattr(index, "rq"):
        index.rq.max_beam_size = RQ_BEAM_SIZE
    if LSQ_ENCODE_ILS_ITERS != -1 and hasattr(index, "lsq"):
        index.lsq.encode_ils_iters = LSQ_ENCODE_ILS_ITERS


def _add(index, xb):
    """Add xb, optionally in ADD_BS-sized batches."""
    if ADD_BS <= 0:
        index.add(xb)
        return
    for i0 in range(0, len(xb), ADD_BS):
        index.add(xb[i0:i0 + ADD_BS])

# The default set of index_factory keys benchmarked, mirroring the cpp sweep.
FACTORY_KEYS = [
    "IVF256,Flat",
    "IVF256,PQ8",
    "IVF256,SQ8",
    "IVF1024,Flat",
]


@functools.lru_cache(maxsize=8)
def _built_index(factory, d, nb, nt, metric="L2"):
    """Trained + populated index for the search / autotune sweeps."""
    index = faiss.index_factory(d, factory, _metric_type(metric))
    _apply_construction_options(index)
    xb = rand_mat(nb, d)
    index.train(xb[: min(nb, nt)])
    _add(index, xb)
    return index


@params(factory=FACTORY_KEYS, metric=["L2"])
def bench_factory_train(benchmark, factory, metric):
    d, nt = D, NT
    xt = rand_mat(nt, d)

    def fresh():
        index = faiss.index_factory(d, factory, _metric_type(metric))
        _apply_construction_options(index)
        return (index,), {}

    # rounds=1: training a fresh index each round is expensive.
    benchmark.pedantic(lambda index: index.train(xt), setup=fresh, rounds=1)


@params(factory=FACTORY_KEYS, metric=["L2"])
def bench_factory_add(benchmark, factory, metric):
    d, nb, nt = D, NB, NT
    xb = rand_mat(nb, d)
    trained = faiss.index_factory(d, factory, _metric_type(metric))
    _apply_construction_options(trained)
    trained.train(xb[: min(nb, nt)])

    def fresh():
        # fresh trained-but-empty index each round: measure pure add cost
        return (faiss.clone_index(trained),), {}

    benchmark.pedantic(lambda index: _add(index, xb), setup=fresh, rounds=3)


@params(factory=FACTORY_KEYS, nprobe=[1, 4, 16, 64], metric=["L2"])
def bench_factory_search(benchmark, factory, nprobe, metric):
    d, nb, nt = D, NB, NT
    index = _built_index(factory, d, nb, nt, metric)
    ivf = faiss.try_extract_index_ivf(index)
    if ivf is not None:
        if nprobe > ivf.nlist:
            import pytest

            pytest.skip(f"nprobe {nprobe} > nlist {ivf.nlist}")
        ivf.nprobe = nprobe
    xq = rand_mat(NQ, d, seed=54321)
    benchmark(index.search, xq, K)

    # Report recall against exact ground truth: R@1 / R@10 / R@100
    # (capped at K) plus the aggregate recall@K.
    _, I = index.search(xq, K)
    gt_I = ground_truth(d, nb, NQ, K, metric=metric)
    benchmark.extra_info["recall"] = compute_recall(I, gt_I)
    for rank in (1, 10, 100):
        r = min(rank, K)
        # fraction of queries whose true top-1 neighbor is in the top-r results
        hits = (I[:, :r] == gt_I[:, :1]).sum() / float(NQ)
        benchmark.extra_info[f"recall_at_{rank}"] = float(hits)


# SIFT1M search (skipped when --data-dir is missing). cpp/bench_index_ivf_factory.cpp
# runs the whole factory harness on SIFT1M when the dataset is present; this
# mirrors its search sweep — the same factory keys over the nprobe sweep on
# the real data, reporting recall / recall_at_r against the dataset ground
# truth. Searches the first NQ dataset queries at k=K.
@params(factory=FACTORY_KEYS, nprobe=[1, 4, 16, 64])
def bench_factory_search_sift1m(benchmark, data_dir, factory, nprobe):
    ds = dataset_or_skip(data_dir)
    index = built_dataset_index(factory, data_dir)
    ivf = faiss.try_extract_index_ivf(index)
    if ivf is not None:
        if nprobe > ivf.nlist:
            import pytest

            pytest.skip(f"nprobe {nprobe} > nlist {ivf.nlist}")
        ivf.nprobe = nprobe
    xq = ds["xq"][:NQ]
    gt = ds["gt"][:NQ]
    _, I = benchmark(index.search, xq, K)

    # Report recall against the dataset ground truth: R@1 / R@10 / R@100
    # (capped at K) plus the aggregate recall@K.
    benchmark.extra_info["recall"] = compute_recall(I, gt)
    for rank in (1, 10, 100):
        r = min(rank, K)
        # fraction of queries whose true top-1 neighbor is in the top-r results
        hits = (I[:, :r] == gt[:, :1]).sum() / float(len(xq))
        benchmark.extra_info[f"recall_at_{rank}"] = float(hits)


@params(factory=FACTORY_KEYS, metric=["L2"])
def bench_factory_autotune(benchmark, factory, metric):
    """ParameterSpace exploration — the autotune search path."""
    require_attr(faiss, "ParameterSpace")
    crit_name = (
        "IntersectionCriterion" if INTER else "OneRecallAtRCriterion")
    require_attr(faiss, crit_name)
    d, nb, nt = D, NB, NT
    index = _built_index(factory, d, nb, nt, metric)
    xq = rand_mat(NQ, d, seed=54321)
    gt_I = ground_truth(d, nb, NQ, K, metric=metric).astype("int64")

    ps = faiss.ParameterSpace()
    ps.initialize(index)
    ps.n_experiments = N_AUTOTUNE
    ps.min_test_duration = MIN_TEST_DURATION

    if INTER:
        crit = faiss.IntersectionCriterion(NQ, K)
    else:
        crit = faiss.OneRecallAtRCriterion(NQ, 1)
    crit.nnn = K
    crit.set_groundtruth(None, gt_I)

    # The exploration itself is the timed work (it runs many searches).
    result = {}

    def run():
        op = ps.explore(index, xq, crit)
        opv = op.optimal_pts
        best_perf, best_t = 0.0, 0.0
        for i in range(opv.size()):
            pt = opv.at(i)
            if pt.perf > best_perf:
                best_perf, best_t = pt.perf, pt.t
        result["best_inter" if INTER else "best_recall_at_1"] = best_perf
        result["best_op_ms"] = best_t

    benchmark.pedantic(run, rounds=1)
    benchmark.extra_info.update(result)
