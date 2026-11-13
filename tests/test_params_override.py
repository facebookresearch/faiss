# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np

import faiss

faiss.omp_set_num_threads(4)


d = 32
nb = 1000
nq = 200
k = 10


def make_data(n, seed=123):
    rng = np.random.RandomState(seed)
    return rng.rand(n, d).astype(np.float32)


def make_index(index_type, metric, xb):
    index = faiss.index_factory(d, index_type, metric)
    index.train(xb)
    index.add(xb)
    return index


def search_index(index, xq):
    _, I = index.search(xq, k)
    return I


def search_index_with_params(index, xq, params):
    _, I = faiss.search_with_parameters(index, xq, k, params=params)
    return I


def check_params_override(index_key, metric, seed):
    xb = make_data(nb, seed)
    index = make_index(index_key, metric, xb)
    xq = make_data(nq, seed + 1)
    ps = faiss.ParameterSpace()
    ps.set_index_parameter(index, "nprobe", 2)
    res2ref = search_index(index, xq)
    ps.set_index_parameter(index, "nprobe", 1)

    params = faiss.IVFSearchParameters()
    params.max_codes = 0
    params.nprobe = 2
    res2new = search_index_with_params(index, xq, params)

    return np.array_equal(res2ref, res2new)


def check_ensure_topk_full(index_key, metric, seed):
    xb = make_data(nb, seed)
    index = make_index(index_key, metric, xb)
    xq = make_data(nq, seed + 1)

    # budget smaller than k codes per query: guaranteed truncation when
    # ensure_topk_full is false
    tight_budget = k // 2

    params = faiss.IVFSearchParameters()
    params.nprobe = 32
    params.max_codes = tight_budget
    params.ensure_topk_full = False
    res_tight = search_index_with_params(index, xq, params)

    params.ensure_topk_full = True
    res_full = search_index_with_params(index, xq, params)

    miss_tight = np.count_nonzero(res_tight == -1)
    miss_full = np.count_nonzero(res_full == -1)

    assert miss_tight > 0, "budget should have caused truncation"
    return miss_full <= miss_tight


def check_defaults_preserve_baseline(index_key, metric, seed):
    xb = make_data(nb, seed)
    index = make_index(index_key, metric, xb)
    xq = make_data(nq, seed + 1)

    ps = faiss.ParameterSpace()
    ps.set_index_parameter(index, "nprobe", 4)
    res_no_params = search_index(index, xq)

    params = faiss.IVFSearchParameters()
    params.nprobe = 4
    res_default_params = search_index_with_params(index, xq, params)

    return np.array_equal(res_no_params, res_default_params)


def check_selector(index_key, seed):
    rng = np.random.RandomState(seed)
    xb = rng.rand(nb, d).astype(np.float32)
    xq = rng.rand(nq, d).astype(np.float32)
    ps = faiss.ParameterSpace()

    kept = np.array([i for i in range(nb) if i % 10 == 2], dtype=np.int64)
    sub_xb = xb[kept]

    index = make_index(index_key, faiss.METRIC_L2, xb)
    ps.set_index_parameter(index, "nprobe", 3)

    sub_index = faiss.clone_index(index)
    sub_index.reset()
    sub_index.add_with_ids(sub_xb, kept)

    ref_result = search_index(sub_index, xq)

    params = faiss.IVFSearchParameters()
    params.max_codes = 0
    params.nprobe = 3
    params.sel = faiss.IDSelectorBatch(kept)
    new_result = search_index_with_params(index, xq, params)

    return np.array_equal(ref_result, new_result)


class TestParamsOverride(unittest.TestCase):
    def test_ivf_flat_pp(self):
        self.assertTrue(
            check_params_override("PCA16,IVF32,SQ8", faiss.METRIC_L2, 107))


class TestEnsureTopkFull(unittest.TestCase):
    def test_ivfpq(self):
        self.assertTrue(
            check_ensure_topk_full("IVF32,PQ8np", faiss.METRIC_L2, 203))

    def test_ivfsq(self):
        self.assertTrue(
            check_ensure_topk_full("IVF32,SQ8", faiss.METRIC_L2, 205))


class TestDefaultsPreserveBaseline(unittest.TestCase):
    def test_ivf_flat(self):
        self.assertTrue(
            check_defaults_preserve_baseline(
                "IVF32,Flat", faiss.METRIC_L2, 301))


class TestSelector(unittest.TestCase):
    def test_ivfpq(self):
        self.assertTrue(check_selector("PCA16,IVF32,PQ4x8np", 402))
