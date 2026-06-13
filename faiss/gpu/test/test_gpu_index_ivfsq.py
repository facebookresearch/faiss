# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import print_function
import unittest
import numpy as np
import faiss


def make_t(num, d, clamp=False):
    rs = np.random.RandomState(123)
    x = rs.rand(num, d).astype('float32')
    if clamp:
        x = (x * 255).astype('uint8').astype('float32')
    return x


def make_indices_copy_from_cpu(nlist, d, qtype, by_residual, metric, clamp):
    to_train = make_t(10000, d, clamp)

    quantizer_cp = faiss.IndexFlat(d, metric)
    idx_cpu = faiss.IndexIVFScalarQuantizer(
        quantizer_cp, d, nlist, qtype, metric, by_residual)

    idx_cpu.train(to_train)
    idx_cpu.add(to_train)

    res = faiss.StandardGpuResources()
    res.noTempMemory()
    config = faiss.GpuIndexIVFScalarQuantizerConfig()
    config.use_cuvs = False
    idx_gpu = faiss.GpuIndexIVFScalarQuantizer(res, idx_cpu, config)

    return idx_cpu, idx_gpu


def make_indices_copy_from_gpu(nlist, d, qtype, by_residual, metric, clamp):
    to_train = make_t(10000, d, clamp)

    res = faiss.StandardGpuResources()
    res.noTempMemory()
    config = faiss.GpuIndexIVFScalarQuantizerConfig()
    config.use_cuvs = False
    idx_gpu = faiss.GpuIndexIVFScalarQuantizer(
        res, d, nlist, qtype, metric, by_residual, config)
    idx_gpu.train(to_train)
    idx_gpu.add(to_train)

    quantizer_cp = faiss.IndexFlat(d, metric)
    idx_cpu = faiss.IndexIVFScalarQuantizer(
        quantizer_cp, d, nlist, qtype, metric, by_residual)
    idx_gpu.copyTo(idx_cpu)

    return idx_cpu, idx_gpu


def make_indices_train(nlist, d, qtype, by_residual, metric, clamp):
    to_train = make_t(10000, d, clamp)

    quantizer_cp = faiss.IndexFlat(d, metric)
    idx_cpu = faiss.IndexIVFScalarQuantizer(
        quantizer_cp, d, nlist, qtype, metric, by_residual)
    assert by_residual == idx_cpu.by_residual

    idx_cpu.train(to_train)
    idx_cpu.add(to_train)

    res = faiss.StandardGpuResources()
    res.noTempMemory()
    config = faiss.GpuIndexIVFScalarQuantizerConfig()
    config.use_cuvs = False
    idx_gpu = faiss.GpuIndexIVFScalarQuantizer(
        res, d, nlist, qtype, metric, by_residual, config)
    assert by_residual == idx_gpu.by_residual

    idx_gpu.train(to_train)
    idx_gpu.add(to_train)

    return idx_cpu, idx_gpu

#
# Testing functions
#


def summarize_results(dist, idx):
    valid = []
    invalid = []
    for query in range(dist.shape[0]):
        valid_sub = {}
        invalid_sub = []

        for order, (d, i) in enumerate(zip(dist[query], idx[query])):
            if i == -1:
                invalid_sub.append(order)
            else:
                valid_sub[i] = [order, d]

        valid.append(valid_sub)
        invalid.append(invalid_sub)

    return valid, invalid


def compare_results(d1, i1, d2, i2):
    # Count number of index differences
    idx_diffs = {}
    idx_diffs_inf = 0
    idx_invalid = 0

    valid1, invalid1 = summarize_results(d1, i1)
    valid2, invalid2 = summarize_results(d2, i2)

    # Invalid results should be the same for both
    # (except if we happen to hit different centroids)
    for inv1, inv2 in zip(invalid1, invalid2):
        if (len(inv1) != len(inv2)):
            print('mismatch ', len(inv1), len(inv2), inv2[0])

        assert len(inv1) == len(inv2)
        idx_invalid += len(inv2)
        for x1, x2 in zip(inv1, inv2):
            assert x1 == x2

    for _, (query1, query2) in enumerate(zip(valid1, valid2)):
        for idx1, order_d1 in query1.items():
            order_d2 = query2.get(idx1, None)
            if order_d2:
                idx_diff = order_d1[0] - order_d2[0]

                if idx_diff not in idx_diffs:
                    idx_diffs[idx_diff] = 1
                else:
                    idx_diffs[idx_diff] += 1
            else:
                idx_diffs_inf += 1

    return idx_diffs, idx_diffs_inf, idx_invalid


def check_diffs(total_num, in_window_thresh, diffs, diff_inf, invalid):
    # We require a certain fraction of results to be within +/- diff_window
    # index differences
    diff_window = 4
    in_window = 0

    for diff in sorted(diffs):
        if abs(diff) <= diff_window:
            in_window += diffs[diff] / total_num

    if (in_window < in_window_thresh):
        print('error {} {}'.format(in_window, in_window_thresh))
        assert in_window >= in_window_thresh


def do_test_with_index(ci, gi, nprobe, k, clamp, in_window_thresh):
    num_query = 11
    to_query = make_t(num_query, ci.d, clamp)

    ci.nprobe = ci.nprobe
    gi.nprobe = gi.nprobe

    total_num = num_query * k
    check_diffs(total_num, in_window_thresh,
                *compare_results(*ci.search(to_query, k),
                                 *gi.search(to_query, k)))


def do_test(nlist, d, qtype, by_residual, metric, nprobe, k):
    clamp = (qtype == faiss.ScalarQuantizer.QT_8bit_direct)
    ci, gi = make_indices_copy_from_cpu(nlist, d, qtype,
                                        by_residual, metric, clamp)
    # A direct copy should be much more closely in agreement
    # (except for fp accumulation order differences)
    do_test_with_index(ci, gi, nprobe, k, clamp, 0.99)

    ci, gi = make_indices_copy_from_gpu(nlist, d, qtype,
                                        by_residual, metric, clamp)
    # A direct copy should be much more closely in agreement
    # (except for fp accumulation order differences)
    do_test_with_index(ci, gi, nprobe, k, clamp, 0.99)

    ci, gi = make_indices_train(nlist, d, qtype,
                                by_residual, metric, clamp)
    # Separate training can produce a slightly different coarse quantizer
    # and residuals
    do_test_with_index(ci, gi, nprobe, k, clamp, 0.8)


def do_multi_test(qtype):
    nlist = 100
    nprobe = 10
    k = 50

    for d in [11, 64, 77]:
        if (qtype != faiss.ScalarQuantizer.QT_8bit_direct):
            # residual doesn't make sense here
            do_test(nlist, d, qtype, True,
                    faiss.METRIC_L2, nprobe, k)
            do_test(nlist, d, qtype, True,
                    faiss.METRIC_INNER_PRODUCT, nprobe, k)
        do_test(nlist, d, qtype, False, faiss.METRIC_L2, nprobe, k)
        do_test(nlist, d, qtype, False, faiss.METRIC_INNER_PRODUCT, nprobe, k)

#
# Test
#


class TestSQ(unittest.TestCase):
    def test_fp16(self):
        do_multi_test(faiss.ScalarQuantizer.QT_fp16)

    def test_8bit(self):
        do_multi_test(faiss.ScalarQuantizer.QT_8bit)

    def test_8bit_uniform(self):
        do_multi_test(faiss.ScalarQuantizer.QT_8bit_uniform)

    def test_6bit(self):
        do_multi_test(faiss.ScalarQuantizer.QT_6bit)

    def test_4bit(self):
        do_multi_test(faiss.ScalarQuantizer.QT_4bit)

    def test_4bit_uniform(self):
        do_multi_test(faiss.ScalarQuantizer.QT_4bit_uniform)

    def test_8bit_direct(self):
        do_multi_test(faiss.ScalarQuantizer.QT_8bit_direct)


@unittest.skipIf(
    "CUVS" not in faiss.get_compile_options(),
    "only if CUVS is compiled in")
class TestCuvsSQ8(unittest.TestCase):

    def make_gpu_index(self, metric):
        nlist = 64
        nprobe = 8
        d = 37
        xt = make_t(4000, d)
        xb = make_t(3000, d)

        res = faiss.StandardGpuResources()
        res.noTempMemory()
        config = faiss.GpuIndexIVFScalarQuantizerConfig()
        config.use_cuvs = True
        config.indicesOptions = faiss.INDICES_64_BIT

        idx_gpu = faiss.GpuIndexIVFScalarQuantizer(
            res, d, nlist, faiss.ScalarQuantizer.QT_8bit,
            metric, True, config)
        idx_gpu.train(xt)
        idx_gpu.add(xb)
        idx_gpu.nprobe = nprobe
        return res, config, idx_gpu, xb, nlist, nprobe

    def check_metric(self, metric):
        res, config, idx_gpu, xb, nlist, nprobe = self.make_gpu_index(metric)
        k = 10
        xq = make_t(13, idx_gpu.d)
        D, I = idx_gpu.search(xq, k)
        self.assertEqual(I.shape, (13, k))
        self.assertTrue(np.any(I != -1))

        quantizer = faiss.IndexFlat(idx_gpu.d, metric)
        idx_cpu = faiss.IndexIVFScalarQuantizer(
            quantizer, idx_gpu.d, nlist,
            faiss.ScalarQuantizer.QT_8bit, metric, True)
        idx_gpu.copyTo(idx_cpu)
        idx_cpu.nprobe = nprobe

        self.assertEqual(idx_cpu.ntotal, idx_gpu.ntotal)
        self.assertEqual(idx_cpu.sq.qtype, faiss.ScalarQuantizer.QT_8bit)
        self.assertEqual(idx_cpu.sq.trained.size(), 2 * idx_gpu.d)
        do_test_with_index(idx_cpu, idx_gpu, nprobe, k, False, 0.8)

        idx_gpu_copy = faiss.GpuIndexIVFScalarQuantizer(
            res, idx_cpu, config)
        idx_gpu_copy.nprobe = nprobe
        self.assertEqual(idx_gpu_copy.ntotal, idx_cpu.ntotal)
        do_test_with_index(idx_cpu, idx_gpu_copy, nprobe, k, False, 0.8)

        xq_nan = make_t(2, idx_gpu.d)
        xq_nan[1, 0] = np.nan
        D_nan, I_nan = idx_gpu.search(xq_nan, k)
        np.testing.assert_array_equal(I_nan[1], -np.ones(k, dtype='int64'))
        np.testing.assert_allclose(
            D_nan[1], np.full(k, np.finfo('float32').max, dtype='float32'))

        idx_gpu.reset()
        self.assertEqual(idx_gpu.ntotal, 0)
        idx_gpu.add(xb[:100])
        self.assertEqual(idx_gpu.ntotal, 100)

    def test_sq8_l2(self):
        self.check_metric(faiss.METRIC_L2)

    def test_sq8_ip(self):
        self.check_metric(faiss.METRIC_INNER_PRODUCT)
