# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import faiss
import unittest



class PartitionTests:

    def test_partition(self):
        self.do_partition(160, 80)

    def test_partition_manydups(self):
        self.do_partition(160, 80, maxval=16)

    def test_partition_lowq(self):
        self.do_partition(160, 10, maxval=16)

    def test_partition_highq(self):
        self.do_partition(165, 155, maxval=16)

    def test_partition_q10(self):
        self.do_partition(32, 10, maxval=500)

    def test_partition_q10_dups(self):
        self.do_partition(32, 10, maxval=16)

    def test_partition_q10_fuzzy(self):
        self.do_partition(32, (10, 15), maxval=500)

    def test_partition_fuzzy(self):
        self.do_partition(160, (70, 80), maxval=500)

    def test_partition_fuzzy_2(self):
        self.do_partition(160, (70, 80))


def pointer_to_minus1():
    return np.array([-1], dtype='int64').view("uint64")

class TestPartitioningFloat(unittest.TestCase, PartitionTests):

    def do_partition(self, n, q, maxval=None, seed=None):
        if seed is None:
            for i in range(50):
                self.do_partition(n, q, maxval, i + 1234)
        rs = np.random.RandomState(seed)
        if maxval is None:
            vals = rs.rand(n).astype('float32')
        else:
            vals = rs.randint(maxval, size=n).astype('float32')

        ids = (rs.permutation(n) + 12345).astype('int64')
        dic = dict(zip(ids, vals))

        vals_orig = vals.copy()

        sp = faiss.swig_ptr
        if type(q) == int:
            faiss.CMax_float_partition_fuzzy(
                sp(vals), sp(ids), n,
                q, q, None
            )
        else:
            q_min, q_max = q
            q = pointer_to_minus1()
            faiss.CMax_float_partition_fuzzy(
                sp(vals), sp(ids), n,
                q_min, q_max, sp(q)
            )
            q = q[0]
            assert q_min <= q <= q_max

        o = vals_orig.argsort()
        thresh = vals_orig[o[q]]
        n_eq = (vals_orig[o[:q]] == thresh).sum()

        for i in range(q):
            self.assertEqual(vals[i], dic[ids[i]])
            self.assertLessEqual(vals[i], thresh)
            if vals[i] == thresh:
                n_eq -= 1
        self.assertEqual(n_eq, 0)


class TestPartitioningFloatMin(unittest.TestCase, PartitionTests):

    def do_partition(self, n, q, maxval=None, seed=None):
        if seed is None:
            for i in range(50):
                self.do_partition(n, q, maxval, i + 1234)
        rs = np.random.RandomState(seed)
        if maxval is None:
            vals = rs.rand(n).astype('float32')
            mirval = 1.0
        else:
            vals = rs.randint(maxval, size=n).astype('float32')
            mirval = 65536

        ids = (rs.permutation(n) + 12345).astype('int64')
        dic = dict(zip(ids, vals))

        vals_orig = vals.copy()

        vals[:] = mirval - vals

        sp = faiss.swig_ptr
        if type(q) == int:
            faiss.CMin_float_partition_fuzzy(
                sp(vals), sp(ids), n,
                q, q, None
            )
        else:
            q_min, q_max = q
            q = pointer_to_minus1()
            faiss.CMin_float_partition_fuzzy(
                sp(vals), sp(ids), n,
                q_min, q_max, sp(q)
            )
            q = q[0]
            assert q_min <= q <= q_max

        vals[:] = mirval - vals

        o = vals_orig.argsort()
        thresh = vals_orig[o[q]]
        n_eq = (vals_orig[o[:q]] == thresh).sum()

        for i in range(q):
            np.testing.assert_almost_equal(vals[i], dic[ids[i]], decimal=5)
            self.assertLessEqual(vals[i], thresh)
            if vals[i] == thresh:
                n_eq -= 1
        self.assertEqual(n_eq, 0)


class TestPartitioningUint16(unittest.TestCase, PartitionTests):

    def do_partition(self, n, q, maxval=65536, seed=None):
        if seed is None:
            for i in range(50):
                self.do_partition(n, q, maxval, i + 1234)

        rs = np.random.RandomState(seed)
        vals = rs.randint(maxval, size=n).astype('uint16')
        ids = (rs.permutation(n) + 12345).astype('int64')
        dic = dict(zip(ids, vals))

        sp = faiss.swig_ptr
        vals_orig = vals.copy()

        tab_a = faiss.AlignedTableUint16()
        faiss.copy_array_to_AlignedTable(vals, tab_a)

        if type(q) == int:
            faiss.CMax_uint16_partition_fuzzy(
                tab_a.get(), sp(ids), n, q, q, None)
        else:
            q_min, q_max = q
            q = pointer_to_minus1()
            faiss.CMax_uint16_partition_fuzzy(
                tab_a.get(), sp(ids), n,
                q_min, q_max, sp(q)
            )
            q = q[0]
            assert q_min <= q <= q_max

        vals = faiss.AlignedTable_to_array(tab_a)

        o = vals_orig.argsort()
        thresh = vals_orig[o[q]]
        n_eq = (vals_orig[o[:q]] == thresh).sum()

        for i in range(q):
            self.assertEqual(vals[i], dic[ids[i]])
            self.assertLessEqual(vals[i], thresh)
            if vals[i] == thresh:
                n_eq -= 1
        self.assertEqual(n_eq, 0)



class TestPartitioningUint16Min(unittest.TestCase, PartitionTests):

    def do_partition(self, n, q, maxval=65536, seed=None):
        #seed = 1235
        if seed is None:
            for i in range(50):
                self.do_partition(n, q, maxval, i + 1234)
        rs = np.random.RandomState(seed)
        vals = rs.randint(maxval, size=n).astype('uint16')
        ids = (rs.permutation(n) + 12345).astype('int64')
        dic = dict(zip(ids, vals))

        sp = faiss.swig_ptr
        vals_orig = vals.copy()

        tab_a = faiss.AlignedTableUint16()
        vals_inv = (65535 - vals).astype('uint16')
        faiss.copy_array_to_AlignedTable(vals_inv, tab_a)

        if type(q) == int:
            faiss.CMin_uint16_partition_fuzzy(
                tab_a.get(), sp(ids), n, q, q, None)
        else:
            q_min, q_max = q
            q = pointer_to_minus1()
            thresh2 = faiss.CMin_uint16_partition_fuzzy(
                tab_a.get(), sp(ids), n,
                q_min, q_max, sp(q)
            )
            q = q[0]
            assert q_min <= q <= q_max

        vals_inv = faiss.AlignedTable_to_array(tab_a)
        vals = 65535 - vals_inv

        o = vals_orig.argsort()
        thresh = vals_orig[o[q]]
        n_eq = (vals_orig[o[:q]] == thresh).sum()

        for i in range(q):
            self.assertEqual(vals[i], dic[ids[i]])
            self.assertLessEqual(vals[i], thresh)
            if vals[i] == thresh:
                n_eq -= 1
        self.assertEqual(n_eq, 0)


class TestHistograms(unittest.TestCase):

    def do_test(self, nbin, n):
        rs = np.random.RandomState(123)
        tab = rs.randint(nbin, size=n).astype('uint16')
        ref_histogram = np.bincount(tab, minlength=nbin)

        tab_a = faiss.AlignedTableUint16()
        faiss.copy_array_to_AlignedTable(tab, tab_a)

        sp = faiss.swig_ptr
        hist = np.zeros(nbin, 'int32')
        if nbin == 8:
            faiss.simd_histogram_8(tab_a.get(), n, 0, -1, sp(hist))
        elif nbin == 16:
            faiss.simd_histogram_16(tab_a.get(), n, 0, -1, sp(hist))
        else:
            raise AssertionError()
        np.testing.assert_array_equal(hist, ref_histogram)

    def test_8bin_even(self):
        self.do_test(8, 5 * 16)

    def test_8bin_odd(self):
        self.do_test(8, 123)

    def test_16bin_even(self):
        self.do_test(16, 5 * 16)

    def test_16bin_odd(self):
        self.do_test(16, 123)


    def do_test_bounded(self, nbin, n, shift=2, minv=500, rspan=None, seed=None):
        if seed is None:
            for run in range(50):
                self.do_test_bounded(nbin, n, shift, minv, rspan, seed=123 + run)
            return

        if rspan is None:
            rmin, rmax = 0, nbin * 6
        else:
            rmin, rmax = rspan

        rs = np.random.RandomState(seed)
        tab = rs.randint(rmin, rmax, size=n).astype('uint16')
        bc = np.bincount(tab, minlength=65536)

        binsize = 1 << shift
        ref_histogram = bc[minv : minv + binsize * nbin]

        def pad_and_reshape(x, m, n):
            xout = np.zeros(m * n, dtype=x.dtype)
            xout[:x.size] = x
            return xout.reshape(m, n)

        ref_histogram = pad_and_reshape(ref_histogram, nbin, binsize)
        ref_histogram = ref_histogram.sum(1)

        tab_a = faiss.AlignedTableUint16()
        faiss.copy_array_to_AlignedTable(tab, tab_a)
        sp = faiss.swig_ptr

        hist = np.zeros(nbin, 'int32')
        if nbin == 8:
            faiss.simd_histogram_8(
                tab_a.get(), n, minv, shift, sp(hist)
            )
        elif nbin == 16:
            faiss.simd_histogram_16(
                tab_a.get(), n, minv, shift, sp(hist)
            )
        else:
            raise AssertionError()

        np.testing.assert_array_equal(hist, ref_histogram)

    def test_8bin_even_bounded(self):
        self.do_test_bounded(8, 22 * 16)

    def test_8bin_odd_bounded(self):
        self.do_test_bounded(8, 10000)

    def test_16bin_even_bounded(self):
        self.do_test_bounded(16, 22 * 16)

    def test_16bin_odd_bounded(self):
        self.do_test_bounded(16, 10000)

    def test_16bin_bounded_bigrange(self):
        self.do_test_bounded(16, 1000, shift=12, rspan=(10, 65500))

    def test_8bin_bounded_bigrange(self):
        self.do_test_bounded(8, 1000, shift=13, rspan=(10, 65500))

    def test_16bin_bounded_bigrange_2(self):
        self.do_test_bounded(16, 10, shift=12, rspan=(65000, 65500))

    def test_16bin_bounded_shift0(self):
        self.do_test_bounded(16, 10000, shift=0, rspan=(10, 65500))

    def test_8bin_bounded_shift0(self):
        self.do_test_bounded(8, 10000, shift=0, rspan=(10, 65500))

    def test_16bin_bounded_ignore_out_range(self):
        self.do_test_bounded(16, 10000, shift=5, rspan=(100, 20000), minv=300)

    def test_8bin_bounded_ignore_out_range(self):
        self.do_test_bounded(8, 10000, shift=5, rspan=(100, 20000), minv=300)
