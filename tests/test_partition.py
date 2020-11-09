# Copyright (c) Facebook, Inc. and its affiliates.
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



class TestPartitioningFloat(unittest.TestCase, PartitionTests):

    def do_partition(self, n, q, maxval=None, seed=None):
        if seed is None:
            for i in range(50):
                self.do_partition(n, q, maxval, i + 1234)
        # print("seed=", seed)
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
            q = np.array([-1], dtype='uint64')
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
        # print("seed=", seed)
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
            q = np.array([-1], dtype='uint64')
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

        # print("seed=", seed)
        rs = np.random.RandomState(seed)
        vals = rs.randint(maxval, size=n).astype('uint16')
        ids = (rs.permutation(n) + 12345).astype('int64')
        dic = dict(zip(ids, vals))

        sp = faiss.swig_ptr
        vals_orig = vals.copy()

        tab_a = faiss.AlignedTableUint16()
        faiss.copy_array_to_AlignedTable(vals, tab_a)

        # print("tab a type", tab_a.get())
        if type(q) == int:
            thresh2 = faiss.CMax_uint16_partition_fuzzy(
                tab_a.get(), sp(ids), n, q, q, None)
        else:
            q_min, q_max = q
            q = np.array([-1], dtype='uint64')
            thresh2 = faiss.CMax_uint16_partition_fuzzy(
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
        seed = 1235
        if seed is None:
            for i in range(50):
                self.do_partition(n, q, maxval, i + 1234)
        print("seed=", seed)
        rs = np.random.RandomState(seed)
        vals = rs.randint(maxval, size=n).astype('uint16')
        ids = (rs.permutation(n) + 12345).astype('int64')
        dic = dict(zip(ids, vals))

        sp = faiss.swig_ptr
        vals_orig = vals.copy()

        tab_a = faiss.AlignedTableUint16()
        vals_inv = (65535 - vals).astype('uint16')
        faiss.copy_array_to_AlignedTable(vals_inv, tab_a)

        # print("tab a type", tab_a.get())
        if type(q) == int:
            thresh2 = faiss.CMin_uint16_partition_fuzzy(
                tab_a.get(), sp(ids), n, q, q, None)
        else:
            q_min, q_max = q
            q = np.array([-1], dtype='uint64')
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

