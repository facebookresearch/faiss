# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Database-parallel exact Hamming k-NN (used when nq <= nthreads / 2)."""

import unittest

import numpy as np

import faiss

_POPCOUNT8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.int32)


def brute_force_distances(xq, xb):
    return _POPCOUNT8[np.bitwise_xor(xq[:, None, :], xb[None, :, :])].sum(2)


class TestHammingDbParallel(unittest.TestCase):

    def setUp(self):
        self.nt0 = faiss.omp_get_max_threads()
        self.min0 = faiss.cvar.hamming_db_parallel_min_vectors

    def tearDown(self):
        faiss.omp_set_num_threads(self.nt0)
        faiss.cvar.hamming_db_parallel_min_vectors = self.min0

    def search(self, index, xq, k, db_parallel, nt=8, params=None):
        faiss.omp_set_num_threads(nt)
        faiss.cvar.hamming_db_parallel_min_vectors = (
            0 if db_parallel else 2**62)
        return index.search(xq, k, params=params)

    def check_valid(self, D, I, Dref_all, k):
        # distances: the k smallest, sorted; ids: valid and consistent
        for i in range(len(D)):
            expected = np.sort(Dref_all[i])[:k]
            np.testing.assert_array_equal(D[i], expected)
            valid = I[i] >= 0
            np.testing.assert_array_equal(Dref_all[i, I[i][valid]], D[i][valid])
            self.assertEqual(len(set(I[i][valid])), valid.sum())

    def do_test(self, d, nb, nq, k, seed):
        rs = np.random.RandomState(seed)
        xb = rs.randint(256, size=(nb, d // 8)).astype("uint8")
        xq = rs.randint(256, size=(nq, d // 8)).astype("uint8")
        index = faiss.IndexBinaryFlat(d)
        index.add(xb)
        Dref_all = brute_force_distances(xq, xb)
        Dp, Ip = self.search(index, xq, k, db_parallel=True)
        Ds, Is = self.search(index, xq, k, db_parallel=False)
        self.check_valid(Dp, Ip, Dref_all, k)
        np.testing.assert_array_equal(Dp, Ds)

    def test_single_query(self):
        self.do_test(d=256, nb=20000, nq=1, k=10, seed=1)

    def test_few_queries(self):
        self.do_test(d=128, nb=20000, nq=4, k=10, seed=2)

    def test_many_ties(self):
        # 32-bit codes: only 33 distinct distances, so most results are ties
        self.do_test(d=32, nb=20000, nq=2, k=100, seed=3)

    def test_k1(self):
        self.do_test(d=64, nb=20000, nq=1, k=1, seed=4)

    def test_id_selector(self):
        d, nb, k = 128, 20000, 10
        rs = np.random.RandomState(5)
        xb = rs.randint(256, size=(nb, d // 8)).astype("uint8")
        xq = rs.randint(256, size=(2, d // 8)).astype("uint8")
        index = faiss.IndexBinaryFlat(d)
        index.add(xb)
        sel = faiss.IDSelectorRange(1000, 6000)
        params = faiss.SearchParameters(sel=sel)
        Dp, Ip = self.search(index, xq, k, True, params=params)
        Ds, Is = self.search(index, xq, k, False, params=params)
        np.testing.assert_array_equal(Dp, Ds)
        self.assertTrue(np.all((Ip >= 1000) & (Ip < 6000)))
        Dref = brute_force_distances(xq, xb[1000:6000])
        for i in range(2):
            np.testing.assert_array_equal(Dp[i], np.sort(Dref[i])[:k])

    def test_k_larger_than_selection(self):
        # fewer selected vectors than k: missing results are -1 / max
        d, nb, k = 64, 20000, 50
        rs = np.random.RandomState(6)
        xb = rs.randint(256, size=(nb, d // 8)).astype("uint8")
        xq = rs.randint(256, size=(1, d // 8)).astype("uint8")
        index = faiss.IndexBinaryFlat(d)
        index.add(xb)
        params = faiss.SearchParameters(sel=faiss.IDSelectorRange(0, 20))
        Dp, Ip = self.search(index, xq, k, True, params=params)
        Ds, Is = self.search(index, xq, k, False, params=params)
        np.testing.assert_array_equal(Dp, Ds)
        np.testing.assert_array_equal(np.sort(Ip[0][:20]), np.sort(Is[0][:20]))
        self.assertTrue(np.all(Ip[0][20:] == -1))

    def test_one_thread_uses_sequential_path(self):
        # with a single thread the db-parallel path must not be taken; the
        # results must still be correct
        d, nb, k = 128, 20000, 10
        rs = np.random.RandomState(7)
        xb = rs.randint(256, size=(nb, d // 8)).astype("uint8")
        xq = rs.randint(256, size=(1, d // 8)).astype("uint8")
        index = faiss.IndexBinaryFlat(d)
        index.add(xb)
        D, I = self.search(index, xq, k, True, nt=1)
        self.check_valid(D, I, brute_force_distances(xq, xb), k)


class TestHammingDbParallelCounting(unittest.TestCase):
    """use_heap=False path (hammings_knn_mc): results must be identical,
    including ids, because the counting variant is deterministic."""

    def setUp(self):
        self.nt0 = faiss.omp_get_max_threads()
        self.min0 = faiss.cvar.hamming_db_parallel_min_vectors

    def tearDown(self):
        faiss.omp_set_num_threads(self.nt0)
        faiss.cvar.hamming_db_parallel_min_vectors = self.min0

    def do_test(self, d, nb, nq, k, seed, sel=None):
        rs = np.random.RandomState(seed)
        xb = rs.randint(256, size=(nb, d // 8)).astype("uint8")
        xq = rs.randint(256, size=(nq, d // 8)).astype("uint8")
        index = faiss.IndexBinaryFlat(d)
        index.use_heap = False
        index.add(xb)
        params = None if sel is None else faiss.SearchParameters(sel=sel)
        faiss.omp_set_num_threads(8)
        faiss.cvar.hamming_db_parallel_min_vectors = 0
        Dp, Ip = index.search(xq, k, params=params)
        faiss.cvar.hamming_db_parallel_min_vectors = 2**62
        Ds, Is = index.search(xq, k, params=params)
        np.testing.assert_array_equal(Dp, Ds)
        np.testing.assert_array_equal(Ip, Is)
        Dref = brute_force_distances(xq, xb)
        for i in range(nq):
            valid = Ip[i] >= 0
            np.testing.assert_array_equal(
                Dref[i, Ip[i][valid]], Dp[i][valid])

    def test_single_query(self):
        self.do_test(d=256, nb=20000, nq=1, k=10, seed=11)

    def test_few_queries_many_ties(self):
        self.do_test(d=32, nb=20000, nq=4, k=100, seed=12)

    def test_k1(self):
        self.do_test(d=64, nb=20000, nq=2, k=1, seed=13)

    def test_id_selector(self):
        self.do_test(d=64, nb=20000, nq=2, k=10, seed=14,
                     sel=faiss.IDSelectorRange(1000, 6000))

    def test_k_larger_than_selection(self):
        self.do_test(d=64, nb=20000, nq=1, k=50, seed=15,
                     sel=faiss.IDSelectorRange(0, 20))


if __name__ == "__main__":
    unittest.main()
