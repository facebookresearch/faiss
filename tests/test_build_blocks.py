# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import numpy as np

import faiss
import unittest

from common_faiss_tests import get_dataset_2


class TestPCA(unittest.TestCase):

    def test_pca(self):
        d = 64
        n = 1000
        np.random.seed(123)
        x = np.random.random(size=(n, d)).astype('float32')

        pca = faiss.PCAMatrix(d, 10)
        pca.train(x)
        y = pca.apply_py(x)

        # check that energy per component is decreasing
        column_norm2 = (y**2).sum(0)

        prev = 1e50
        for o in column_norm2:
            self.assertGreater(prev, o)
            prev = o

    def test_pca_epsilon(self):
        d = 64
        n = 1000
        np.random.seed(123)
        x = np.random.random(size=(n, d)).astype('float32')

        # make sure data is in a sub-space
        x[:, ::2] = 0

        # check division by 0 with default computation
        pca = faiss.PCAMatrix(d, 60, -0.5)
        pca.train(x)
        y = pca.apply(x)
        self.assertFalse(np.all(np.isfinite(y)))

        # check add epsilon
        pca = faiss.PCAMatrix(d, 60, -0.5)
        pca.epsilon = 1e-5
        pca.train(x)
        y = pca.apply(x)
        self.assertTrue(np.all(np.isfinite(y)))

        # check I/O
        index = faiss.index_factory(d, "PCAW60,Flat")
        index = faiss.deserialize_index(faiss.serialize_index(index))
        pca1 = faiss.downcast_VectorTransform(index.chain.at(0))
        pca1.epsilon = 1e-5
        index.train(x)
        pca = faiss.downcast_VectorTransform(index.chain.at(0))
        y = pca.apply(x)
        self.assertTrue(np.all(np.isfinite(y)))


class TestMapLong2Long(unittest.TestCase):

    def test_maplong2long(self):
        keys = np.array([13, 45, 67], dtype=np.int64)
        vals = np.array([3, 8, 2], dtype=np.int64)

        m = faiss.MapLong2Long()
        m.add(keys, vals)

        assert np.all(m.search_multiple(keys) == vals)

        assert m.search(12343) == -1


class TestOrthognalReconstruct(unittest.TestCase):

    def test_recons_orthonormal(self):
        lt = faiss.LinearTransform(20, 10, True)
        rs = np.random.RandomState(10)
        A, _ = np.linalg.qr(rs.randn(20, 20))
        A = A[:10].astype('float32')
        faiss.copy_array_to_vector(A.ravel(), lt.A)
        faiss.copy_array_to_vector(rs.randn(10).astype('float32'), lt.b)

        lt.set_is_orthonormal()
        lt.is_trained = True
        assert lt.is_orthonormal

        x = rs.rand(30, 20).astype('float32')
        xt = lt.apply_py(x)
        xtt = lt.reverse_transform(xt)
        xttt = lt.apply_py(xtt)

        err = ((xt - xttt)**2).sum()

        self.assertGreater(1e-5, err)

    def test_recons_orthogona_impossible(self):
        lt = faiss.LinearTransform(20, 10, True)
        rs = np.random.RandomState(10)
        A = rs.randn(10 * 20).astype('float32')
        faiss.copy_array_to_vector(A.ravel(), lt.A)
        faiss.copy_array_to_vector(rs.randn(10).astype('float32'), lt.b)
        lt.is_trained = True

        lt.set_is_orthonormal()
        assert not lt.is_orthonormal

        x = rs.rand(30, 20).astype('float32')
        xt = lt.apply_py(x)
        try:
            lt.reverse_transform(xt)
        except Exception:
            pass
        else:
            self.assertFalse('should do an exception')


class TestMAdd(unittest.TestCase):

    def test_1(self):
        # try with dimensions that are multiples of 16 or not
        rs = np.random.RandomState(123)
        swig_ptr = faiss.swig_ptr
        for dim in 16, 32, 20, 25:
            for _repeat in 1, 2, 3, 4, 5:
                a = rs.rand(dim).astype('float32')
                b = rs.rand(dim).astype('float32')
                c = np.zeros(dim, dtype='float32')
                bf = rs.uniform(5.0) - 2.5
                idx = faiss.fvec_madd_and_argmin(
                    dim, swig_ptr(a), bf, swig_ptr(b),
                    swig_ptr(c))
                ref_c = a + b * bf
                assert np.abs(c - ref_c).max() < 1e-5
                assert idx == ref_c.argmin()


class TestNyFuncs(unittest.TestCase):

    def test_l2(self):
        rs = np.random.RandomState(123)
        swig_ptr = faiss.swig_ptr
        for d in 1, 2, 4, 8, 12, 16:
            x = rs.rand(d).astype('float32')
            for ny in 128, 129, 130:
                y = rs.rand(ny, d).astype('float32')
                ref = ((x - y) ** 2).sum(1)
                new = np.zeros(ny, dtype='float32')
                faiss.fvec_L2sqr_ny(swig_ptr(new), swig_ptr(x),
                                    swig_ptr(y), d, ny)
                assert np.abs(ref - new).max() < 1e-4

    def test_IP(self):
        # this one is not optimized with SIMD but just in case
        rs = np.random.RandomState(123)
        swig_ptr = faiss.swig_ptr
        for d in 1, 2, 4, 8, 12, 16:
            x = rs.rand(d).astype('float32')
            for ny in 128, 129, 130:
                y = rs.rand(ny, d).astype('float32')
                ref = (x * y).sum(1)
                new = np.zeros(ny, dtype='float32')
                faiss.fvec_inner_products_ny(
                    swig_ptr(new), swig_ptr(x), swig_ptr(y), d, ny)
                assert np.abs(ref - new).max() < 1e-4


class TestMatrixStats(unittest.TestCase):

    def test_0s(self):
        rs = np.random.RandomState(123)
        m = rs.rand(40, 20).astype('float32')
        m[5:10] = 0
        comments = faiss.MatrixStats(m).comments
        assert 'has 5 copies' in comments
        assert '5 null vectors' in comments

    def test_copies(self):
        rs = np.random.RandomState(123)
        m = rs.rand(40, 20).astype('float32')
        m[::2] = m[1::2]
        comments = faiss.MatrixStats(m).comments
        assert '20 vectors are distinct' in comments

    def test_dead_dims(self):
        rs = np.random.RandomState(123)
        m = rs.rand(40, 20).astype('float32')
        m[:, 5:10] = 0
        comments = faiss.MatrixStats(m).comments
        assert '5 dimensions are constant' in comments

    def test_rogue_means(self):
        rs = np.random.RandomState(123)
        m = rs.rand(40, 20).astype('float32')
        m[:, 5:10] += 12345
        comments = faiss.MatrixStats(m).comments
        assert '5 dimensions are too large wrt. their variance' in comments

    def test_normalized(self):
        rs = np.random.RandomState(123)
        m = rs.rand(40, 20).astype('float32')
        faiss.normalize_L2(m)
        comments = faiss.MatrixStats(m).comments
        assert 'vectors are normalized' in comments

    def test_hash(self):
        cc = []
        for _ in range(2):
            rs = np.random.RandomState(123)
            m = rs.rand(40, 20).astype('float32')
            cc.append(faiss.MatrixStats(m).hash_value)
        self.assertTrue(cc[0] == cc[1])


class TestScalarQuantizer(unittest.TestCase):

    def test_8bit_equiv(self):
        rs = np.random.RandomState(123)
        for _it in range(20):
            for d in 13, 16, 24:
                x = np.floor(rs.rand(5, d) * 256).astype('float32')
                x[0] = 0
                x[1] = 255

                # make sure to test extreme cases
                x[2, 0] = 0
                x[3, 0] = 255
                x[2, 1] = 255
                x[3, 1] = 0

                ref_index = faiss.IndexScalarQuantizer(
                    d, faiss.ScalarQuantizer.QT_8bit)
                ref_index.train(x[:2])
                ref_index.add(x[2:3])

                index = faiss.IndexScalarQuantizer(
                    d, faiss.ScalarQuantizer.QT_8bit_direct)
                assert index.is_trained
                index.add(x[2:3])

                assert np.all(
                    faiss.vector_to_array(ref_index.codes) ==
                    faiss.vector_to_array(index.codes))

                # Note that distances are not the same because ref_index
                # reconstructs x as x + 0.5
                D, I = index.search(x[3:], 1)

                # assert D[0, 0] == Dref[0, 0]
                assert D[0, 0] == ((x[3] - x[2]) ** 2).sum()

    def test_6bit_equiv(self):
        rs = np.random.RandomState(123)
        for d in 3, 6, 8, 16, 36:
            trainset = np.zeros((2, d), dtype='float32')
            trainset[0, :] = 0
            trainset[0, :] = 63

            index = faiss.IndexScalarQuantizer(
                d, faiss.ScalarQuantizer.QT_6bit)
            index.train(trainset)

            x = rs.randint(64, size=(100, d)).astype('float32')

            # verify encoder / decoder
            index.add(x)
            x2 = index.reconstruct_n(0, x.shape[0])
            assert np.all(x == x2 - 0.5)

            # verify AVX decoder (used only for search)
            y = 63 * rs.rand(20, d).astype('float32')

            D, I = index.search(y, 10)
            for i in range(20):
                for j in range(10):
                    dis = ((y[i] - x2[I[i, j]]) ** 2).sum()
                    assert abs(D[i, j] - dis) / dis < 1e-5

    def test_reconstruct(self):
        self.do_reconstruct(True)

    def test_reconstruct_no_residual(self):
        self.do_reconstruct(False)

    def do_reconstruct(self, by_residual):
        d = 32
        xt, xb, xq = get_dataset_2(d, 100, 5, 5)

        index = faiss.index_factory(d, "IVF10,SQ8")
        index.by_residual = by_residual
        index.train(xt)
        index.add(xb)
        index.nprobe = 10
        D, I = index.search(xq, 4)
        xb2 = index.reconstruct_n(0, index.ntotal)
        for i in range(5):
            for j in range(4):
                self.assertAlmostEqual(
                    ((xq[i] - xb2[I[i, j]]) ** 2).sum(),
                    D[i, j],
                    places=4
                )


class TestRandom(unittest.TestCase):

    def test_rand(self):
        x = faiss.rand(2000)
        assert np.all(x >= 0) and np.all(x < 1)
        h, _ = np.histogram(x, np.arange(0, 1, 0.1))
        assert h.min() > 160 and h.max() < 240

    def test_randint(self):
        x = faiss.randint(20000, vmax=100)
        assert np.all(x >= 0) and np.all(x < 100)
        c = np.bincount(x, minlength=100)
        assert c.max() - c.min() < 50 * 2

    def test_rand_vector(self):
        """ test if the smooth_vectors function is reasonably compressible with
        a small PQ """
        x = faiss.rand_smooth_vectors(1300, 32)
        xt = x[:1000]
        xb = x[1000:1200]
        xq = x[1200:]
        _, gt = faiss.knn(xq, xb, 10)
        index = faiss.IndexPQ(32, 4, 4)
        index.train(xt)
        index.add(xb)
        D, I = index.search(xq, 10)
        ninter = faiss.eval_intersection(I, gt)
        # 445 for SyntheticDataset
        self.assertGreater(ninter, 420)
        self.assertLess(ninter, 460)


class TestPairwiseDis(unittest.TestCase):

    def test_L2(self):
        swig_ptr = faiss.swig_ptr
        x = faiss.rand((100, 10), seed=1)
        y = faiss.rand((200, 10), seed=2)
        ix = faiss.randint(50, vmax=100)
        iy = faiss.randint(50, vmax=200)
        dis = np.empty(50, dtype='float32')
        faiss.pairwise_indexed_L2sqr(
            10, 50,
            swig_ptr(x), swig_ptr(ix),
            swig_ptr(y), swig_ptr(iy),
            swig_ptr(dis))

        for i in range(50):
            assert np.allclose(
                dis[i], ((x[ix[i]] - y[iy[i]]) ** 2).sum())

    def test_IP(self):
        swig_ptr = faiss.swig_ptr
        x = faiss.rand((100, 10), seed=1)
        y = faiss.rand((200, 10), seed=2)
        ix = faiss.randint(50, vmax=100)
        iy = faiss.randint(50, vmax=200)
        dis = np.empty(50, dtype='float32')
        faiss.pairwise_indexed_inner_product(
            10, 50,
            swig_ptr(x), swig_ptr(ix),
            swig_ptr(y), swig_ptr(iy),
            swig_ptr(dis))

        for i in range(50):
            assert np.allclose(
                dis[i], np.dot(x[ix[i]], y[iy[i]]))


class TestResultHeap(unittest.TestCase):

    def test_keep_min(self):
        self.run_test(False)

    def test_keep_max(self):
        self.run_test(True)

    def run_test(self, keep_max):
        nq = 100
        nb = 1000
        restab = faiss.rand((nq, nb), 123)
        ids = faiss.randint((nq, nb), 1324, 10000)
        all_rh = {}
        for nstep in 1, 3:
            rh = faiss.ResultHeap(nq, 10, keep_max=keep_max)
            for i in range(nstep):
                i0, i1 = i * nb // nstep, (i + 1) * nb // nstep
                D = restab[:, i0:i1].copy()
                I = ids[:, i0:i1].copy()
                rh.add_result(D, I)
            rh.finalize()
            if keep_max:
                assert np.all(rh.D[:, :-1] >= rh.D[:, 1:])
            else:
                assert np.all(rh.D[:, :-1] <= rh.D[:, 1:])
            all_rh[nstep] = rh

        np.testing.assert_equal(all_rh[1].D, all_rh[3].D)
        np.testing.assert_equal(all_rh[1].I, all_rh[3].I)


class TestReconstructBatch(unittest.TestCase):

    def test_indexflat(self):
        index = faiss.IndexFlatL2(32)
        x = faiss.randn((100, 32), 1234)
        index.add(x)

        subset = [4, 7, 45]
        np.testing.assert_equal(x[subset], index.reconstruct_batch(subset))

    def test_exception(self):
        index = faiss.index_factory(32, "IVF2,Flat")
        x = faiss.randn((100, 32), 1234)
        index.train(x)
        index.add(x)

        # make sure it raises an exception even if it enters the openmp for
        subset = np.zeros(1200, dtype=int)
        self.assertRaises(
            RuntimeError,
            lambda : index.reconstruct_batch(subset),
        )


class TestBucketSort(unittest.TestCase):

    def do_test_bucket_sort(self, nt):
        rs = np.random.RandomState(123)
        tab = rs.randint(100, size=1000, dtype='int64')
        lims, perm = faiss.bucket_sort(tab, nt=nt)
        for i in range(max(tab) + 1):
            assert np.all(tab[perm[lims[i]: lims[i + 1]]] == i)

    def test_bucket_sort(self):
        self.do_test_bucket_sort(0)

    def test_bucket_sort_parallel(self):
        self.do_test_bucket_sort(4)

    def do_test_bucket_sort_inplace(
            self, nt, nrow=500, ncol=20, nbucket=300, repro=False,
            dtype='int32'):
        rs = np.random.RandomState(123)
        tab = rs.randint(nbucket, size=(nrow, ncol), dtype=dtype)

        tab2 = tab.copy()
        faiss.cvar.bucket_sort_verbose
        faiss.cvar.bucket_sort_verbose = 1

        lims = faiss.matrix_bucket_sort_inplace(tab2, nt=nt)
        tab2 = tab2.ravel()

        for b in range(nbucket):
            rows, _ = np.where(tab == b)
            rows.sort()
            tab2[lims[b]:lims[b + 1]].sort()
            rows = set(rows)
            self.assertEqual(rows, set(tab2[lims[b]:lims[b + 1]]))

    def test_bucket_sort_inplace(self):
        self.do_test_bucket_sort_inplace(0)

    def test_bucket_sort_inplace_parallel(self):
        self.do_test_bucket_sort_inplace(4)

    def test_bucket_sort_inplace_parallel_fewcol(self):
        self.do_test_bucket_sort_inplace(4, ncol=3)

    def test_bucket_sort_inplace_parallel_fewbucket(self):
        self.do_test_bucket_sort_inplace(4, nbucket=5)

    def test_bucket_sort_inplace_int64(self):
        self.do_test_bucket_sort_inplace(0, dtype='int64')

    def test_bucket_sort_inplace_parallel_int64(self):
        self.do_test_bucket_sort_inplace(4, dtype='int64')


class TestMergeKNNResults(unittest.TestCase):

    def do_test(self, ismax, dtype):
        rs = np.random.RandomState()
        n, k, nshard = 10, 5, 3
        all_ids = rs.randint(100000, size=(nshard, n, k)).astype('int64')
        all_dis = rs.rand(nshard, n, k)
        if dtype == 'int32':
            all_dis = (all_dis * 1000000).astype("int32")
        else:
            all_dis = all_dis.astype(dtype)
        for i in range(nshard):
            for j in range(n):
                all_dis[i, j].sort()
                if ismax:
                    all_dis[i, j] = all_dis[i, j][::-1]
        Dref = np.zeros((n, k), dtype=dtype)
        Iref = np.zeros((n, k), dtype='int64')

        for i in range(n):
            dis = all_dis[:, i, :].ravel()
            ids = all_ids[:, i, :].ravel()
            o = dis.argsort()
            if ismax:
                o = o[::-1]
            Dref[i] = dis[o[:k]]
            Iref[i] = ids[o[:k]]

        Dnew, Inew = faiss.merge_knn_results(all_dis, all_ids, keep_max=ismax)
        np.testing.assert_array_equal(Dnew, Dref)
        np.testing.assert_array_equal(Inew, Iref)

    def test_min_float(self):
        self.do_test(ismax=False, dtype='float32')

    def test_max_int(self):
        self.do_test(ismax=True, dtype='int32')

    def test_max_float(self):
        self.do_test(ismax=True, dtype='float32')


class TestMapInt64ToInt64(unittest.TestCase):

    def do_test(self, capacity, n):
        """ test that we are able to lookup """
        rs = np.random.RandomState(123)
        # make sure we have unique values
        keys = np.unique(rs.choice(2 ** 29, size=n).astype("int64"))
        rs.shuffle(keys)
        n = keys.size
        vals = rs.choice(2 ** 30, size=n).astype('int64')
        tab = faiss.MapInt64ToInt64(capacity)
        tab.add(keys, vals)

        # lookup and check
        vals2 = tab.lookup(keys)
        np.testing.assert_array_equal(vals, vals2)

        # make a few keys that we know are not there
        mask = rs.rand(n) < 0.3
        keys[mask] = rs.choice(2 ** 29, size=n)[mask] + 2 ** 29
        vals2 = tab.lookup(keys)
        np.testing.assert_array_equal(-1, vals2[mask])
        np.testing.assert_array_equal(vals[~mask], vals2[~mask])

    def test_small(self):
        self.do_test(16384, 10000)

    def xx_test_large(self):
        # don't run by default because it's slow
        self.do_test(2 ** 21, 10 ** 6)
