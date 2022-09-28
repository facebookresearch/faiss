# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""this is a basic test script for simple indices work"""
from __future__ import absolute_import, division, print_function
# no unicode_literals because it messes up in py2

import numpy as np
import unittest
import faiss
import tempfile
import os
import re
import warnings

from common_faiss_tests import get_dataset, get_dataset_2

class TestModuleInterface(unittest.TestCase):

    def test_version_attribute(self):
        assert hasattr(faiss, '__version__')
        assert re.match('^\\d+\\.\\d+\\.\\d+$', faiss.__version__)

class TestIndexFlat(unittest.TestCase):

    def do_test(self, nq, metric_type=faiss.METRIC_L2, k=10):
        d = 32
        nb = 1000
        nt = 0

        (xt, xb, xq) = get_dataset_2(d, nt, nb, nq)
        index = faiss.IndexFlat(d, metric_type)

        ### k-NN search

        index.add(xb)
        D1, I1 = index.search(xq, k)

        if metric_type == faiss.METRIC_L2:
            all_dis = ((xq.reshape(nq, 1, d) - xb.reshape(1, nb, d)) ** 2).sum(2)
            Iref = all_dis.argsort(axis=1)[:, :k]
        else:
            all_dis = np.dot(xq, xb.T)
            Iref = all_dis.argsort(axis=1)[:, ::-1][:, :k]

        Dref = all_dis[np.arange(nq)[:, None], Iref]
        self.assertLessEqual((Iref != I1).sum(), Iref.size * 0.0001)
        #  np.testing.assert_equal(Iref, I1)
        np.testing.assert_almost_equal(Dref, D1, decimal=5)

        ### Range search

        radius = float(np.median(Dref[:, -1]))

        lims, D2, I2 = index.range_search(xq, radius)

        for i in range(nq):
            l0, l1 = lims[i:i + 2]
            _, Il = D2[l0:l1], I2[l0:l1]
            if metric_type == faiss.METRIC_L2:
                Ilref, = np.where(all_dis[i] < radius)
            else:
                Ilref, = np.where(all_dis[i] > radius)
            Il.sort()
            Ilref.sort()
            np.testing.assert_equal(Il, Ilref)
            np.testing.assert_almost_equal(
                all_dis[i, Ilref], D2[l0:l1],
                decimal=5
            )

    def set_blas_blocks(self, small):
        if small:
            faiss.cvar.distance_compute_blas_query_bs = 16
            faiss.cvar.distance_compute_blas_database_bs = 12
        else:
            faiss.cvar.distance_compute_blas_query_bs = 4096
            faiss.cvar.distance_compute_blas_database_bs = 1024

    def test_with_blas(self):
        self.set_blas_blocks(small=True)
        self.do_test(200)
        self.set_blas_blocks(small=False)

    def test_noblas(self):
        self.do_test(10)

    def test_with_blas_ip(self):
        self.set_blas_blocks(small=True)
        self.do_test(200, faiss.METRIC_INNER_PRODUCT)
        self.set_blas_blocks(small=False)

    def test_noblas_ip(self):
        self.do_test(10, faiss.METRIC_INNER_PRODUCT)

    def test_noblas_reservoir(self):
        self.do_test(10, k=150)

    def test_with_blas_reservoir(self):
        self.do_test(200, k=150)

    def test_noblas_reservoir_ip(self):
        self.do_test(10, faiss.METRIC_INNER_PRODUCT, k=150)

    def test_with_blas_reservoir_ip(self):
        self.do_test(200, faiss.METRIC_INNER_PRODUCT, k=150)





class EvalIVFPQAccuracy(unittest.TestCase):

    def test_IndexIVFPQ(self):
        d = 32
        nb = 1000
        nt = 1500
        nq = 200

        (xt, xb, xq) = get_dataset_2(d, nt, nb, nq)

        gt_index = faiss.IndexFlatL2(d)
        gt_index.add(xb)
        D, gt_nns = gt_index.search(xq, 1)

        coarse_quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFPQ(coarse_quantizer, d, 32, 8, 8)
        index.cp.min_points_per_centroid = 5    # quiet warning
        index.train(xt)
        index.add(xb)
        index.nprobe = 4
        D, nns = index.search(xq, 10)
        n_ok = (nns == gt_nns).sum()
        nq = xq.shape[0]

        self.assertGreater(n_ok, nq * 0.66)

        # check that and Index2Layer gives the same reconstruction
        # this is a bit fragile: it assumes 2 runs of training give
        # the exact same result.
        index2 = faiss.Index2Layer(coarse_quantizer, 32, 8)
        if True:
            index2.train(xt)
        else:
            index2.pq = index.pq
            index2.is_trained = True
        index2.add(xb)
        ref_recons = index.reconstruct_n(0, nb)
        new_recons = index2.reconstruct_n(0, nb)
        self.assertTrue(np.all(ref_recons == new_recons))


    def test_IMI(self):
        d = 32
        nb = 1000
        nt = 1500
        nq = 200

        (xt, xb, xq) = get_dataset_2(d, nt, nb, nq)
        d = xt.shape[1]

        gt_index = faiss.IndexFlatL2(d)
        gt_index.add(xb)
        D, gt_nns = gt_index.search(xq, 1)

        nbits = 5
        coarse_quantizer = faiss.MultiIndexQuantizer(d, 2, nbits)
        index = faiss.IndexIVFPQ(coarse_quantizer, d, (1 << nbits) ** 2, 8, 8)
        index.quantizer_trains_alone = 1
        index.train(xt)
        index.add(xb)
        index.nprobe = 100
        D, nns = index.search(xq, 10)
        n_ok = (nns == gt_nns).sum()

        # Should return 166 on mac, and 170 on linux.
        self.assertGreater(n_ok, 165)

        ############# replace with explicit assignment indexes
        nbits = 5
        pq = coarse_quantizer.pq
        centroids = faiss.vector_to_array(pq.centroids)
        centroids = centroids.reshape(pq.M, pq.ksub, pq.dsub)
        ai0 = faiss.IndexFlatL2(pq.dsub)
        ai0.add(centroids[0])
        ai1 = faiss.IndexFlatL2(pq.dsub)
        ai1.add(centroids[1])

        coarse_quantizer_2 = faiss.MultiIndexQuantizer2(d, nbits, ai0, ai1)
        coarse_quantizer_2.pq = pq
        coarse_quantizer_2.is_trained = True

        index.quantizer = coarse_quantizer_2

        index.reset()
        index.add(xb)

        D, nns = index.search(xq, 10)
        n_ok = (nns == gt_nns).sum()

        # should return the same result
        self.assertGreater(n_ok, 165)


    def test_IMI_2(self):
        d = 32
        nb = 1000
        nt = 1500
        nq = 200

        (xt, xb, xq) = get_dataset_2(d, nt, nb, nq)
        d = xt.shape[1]

        gt_index = faiss.IndexFlatL2(d)
        gt_index.add(xb)
        D, gt_nns = gt_index.search(xq, 1)

        ############# redo including training
        nbits = 5
        ai0 = faiss.IndexFlatL2(int(d / 2))
        ai1 = faiss.IndexFlatL2(int(d / 2))

        coarse_quantizer = faiss.MultiIndexQuantizer2(d, nbits, ai0, ai1)
        index = faiss.IndexIVFPQ(coarse_quantizer, d, (1 << nbits) ** 2, 8, 8)
        index.quantizer_trains_alone = 1
        index.train(xt)
        index.add(xb)
        index.nprobe = 100
        D, nns = index.search(xq, 10)
        n_ok = (nns == gt_nns).sum()

        # should return the same result
        self.assertGreater(n_ok, 165)





class TestMultiIndexQuantizer(unittest.TestCase):

    def test_search_k1(self):

        # verify codepath for k = 1 and k > 1

        d = 64
        nb = 0
        nt = 1500
        nq = 200

        (xt, xb, xq) = get_dataset(d, nb, nt, nq)

        miq = faiss.MultiIndexQuantizer(d, 2, 6)

        miq.train(xt)

        D1, I1 = miq.search(xq, 1)

        D5, I5 = miq.search(xq, 5)

        self.assertEqual(np.abs(I1[:, :1] - I5[:, :1]).max(), 0)
        self.assertEqual(np.abs(D1[:, :1] - D5[:, :1]).max(), 0)


class TestScalarQuantizer(unittest.TestCase):

    def test_4variants_ivf(self):
        d = 32
        nt = 2500
        nq = 400
        nb = 5000

        (xt, xb, xq) = get_dataset_2(d, nt, nb, nq)

        # common quantizer
        quantizer = faiss.IndexFlatL2(d)

        ncent = 64

        index_gt = faiss.IndexFlatL2(d)
        index_gt.add(xb)
        D, I_ref = index_gt.search(xq, 10)

        nok = {}

        index = faiss.IndexIVFFlat(quantizer, d, ncent,
                                   faiss.METRIC_L2)
        index.cp.min_points_per_centroid = 5    # quiet warning
        index.nprobe = 4
        index.train(xt)
        index.add(xb)
        D, I = index.search(xq, 10)
        nok['flat'] = (I[:, 0] == I_ref[:, 0]).sum()

        for qname in "QT_4bit QT_4bit_uniform QT_8bit QT_8bit_uniform QT_fp16".split():
            qtype = getattr(faiss.ScalarQuantizer, qname)
            index = faiss.IndexIVFScalarQuantizer(quantizer, d, ncent,
                                                  qtype, faiss.METRIC_L2)

            index.nprobe = 4
            index.train(xt)
            index.add(xb)
            D, I = index.search(xq, 10)

            nok[qname] = (I[:, 0] == I_ref[:, 0]).sum()
        print(nok, nq)

        self.assertGreaterEqual(nok['flat'], nq * 0.6)
        # The tests below are a bit fragile, it happens that the
        # ordering between uniform and non-uniform are reverted,
        # probably because the dataset is small, which introduces
        # jitter
        self.assertGreaterEqual(nok['flat'], nok['QT_8bit'])
        self.assertGreaterEqual(nok['QT_8bit'], nok['QT_4bit'])
        self.assertGreaterEqual(nok['QT_8bit'], nok['QT_8bit_uniform'])
        self.assertGreaterEqual(nok['QT_4bit'], nok['QT_4bit_uniform'])
        self.assertGreaterEqual(nok['QT_fp16'], nok['QT_8bit'])

    def test_4variants(self):
        d = 32
        nt = 2500
        nq = 400
        nb = 5000

        (xt, xb, xq) = get_dataset(d, nb, nt, nq)

        index_gt = faiss.IndexFlatL2(d)
        index_gt.add(xb)
        D_ref, I_ref = index_gt.search(xq, 10)

        nok = {}

        for qname in "QT_4bit QT_4bit_uniform QT_8bit QT_8bit_uniform QT_fp16".split():
            qtype = getattr(faiss.ScalarQuantizer, qname)
            index = faiss.IndexScalarQuantizer(d, qtype, faiss.METRIC_L2)
            index.train(xt)
            index.add(xb)
            D, I = index.search(xq, 10)
            nok[qname] = (I[:, 0] == I_ref[:, 0]).sum()

        print(nok, nq)

        self.assertGreaterEqual(nok['QT_8bit'], nq * 0.9)
        self.assertGreaterEqual(nok['QT_8bit'], nok['QT_4bit'])
        self.assertGreaterEqual(nok['QT_8bit'], nok['QT_8bit_uniform'])
        self.assertGreaterEqual(nok['QT_4bit'], nok['QT_4bit_uniform'])
        self.assertGreaterEqual(nok['QT_fp16'], nok['QT_8bit'])


class TestRangeSearch(unittest.TestCase):

    def test_range_search(self):
        d = 4
        nt = 100
        nq = 10
        nb = 50

        (xt, xb, xq) = get_dataset(d, nb, nt, nq)

        index = faiss.IndexFlatL2(d)
        index.add(xb)

        Dref, Iref = index.search(xq, 5)

        thresh = 0.1   # *squared* distance
        lims, D, I = index.range_search(xq, thresh)

        for i in range(nq):
            Iline = I[lims[i]:lims[i + 1]]
            Dline = D[lims[i]:lims[i + 1]]
            for j, dis in zip(Iref[i], Dref[i]):
                if dis < thresh:
                    li, = np.where(Iline == j)
                    self.assertTrue(li.size == 1)
                    idx = li[0]
                    self.assertGreaterEqual(1e-4, abs(Dline[idx] - dis))


class TestSearchAndReconstruct(unittest.TestCase):

    def run_search_and_reconstruct(self, index, xb, xq, k=10, eps=None):
        n, d = xb.shape
        assert xq.shape[1] == d
        assert index.d == d

        D_ref, I_ref = index.search(xq, k)
        R_ref = index.reconstruct_n(0, n)
        D, I, R = index.search_and_reconstruct(xq, k)

        np.testing.assert_almost_equal(D, D_ref, decimal=5)
        self.assertTrue((I == I_ref).all())
        self.assertEqual(R.shape[:2], I.shape)
        self.assertEqual(R.shape[2], d)

        # (n, k, ..) -> (n * k, ..)
        I_flat = I.reshape(-1)
        R_flat = R.reshape(-1, d)
        # Filter out -1s when not enough results
        R_flat = R_flat[I_flat >= 0]
        I_flat = I_flat[I_flat >= 0]

        recons_ref_err = np.mean(np.linalg.norm(R_flat - R_ref[I_flat]))
        self.assertLessEqual(recons_ref_err, 1e-6)

        def norm1(x):
            return np.sqrt((x ** 2).sum(axis=1))

        recons_err = np.mean(norm1(R_flat - xb[I_flat]))

        print('Reconstruction error = %.3f' % recons_err)
        if eps is not None:
            self.assertLessEqual(recons_err, eps)

        return D, I, R

    def test_IndexFlat(self):
        d = 32
        nb = 1000
        nt = 1500
        nq = 200

        (xt, xb, xq) = get_dataset(d, nb, nt, nq)

        index = faiss.IndexFlatL2(d)
        index.add(xb)

        self.run_search_and_reconstruct(index, xb, xq, eps=0.0)

    def test_IndexIVFFlat(self):
        d = 32
        nb = 1000
        nt = 1500
        nq = 200

        (xt, xb, xq) = get_dataset(d, nb, nt, nq)

        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, 32, faiss.METRIC_L2)
        index.cp.min_points_per_centroid = 5    # quiet warning
        index.nprobe = 4
        index.train(xt)
        index.add(xb)

        self.run_search_and_reconstruct(index, xb, xq, eps=0.0)

    def test_IndexIVFPQ(self):
        d = 32
        nb = 1000
        nt = 1500
        nq = 200

        (xt, xb, xq) = get_dataset(d, nb, nt, nq)

        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFPQ(quantizer, d, 32, 8, 8)
        index.cp.min_points_per_centroid = 5    # quiet warning
        index.nprobe = 4
        index.train(xt)
        index.add(xb)

        self.run_search_and_reconstruct(index, xb, xq, eps=1.0)

    def test_MultiIndex(self):
        d = 32
        nb = 1000
        nt = 1500
        nq = 200

        (xt, xb, xq) = get_dataset(d, nb, nt, nq)

        index = faiss.index_factory(d, "IMI2x5,PQ8np")
        faiss.ParameterSpace().set_index_parameter(index, "nprobe", 4)
        index.train(xt)
        index.add(xb)

        self.run_search_and_reconstruct(index, xb, xq, eps=1.0)

    def test_IndexTransform(self):
        d = 32
        nb = 1000
        nt = 1500
        nq = 200

        (xt, xb, xq) = get_dataset(d, nb, nt, nq)

        index = faiss.index_factory(d, "L2norm,PCA8,IVF32,PQ8np")
        faiss.ParameterSpace().set_index_parameter(index, "nprobe", 4)
        index.train(xt)
        index.add(xb)

        self.run_search_and_reconstruct(index, xb, xq)


class TestHNSW(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        d = 32
        nt = 0
        nb = 1500
        nq = 500

        (_, self.xb, self.xq) = get_dataset_2(d, nt, nb, nq)
        index = faiss.IndexFlatL2(d)
        index.add(self.xb)
        Dref, Iref = index.search(self.xq, 1)
        self.Iref = Iref

    def test_hnsw(self):
        d = self.xq.shape[1]

        index = faiss.IndexHNSWFlat(d, 16)
        index.add(self.xb)
        Dhnsw, Ihnsw = index.search(self.xq, 1)

        self.assertGreaterEqual((self.Iref == Ihnsw).sum(), 460)

        self.io_and_retest(index, Dhnsw, Ihnsw)

    def test_hnsw_unbounded_queue(self):
        d = self.xq.shape[1]

        index = faiss.IndexHNSWFlat(d, 16)
        index.add(self.xb)
        index.search_bounded_queue = False
        Dhnsw, Ihnsw = index.search(self.xq, 1)

        self.assertGreaterEqual((self.Iref == Ihnsw).sum(), 460)

        self.io_and_retest(index, Dhnsw, Ihnsw)

    def io_and_retest(self, index, Dhnsw, Ihnsw):
        fd, tmpfile = tempfile.mkstemp()
        os.close(fd)
        try:
            faiss.write_index(index, tmpfile)
            index2 = faiss.read_index(tmpfile)
        finally:
            if os.path.exists(tmpfile):
                os.unlink(tmpfile)

        Dhnsw2, Ihnsw2 = index2.search(self.xq, 1)

        self.assertTrue(np.all(Dhnsw2 == Dhnsw))
        self.assertTrue(np.all(Ihnsw2 == Ihnsw))

        # also test clone
        index3 = faiss.clone_index(index)
        Dhnsw3, Ihnsw3 = index3.search(self.xq, 1)

        self.assertTrue(np.all(Dhnsw3 == Dhnsw))
        self.assertTrue(np.all(Ihnsw3 == Ihnsw))


    def test_hnsw_2level(self):
        d = self.xq.shape[1]

        quant = faiss.IndexFlatL2(d)

        index = faiss.IndexHNSW2Level(quant, 256, 8, 8)
        index.train(self.xb)
        index.add(self.xb)
        Dhnsw, Ihnsw = index.search(self.xq, 1)

        self.assertGreaterEqual((self.Iref == Ihnsw).sum(), 307)

        self.io_and_retest(index, Dhnsw, Ihnsw)

    def test_add_0_vecs(self):
        index = faiss.IndexHNSWFlat(10, 16)
        zero_vecs = np.zeros((0, 10), dtype='float32')
        # infinite loop
        index.add(zero_vecs)

    def test_hnsw_IP(self):
        d = self.xq.shape[1]

        index_IP = faiss.IndexFlatIP(d)
        index_IP.add(self.xb)
        Dref, Iref = index_IP.search(self.xq, 1)

        index = faiss.IndexHNSWFlat(d, 16, faiss.METRIC_INNER_PRODUCT)
        index.add(self.xb)
        Dhnsw, Ihnsw = index.search(self.xq, 1)

        print('nb equal: ', (Iref == Ihnsw).sum())

        self.assertGreaterEqual((Iref == Ihnsw).sum(), 470)

        mask = Iref[:, 0] == Ihnsw[:, 0]
        assert np.allclose(Dref[mask, 0], Dhnsw[mask, 0])


class TestNSG(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        d = 32
        nt = 0
        nb = 1500
        nq = 500
        self.GK = 32

        _, self.xb, self.xq = get_dataset_2(d, nt, nb, nq)

    def make_knn_graph(self, metric):
        n = self.xb.shape[0]
        d = self.xb.shape[1]
        index = faiss.IndexFlat(d, metric)
        index.add(self.xb)
        _, I = index.search(self.xb, self.GK + 1)
        knn_graph = np.zeros((n, self.GK), dtype=np.int64)

        # For the inner product distance, the distance between a vector and itself
        # may not be the smallest, so it is not guaranteed that I[:, 0] is the query itself.
        for i in range(n):
            cnt = 0
            for j in range(self.GK + 1):
                if I[i, j] != i:
                    knn_graph[i, cnt] = I[i, j]
                    cnt += 1
                if cnt == self.GK:
                    break
        return knn_graph

    def subtest_io_and_clone(self, index, Dnsg, Insg):
        fd, tmpfile = tempfile.mkstemp()
        os.close(fd)
        try:
            faiss.write_index(index, tmpfile)
            index2 = faiss.read_index(tmpfile)
        finally:
            if os.path.exists(tmpfile):
                os.unlink(tmpfile)

        Dnsg2, Insg2 = index2.search(self.xq, 1)
        np.testing.assert_array_equal(Dnsg2, Dnsg)
        np.testing.assert_array_equal(Insg2, Insg)

        # also test clone
        index3 = faiss.clone_index(index)
        Dnsg3, Insg3 = index3.search(self.xq, 1)
        np.testing.assert_array_equal(Dnsg3, Dnsg)
        np.testing.assert_array_equal(Insg3, Insg)

    def subtest_connectivity(self, index, nb):
        vt = faiss.VisitedTable(nb)
        count = index.nsg.dfs(vt, index.nsg.enterpoint, 0)
        self.assertEqual(count, nb)

    def subtest_add(self, build_type, thresh, metric=faiss.METRIC_L2):
        d = self.xq.shape[1]
        metrics = {faiss.METRIC_L2: 'L2',
                   faiss.METRIC_INNER_PRODUCT: 'IP'}

        flat_index = faiss.IndexFlat(d, metric)
        flat_index.add(self.xb)
        Dref, Iref = flat_index.search(self.xq, 1)

        index = faiss.IndexNSGFlat(d, 16, metric)
        index.verbose = True
        index.build_type = build_type
        index.GK = self.GK
        index.add(self.xb)
        Dnsg, Insg = index.search(self.xq, 1)

        recalls = (Iref == Insg).sum()
        print('metric: {}, nb equal: {}'.format(metrics[metric], recalls))
        self.assertGreaterEqual(recalls, thresh)
        self.subtest_connectivity(index, self.xb.shape[0])
        self.subtest_io_and_clone(index, Dnsg, Insg)

    def subtest_build(self, knn_graph, thresh, metric=faiss.METRIC_L2):
        d = self.xq.shape[1]
        metrics = {faiss.METRIC_L2: 'L2',
                   faiss.METRIC_INNER_PRODUCT: 'IP'}

        flat_index = faiss.IndexFlat(d, metric)
        flat_index.add(self.xb)
        Dref, Iref = flat_index.search(self.xq, 1)

        index = faiss.IndexNSGFlat(d, 16, metric)
        index.verbose = True

        index.build(self.xb, knn_graph)
        Dnsg, Insg = index.search(self.xq, 1)

        recalls = (Iref == Insg).sum()
        print('metric: {}, nb equal: {}'.format(metrics[metric], recalls))
        self.assertGreaterEqual(recalls, thresh)
        self.subtest_connectivity(index, self.xb.shape[0])

    def test_add_bruteforce_L2(self):
        self.subtest_add(0, 475, faiss.METRIC_L2)

    def test_add_nndescent_L2(self):
        self.subtest_add(1, 475, faiss.METRIC_L2)

    def test_add_bruteforce_IP(self):
        self.subtest_add(0, 480, faiss.METRIC_INNER_PRODUCT)

    def test_add_nndescent_IP(self):
        self.subtest_add(1, 480, faiss.METRIC_INNER_PRODUCT)

    def test_build_L2(self):
        knn_graph = self.make_knn_graph(faiss.METRIC_L2)
        self.subtest_build(knn_graph, 475, faiss.METRIC_L2)

    def test_build_IP(self):
        knn_graph = self.make_knn_graph(faiss.METRIC_INNER_PRODUCT)
        self.subtest_build(knn_graph, 480, faiss.METRIC_INNER_PRODUCT)

    def test_build_invalid_knng(self):
        """Make some invalid entries in the input knn graph.

        It would cause a warning but IndexNSG should be able
        to handel this.
        """
        knn_graph = self.make_knn_graph(faiss.METRIC_L2)
        knn_graph[:100, 5] = -111
        self.subtest_build(knn_graph, 475, faiss.METRIC_L2)

        knn_graph = self.make_knn_graph(faiss.METRIC_INNER_PRODUCT)
        knn_graph[:100, 5] = -111
        self.subtest_build(knn_graph, 480, faiss.METRIC_INNER_PRODUCT)

    def test_reset(self):
        """test IndexNSG.reset()"""
        d = self.xq.shape[1]
        metrics = {faiss.METRIC_L2: 'L2',
                   faiss.METRIC_INNER_PRODUCT: 'IP'}

        metric = faiss.METRIC_L2
        flat_index = faiss.IndexFlat(d, metric)
        flat_index.add(self.xb)
        Dref, Iref = flat_index.search(self.xq, 1)

        index = faiss.IndexNSGFlat(d, 16)
        index.verbose = True
        index.GK = 32

        index.add(self.xb)
        Dnsg, Insg = index.search(self.xq, 1)
        recalls = (Iref == Insg).sum()
        print('metric: {}, nb equal: {}'.format(metrics[metric], recalls))
        self.assertGreaterEqual(recalls, 475)
        self.subtest_connectivity(index, self.xb.shape[0])

        index.reset()
        index.add(self.xb)
        Dnsg, Insg = index.search(self.xq, 1)
        recalls = (Iref == Insg).sum()
        print('metric: {}, nb equal: {}'.format(metrics[metric], recalls))
        self.assertGreaterEqual(recalls, 475)
        self.subtest_connectivity(index, self.xb.shape[0])

    def test_order(self):
        """make sure that output results are sorted"""
        d = self.xq.shape[1]
        index = faiss.IndexNSGFlat(d, 32)

        index.train(self.xb)
        index.add(self.xb)

        k = 10
        nq = self.xq.shape[0]
        D, _ = index.search(self.xq, k)

        indices = np.argsort(D, axis=1)
        gt = np.arange(0, k)[np.newaxis, :]  # [1, k]
        gt = np.repeat(gt, nq, axis=0)  # [nq, k]
        np.testing.assert_array_equal(indices, gt)

    def test_nsg_pq(self):
        """Test IndexNSGPQ"""
        d = self.xq.shape[1]
        R, pq_M = 32, 4
        index = faiss.index_factory(d, f"NSG{R}_PQ{pq_M}")
        assert isinstance(index, faiss.IndexNSGPQ)
        idxpq = faiss.downcast_index(index.storage)
        assert index.nsg.R == R and idxpq.pq.M == pq_M

        flat_index = faiss.IndexFlat(d)
        flat_index.add(self.xb)
        Dref, Iref = flat_index.search(self.xq, k=1)

        index.GK = 32
        index.train(self.xb)
        index.add(self.xb)
        D, I = index.search(self.xq, k=1)

        # test accuracy
        recalls = (Iref == I).sum()
        print("IndexNSGPQ", recalls)
        self.assertGreaterEqual(recalls, 190)  # 193

        # test I/O
        self.subtest_io_and_clone(index, D, I)

    def test_nsg_sq(self):
        """Test IndexNSGSQ"""
        d = self.xq.shape[1]
        R = 32
        index = faiss.index_factory(d, f"NSG{R}_SQ8")
        assert isinstance(index, faiss.IndexNSGSQ)
        idxsq = faiss.downcast_index(index.storage)
        assert index.nsg.R == R
        assert idxsq.sq.qtype == faiss.ScalarQuantizer.QT_8bit

        flat_index = faiss.IndexFlat(d)
        flat_index.add(self.xb)
        Dref, Iref = flat_index.search(self.xq, k=1)

        index.train(self.xb)
        index.add(self.xb)
        D, I = index.search(self.xq, k=1)

        # test accuracy
        recalls = (Iref == I).sum()
        print("IndexNSGSQ", recalls)
        self.assertGreaterEqual(recalls, 405)  # 411

        # test I/O
        self.subtest_io_and_clone(index, D, I)


class TestNNDescent(unittest.TestCase):

    def __init__(self, *args, **kwargs):
        unittest.TestCase.__init__(self, *args, **kwargs)
        d = 32
        nt = 0
        nb = 1500
        nq = 500
        self.GK = 32

        _, self.xb, self.xq = get_dataset_2(d, nt, nb, nq)

    def test_nndescentflat(self):
        d = self.xq.shape[1]
        index = faiss.IndexNNDescentFlat(d, 32)
        index.nndescent.search_L = 8

        flat_index = faiss.IndexFlat(d)
        flat_index.add(self.xb)
        Dref, Iref = flat_index.search(self.xq, k=1)

        index.train(self.xb)
        index.add(self.xb)
        D, I = index.search(self.xq, k=1)

        # test accuracy
        recalls = (Iref == I).sum()
        print("IndexNNDescentFlat", recalls)
        self.assertGreaterEqual(recalls, 450)  # 462

        # do some IO tests
        fd, tmpfile = tempfile.mkstemp()
        os.close(fd)
        try:
            faiss.write_index(index, tmpfile)
            index2 = faiss.read_index(tmpfile)
        finally:
            if os.path.exists(tmpfile):
                os.unlink(tmpfile)

        D2, I2 = index2.search(self.xq, 1)
        np.testing.assert_array_equal(D2, D)
        np.testing.assert_array_equal(I2, I)

        # also test clone
        index3 = faiss.clone_index(index)
        D3, I3 = index3.search(self.xq, 1)
        np.testing.assert_array_equal(D3, D)
        np.testing.assert_array_equal(I3, I)

    def test_order(self):
        """make sure that output results are sorted"""
        d = self.xq.shape[1]
        index = faiss.IndexNNDescentFlat(d, 32)

        index.train(self.xb)
        index.add(self.xb)

        k = 10
        nq = self.xq.shape[0]
        D, _ = index.search(self.xq, k)

        indices = np.argsort(D, axis=1)
        gt = np.arange(0, k)[np.newaxis, :]  # [1, k]
        gt = np.repeat(gt, nq, axis=0)  # [nq, k]
        np.testing.assert_array_equal(indices, gt)


class TestDistancesPositive(unittest.TestCase):

    def test_l2_pos(self):
        """
        roundoff errors occur only with the L2 decomposition used
        with BLAS, ie. in IndexFlatL2 and with
        n > distance_compute_blas_threshold = 20
        """

        d = 128
        n = 100

        rs = np.random.RandomState(1234)
        x = rs.rand(n, d).astype('float32')

        index = faiss.IndexFlatL2(d)
        index.add(x)

        D, I = index.search(x, 10)

        assert np.all(D >= 0)


class TestShardReplicas(unittest.TestCase):
    def test_shard_flag_propagation(self):
        d = 64                           # dimension
        nb = 1000
        rs = np.random.RandomState(1234)
        xb = rs.rand(nb, d).astype('float32')
        nlist = 10
        quantizer1 = faiss.IndexFlatL2(d)
        quantizer2 = faiss.IndexFlatL2(d)
        index1 = faiss.IndexIVFFlat(quantizer1, d, nlist)
        index2 = faiss.IndexIVFFlat(quantizer2, d, nlist)

        index = faiss.IndexShards(d, True)
        index.add_shard(index1)
        index.add_shard(index2)

        self.assertFalse(index.is_trained)
        index.train(xb)
        self.assertTrue(index.is_trained)

        self.assertEqual(index.ntotal, 0)
        index.add(xb)
        self.assertEqual(index.ntotal, nb)

        index.remove_shard(index2)
        self.assertEqual(index.ntotal, nb / 2)
        index.remove_shard(index1)
        self.assertEqual(index.ntotal, 0)

    def test_replica_flag_propagation(self):
        d = 64                           # dimension
        nb = 1000
        rs = np.random.RandomState(1234)
        xb = rs.rand(nb, d).astype('float32')
        nlist = 10
        quantizer1 = faiss.IndexFlatL2(d)
        quantizer2 = faiss.IndexFlatL2(d)
        index1 = faiss.IndexIVFFlat(quantizer1, d, nlist)
        index2 = faiss.IndexIVFFlat(quantizer2, d, nlist)

        index = faiss.IndexReplicas(d, True)
        index.add_replica(index1)
        index.add_replica(index2)

        self.assertFalse(index.is_trained)
        index.train(xb)
        self.assertTrue(index.is_trained)

        self.assertEqual(index.ntotal, 0)
        index.add(xb)
        self.assertEqual(index.ntotal, nb)

        index.remove_replica(index2)
        self.assertEqual(index.ntotal, nb)
        index.remove_replica(index1)
        self.assertEqual(index.ntotal, 0)

class TestReconsException(unittest.TestCase):

    def test_recons_exception(self):

        d = 64                           # dimension
        nb = 1000
        rs = np.random.RandomState(1234)
        xb = rs.rand(nb, d).astype('float32')
        nlist = 10
        quantizer = faiss.IndexFlatL2(d)  # the other index
        index = faiss.IndexIVFFlat(quantizer, d, nlist)
        index.train(xb)
        index.add(xb)
        index.make_direct_map()

        index.reconstruct(9)

        self.assertRaises(
            RuntimeError,
            index.reconstruct, 100001
        )

    def test_reconstuct_after_add(self):
        index = faiss.index_factory(10, 'IVF5,SQfp16')
        index.train(faiss.randn((100, 10), 123))
        index.add(faiss.randn((100, 10), 345))
        index.make_direct_map()
        index.add(faiss.randn((100, 10), 678))

        # should not raise an exception
        index.reconstruct(5)
        print(index.ntotal)
        index.reconstruct(150)


class TestReconsHash(unittest.TestCase):

    def do_test(self, index_key):
        d = 32
        index = faiss.index_factory(d, index_key)
        index.train(faiss.randn((100, d), 123))

        # reference reconstruction
        index.add(faiss.randn((100, d), 345))
        index.add(faiss.randn((100, d), 678))
        ref_recons = index.reconstruct_n(0, 200)

        # with lookup
        index.reset()
        rs = np.random.RandomState(123)
        ids = rs.choice(10000, size=200, replace=False).astype(np.int64)
        index.add_with_ids(faiss.randn((100, d), 345), ids[:100])
        index.set_direct_map_type(faiss.DirectMap.Hashtable)
        index.add_with_ids(faiss.randn((100, d), 678), ids[100:])

        # compare
        for i in range(0, 200, 13):
            recons = index.reconstruct(int(ids[i]))
            self.assertTrue(np.all(recons == ref_recons[i]))

        # test I/O
        buf = faiss.serialize_index(index)
        index2 = faiss.deserialize_index(buf)

        # compare
        for i in range(0, 200, 13):
            recons = index2.reconstruct(int(ids[i]))
            self.assertTrue(np.all(recons == ref_recons[i]))

        # remove
        toremove = np.ascontiguousarray(ids[0:200:3])

        sel = faiss.IDSelectorArray(50, faiss.swig_ptr(toremove[:50]))

        # test both ways of removing elements
        nremove = index2.remove_ids(sel)
        nremove += index2.remove_ids(toremove[50:])

        self.assertEqual(nremove, len(toremove))

        for i in range(0, 200, 13):
            if i % 3 == 0:
                self.assertRaises(
                    RuntimeError,
                    index2.reconstruct, int(ids[i])
                )
            else:
                recons = index2.reconstruct(int(ids[i]))
                self.assertTrue(np.all(recons == ref_recons[i]))

        # index error should raise
        self.assertRaises(
            RuntimeError,
            index.reconstruct, 20000
        )

    def test_IVFFlat(self):
        self.do_test("IVF5,Flat")

    def test_IVFSQ(self):
        self.do_test("IVF5,SQfp16")

    def test_IVFPQ(self):
        self.do_test("IVF5,PQ4x4np")

if __name__ == '__main__':
    unittest.main()


class TestValidIndexParams(unittest.TestCase):

    def test_IndexIVFPQ(self):
        d = 32
        nb = 1000
        nt = 1500
        nq = 200

        (xt, xb, xq) = get_dataset_2(d, nt, nb, nq)

        coarse_quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFPQ(coarse_quantizer, d, 32, 8, 8)
        index.cp.min_points_per_centroid = 5    # quiet warning
        index.train(xt)
        index.add(xb)

        # invalid nprobe
        index.nprobe = 0
        k = 10
        self.assertRaises(RuntimeError, index.search, xq, k)

        # invalid k
        index.nprobe = 4
        k = -10
        self.assertRaises(AssertionError, index.search, xq, k)

        # valid params
        index.nprobe = 4
        k = 10
        D, nns = index.search(xq, k)

        self.assertEqual(D.shape[0], nq)
        self.assertEqual(D.shape[1], k)

    def test_IndexFlat(self):
        d = 32
        nb = 1000
        nt = 0
        nq = 200

        (xt, xb, xq) = get_dataset_2(d, nt, nb, nq)
        index = faiss.IndexFlat(d, faiss.METRIC_L2)

        index.add(xb)

        # invalid k
        k = -5
        self.assertRaises(AssertionError, index.search, xq, k)

        # valid k
        k = 5
        D, I = index.search(xq, k)

        self.assertEqual(D.shape[0], nq)
        self.assertEqual(D.shape[1], k)


class TestLargeRangeSearch(unittest.TestCase):

    def test_range_search(self):
        # test for https://github.com/facebookresearch/faiss/issues/1889
        d = 256
        nq = 16
        nb = 1000000

        # faiss.cvar.distance_compute_blas_threshold = 10
        faiss.omp_set_num_threads(1)

        index = faiss.IndexFlatL2(d)
        xb = np.zeros((nb, d), dtype="float32")
        index.add(xb)

        xq = np.zeros((nq, d), dtype="float32")
        lims, D, I = index.range_search(xq, 1.0)

        assert len(D) == len(xb) * len(xq)
