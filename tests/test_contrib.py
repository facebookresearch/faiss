# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import faiss
import unittest
import numpy as np
import platform
import os
import random
import shutil
import tempfile

from faiss.contrib import datasets
from faiss.contrib import inspect_tools
from faiss.contrib import evaluation
from faiss.contrib import ivf_tools
from faiss.contrib import clustering
from faiss.contrib import big_batch_search
from faiss.contrib.ondisk import merge_ondisk

from common_faiss_tests import get_dataset_2
from faiss.contrib.exhaustive_search import \
    knn_ground_truth, knn, range_ground_truth, \
    range_search_max_results, exponential_query_iterator
from contextlib import contextmanager


class TestComputeGT(unittest.TestCase):

    def do_test_compute_GT(self, metric=faiss.METRIC_L2):
        d = 64
        xt, xb, xq = get_dataset_2(d, 0, 10000, 100)

        index = faiss.IndexFlat(d, metric)
        index.add(xb)
        Dref, Iref = index.search(xq, 10)

        # iterator function on the matrix

        def matrix_iterator(xb, bs):
            for i0 in range(0, xb.shape[0], bs):
                yield xb[i0:i0 + bs]

        Dnew, Inew = knn_ground_truth(
            xq, matrix_iterator(xb, 1000), 10, metric)

        np.testing.assert_array_equal(Iref, Inew)
        # decimal = 4 required when run on GPU
        np.testing.assert_almost_equal(Dref, Dnew, decimal=4)

    def test_compute_GT(self):
        self.do_test_compute_GT()

    def test_compute_GT_ip(self):
        self.do_test_compute_GT(faiss.METRIC_INNER_PRODUCT)


class TestDatasets(unittest.TestCase):
    """here we test only the synthetic dataset. Datasets that require
    disk or manifold access are in
    //deeplearning/projects/faiss-forge/test_faiss_datasets/:test_faiss_datasets
    """

    def test_synthetic(self):
        ds = datasets.SyntheticDataset(32, 1000, 2000, 10)
        xq = ds.get_queries()
        self.assertEqual(xq.shape, (10, 32))
        xb = ds.get_database()
        self.assertEqual(xb.shape, (2000, 32))
        ds.check_sizes()

    def test_synthetic_ip(self):
        ds = datasets.SyntheticDataset(32, 1000, 2000, 10, "IP")
        index = faiss.IndexFlatIP(32)
        index.add(ds.get_database())
        np.testing.assert_array_equal(
            ds.get_groundtruth(100),
            index.search(ds.get_queries(), 100)[1]
        )

    def test_synthetic_iterator(self):
        ds = datasets.SyntheticDataset(32, 1000, 2000, 10)
        xb = ds.get_database()
        xb2 = []
        for xbi in ds.database_iterator():
            xb2.append(xbi)
        xb2 = np.vstack(xb2)
        np.testing.assert_array_equal(xb, xb2)


class TestExhaustiveSearch(unittest.TestCase):

    def test_knn_cpu(self):
        xb = np.random.rand(200, 32).astype('float32')
        xq = np.random.rand(100, 32).astype('float32')

        index = faiss.IndexFlatL2(32)
        index.add(xb)
        Dref, Iref = index.search(xq, 10)

        Dnew, Inew = knn(xq, xb, 10)

        assert np.all(Inew == Iref)
        assert np.allclose(Dref, Dnew)

        index = faiss.IndexFlatIP(32)
        index.add(xb)
        Dref, Iref = index.search(xq, 10)

        Dnew, Inew = knn(xq, xb, 10, metric=faiss.METRIC_INNER_PRODUCT)

        assert np.all(Inew == Iref)
        assert np.allclose(Dref, Dnew)

    def do_test_range(self, metric):
        ds = datasets.SyntheticDataset(32, 0, 1000, 10)
        xq = ds.get_queries()
        xb = ds.get_database()
        D, I = faiss.knn(xq, xb, 10, metric=metric)
        threshold = float(D[:, -1].mean())

        index = faiss.IndexFlat(32, metric)
        index.add(xb)
        ref_lims, ref_D, ref_I = index.range_search(xq, threshold)

        new_lims, new_D, new_I = range_ground_truth(
            xq, ds.database_iterator(bs=100), threshold, ngpu=0,
            metric_type=metric)

        evaluation.check_ref_range_results(
            ref_lims, ref_D, ref_I,
            new_lims, new_D, new_I
        )

    def test_range_L2(self):
        self.do_test_range(faiss.METRIC_L2)

    def test_range_IP(self):
        self.do_test_range(faiss.METRIC_INNER_PRODUCT)

    def test_query_iterator(self, metric=faiss.METRIC_L2):
        ds = datasets.SyntheticDataset(32, 0, 1000, 1000)
        xq = ds.get_queries()
        xb = ds.get_database()
        D, I = faiss.knn(xq, xb, 10, metric=metric)
        threshold = float(D[:, -1].mean())

        index = faiss.IndexFlat(32, metric)
        index.add(xb)
        ref_lims, ref_D, ref_I = index.range_search(xq, threshold)

        def matrix_iterator(xb, bs):
            for i0 in range(0, xb.shape[0], bs):
                yield xb[i0:i0 + bs]

        # check repro OK
        _, new_lims, new_D, new_I = range_search_max_results(
            index, matrix_iterator(xq, 100), threshold, max_results=1e10)

        evaluation.check_ref_range_results(
            ref_lims, ref_D, ref_I,
            new_lims, new_D, new_I
        )

        max_res = ref_lims[-1] // 2

        new_threshold, new_lims, new_D, new_I = range_search_max_results(
            index, matrix_iterator(xq, 100), threshold, max_results=max_res)

        self.assertLessEqual(new_lims[-1], max_res)

        ref_lims, ref_D, ref_I = index.range_search(xq, new_threshold)

        evaluation.check_ref_range_results(
            ref_lims, ref_D, ref_I,
            new_lims, new_D, new_I
        )


class TestInspect(unittest.TestCase):

    def test_LinearTransform(self):
        # training data
        xt = np.random.rand(1000, 20).astype('float32')
        # test data
        x = np.random.rand(10, 20).astype('float32')
        # make the PCA matrix
        pca = faiss.PCAMatrix(20, 10)
        pca.train(xt)
        # apply it to test data
        yref = pca.apply_py(x)

        A, b = inspect_tools.get_LinearTransform_matrix(pca)

        # verify
        ynew = x @ A.T + b
        np.testing.assert_array_almost_equal(yref, ynew)

    def test_IndexFlat(self):
        xb = np.random.rand(13, 20).astype('float32')
        index = faiss.IndexFlatL2(20)
        index.add(xb)
        np.testing.assert_array_equal(
            xb, inspect_tools.get_flat_data(index)
        )

    def test_make_LT(self):
        rs = np.random.RandomState(123)
        X = rs.rand(13, 20).astype('float32')
        A = rs.rand(5, 20).astype('float32')
        b = rs.rand(5).astype('float32')
        Yref = X @ A.T + b
        lt = inspect_tools.make_LinearTransform_matrix(A, b)
        Ynew = lt.apply(X)
        np.testing.assert_allclose(Yref, Ynew, rtol=1e-06)

    def test_NSG_neighbors(self):
        # FIXME number of elements to add should be >> 100
        ds = datasets.SyntheticDataset(32, 0, 200, 10)
        index = faiss.index_factory(ds.d, "NSG")
        index.add(ds.get_database())
        neighbors = inspect_tools.get_NSG_neighbors(index.nsg)
        # neighbors should be either valid indexes or -1
        np.testing.assert_array_less(-2, neighbors)
        np.testing.assert_array_less(neighbors, ds.nb)


class TestRangeEval(unittest.TestCase):

    def test_precision_recall(self):
        Iref = [
            [1, 2, 3],
            [5, 6],
            [],
            []
        ]
        Inew = [
            [1, 2],
            [6, 7],
            [1],
            []
        ]

        lims_ref = np.cumsum([0] + [len(x) for x in Iref])
        Iref = np.hstack(Iref)
        lims_new = np.cumsum([0] + [len(x) for x in Inew])
        Inew = np.hstack(Inew)

        precision, recall = evaluation.range_PR(lims_ref, Iref, lims_new, Inew)

        self.assertEqual(precision, 0.6)
        self.assertEqual(recall, 0.6)

    def test_PR_multiple(self):
        metric = faiss.METRIC_L2
        ds = datasets.SyntheticDataset(32, 1000, 1000, 10)
        xq = ds.get_queries()
        xb = ds.get_database()

        # good for ~10k results
        threshold = 15

        index = faiss.IndexFlat(32, metric)
        index.add(xb)
        ref_lims, ref_D, ref_I = index.range_search(xq, threshold)

        # now make a slightly suboptimal index
        index2 = faiss.index_factory(32, "PCA16,Flat")
        index2.train(ds.get_train())
        index2.add(xb)

        # PCA reduces distances so will have more results
        new_lims, new_D, new_I = index2.range_search(xq, threshold)

        all_thr = np.array([5.0, 10.0, 12.0, 15.0])
        for mode in "overall", "average":
            ref_precisions = np.zeros_like(all_thr)
            ref_recalls = np.zeros_like(all_thr)

            for i, thr in enumerate(all_thr):

                lims2, _, I2 = evaluation.filter_range_results(
                    new_lims, new_D, new_I, thr)

                prec, recall = evaluation.range_PR(
                    ref_lims, ref_I, lims2, I2, mode=mode)

                ref_precisions[i] = prec
                ref_recalls[i] = recall

            precisions, recalls = evaluation.range_PR_multiple_thresholds(
                ref_lims, ref_I,
                new_lims, new_D, new_I, all_thr,
                mode=mode
            )

            np.testing.assert_array_almost_equal(ref_precisions, precisions)
            np.testing.assert_array_almost_equal(ref_recalls, recalls)


class TestPreassigned(unittest.TestCase):

    def test_index_pretransformed(self):

        ds = datasets.SyntheticDataset(128, 2000, 2000, 200)
        xt = ds.get_train()
        xq = ds.get_queries()
        xb = ds.get_database()
        index = faiss.index_factory(128, 'PCA64,IVF64,PQ4np')
        index.train(xt)
        index.add(xb)
        index_downcasted = faiss.extract_index_ivf(index)
        index_downcasted.nprobe = 10
        xq_trans = index.chain.at(0).apply_py(xq)
        D_ref, I_ref = index.search(xq, 4)

        quantizer = index_downcasted.quantizer
        Dq, Iq = quantizer.search(xq_trans, index_downcasted.nprobe)
        D, I = ivf_tools.search_preassigned(index, xq, 4, Iq, Dq)
        np.testing.assert_almost_equal(D_ref, D, decimal=4)
        np.testing.assert_array_equal(I_ref, I)

    def test_float(self):
        ds = datasets.SyntheticDataset(128, 2000, 2000, 200)

        d = ds.d
        xt = ds.get_train()
        xq = ds.get_queries()
        xb = ds.get_database()

        # define alternative quantizer on the 20 first dims of vectors
        km = faiss.Kmeans(20, 50)
        km.train(xt[:, :20].copy())
        alt_quantizer = km.index

        index = faiss.index_factory(d, "IVF50,PQ16np")
        index.by_residual = False

        # (optional) fake coarse quantizer
        fake_centroids = np.zeros((index.nlist, index.d), dtype="float32")
        index.quantizer.add(fake_centroids)

        # train the PQ part
        index.train(xt)

        # add elements xb
        a = alt_quantizer.search(xb[:, :20].copy(), 1)[1].ravel()
        ivf_tools.add_preassigned(index, xb, a)

        # search elements xq, increase nprobe, check 4 first results w/
        # groundtruth
        prev_inter_perf = 0
        for nprobe in 1, 10, 20:

            index.nprobe = nprobe
            a = alt_quantizer.search(xq[:, :20].copy(), index.nprobe)[1]
            D, I = ivf_tools.search_preassigned(index, xq, 4, a)
            inter_perf = faiss.eval_intersection(
                I, ds.get_groundtruth()[:, :4])
            self.assertTrue(inter_perf >= prev_inter_perf)
            prev_inter_perf = inter_perf

        # test range search

        index.nprobe = 20

        a = alt_quantizer.search(xq[:, :20].copy(), index.nprobe)[1]

        # just to find a reasonable radius
        D, I = ivf_tools.search_preassigned(index, xq, 4, a)
        radius = D.max() * 1.01

        lims, DR, IR = ivf_tools.range_search_preassigned(index, xq, radius, a)

        # with that radius the k-NN results are a subset of the range search
        # results
        for q in range(len(xq)):
            l0, l1 = lims[q], lims[q + 1]
            self.assertTrue(set(I[q]) <= set(IR[l0:l1]))

    def test_binary(self):
        ds = datasets.SyntheticDataset(128, 2000, 2000, 200)

        d = ds.d
        xt = ds.get_train()
        xq = ds.get_queries()
        xb = ds.get_database()

        # define alternative quantizer on the 20 first dims of vectors
        # (will be in float)
        km = faiss.Kmeans(20, 50)
        km.train(xt[:, :20].copy())
        alt_quantizer = km.index

        binarizer = faiss.index_factory(d, "ITQ,LSHt")
        binarizer.train(xt)

        xb_bin = binarizer.sa_encode(xb)
        xq_bin = binarizer.sa_encode(xq)

        index = faiss.index_binary_factory(d, "BIVF200")

        fake_centroids = np.zeros((index.nlist, index.d // 8), dtype="uint8")
        index.quantizer.add(fake_centroids)
        index.is_trained = True

        # add elements xb
        a = alt_quantizer.search(xb[:, :20].copy(), 1)[1].ravel()
        ivf_tools.add_preassigned(index, xb_bin, a)

        # recompute GT in binary
        k = 15
        ib = faiss.IndexBinaryFlat(128)
        ib.add(xb_bin)
        Dgt, Igt = ib.search(xq_bin, k)

        # search elements xq, increase nprobe, check 4 first results w/
        # groundtruth
        prev_inter_perf = 0
        for nprobe in 1, 10, 20:

            index.nprobe = nprobe
            a = alt_quantizer.search(xq[:, :20].copy(), index.nprobe)[1]
            D, I = ivf_tools.search_preassigned(index, xq_bin, k, a)
            inter_perf = faiss.eval_intersection(I, Igt)
            self.assertGreaterEqual(inter_perf, prev_inter_perf)
            prev_inter_perf = inter_perf

        # test range search

        index.nprobe = 20

        a = alt_quantizer.search(xq[:, :20].copy(), index.nprobe)[1]

        # just to find a reasonable radius
        D, I = ivf_tools.search_preassigned(index, xq_bin, 4, a)
        radius = int(D.max() + 1)

        lims, DR, IR = ivf_tools.range_search_preassigned(
            index, xq_bin, radius, a)

        # with that radius the k-NN results are a subset of the range
        # search results
        for q in range(len(xq)):
            l0, l1 = lims[q], lims[q + 1]
            self.assertTrue(set(I[q]) <= set(IR[l0:l1]))


class TestRangeSearchMaxResults(unittest.TestCase):

    def do_test(self, metric_type):
        ds = datasets.SyntheticDataset(32, 0, 1000, 200)
        index = faiss.IndexFlat(ds.d, metric_type)
        index.add(ds.get_database())

        # find a reasonable radius
        D, _ = index.search(ds.get_queries(), 10)
        radius0 = float(np.median(D[:, -1]))

        # baseline = search with that radius
        lims_ref, Dref, Iref = index.range_search(ds.get_queries(), radius0)

        # now see if using just the total number of results, we can get back
        # the same result table
        query_iterator = exponential_query_iterator(ds.get_queries())

        init_radius = 1e10 if metric_type == faiss.METRIC_L2 else -1e10
        radius1, lims_new, Dnew, Inew = range_search_max_results(
            index, query_iterator, init_radius,
            min_results=Dref.size, clip_to_min=True
        )

        evaluation.check_ref_range_results(
            lims_ref, Dref, Iref,
            lims_new, Dnew, Inew
        )

    def test_L2(self):
        self.do_test(faiss.METRIC_L2)

    def test_IP(self):
        self.do_test(faiss.METRIC_INNER_PRODUCT)

    def test_binary(self):
        ds = datasets.SyntheticDataset(64, 1000, 1000, 200)
        tobinary = faiss.index_factory(ds.d, "LSHrt")
        tobinary.train(ds.get_train())
        index = faiss.IndexBinaryFlat(ds.d)
        xb = tobinary.sa_encode(ds.get_database())
        xq = tobinary.sa_encode(ds.get_queries())
        index.add(xb)

        # find a reasonable radius
        D, _ = index.search(xq, 10)
        radius0 = int(np.median(D[:, -1]))

        # baseline = search with that radius
        lims_ref, Dref, Iref = index.range_search(xq, radius0)

        # now see if using just the total number of results, we can get back
        # the same result table
        query_iterator = exponential_query_iterator(xq)

        radius1, lims_new, Dnew, Inew = range_search_max_results(
            index, query_iterator, ds.d // 2,
            min_results=Dref.size, clip_to_min=True
        )

        evaluation.check_ref_range_results(
            lims_ref, Dref, Iref,
            lims_new, Dnew, Inew
        )


class TestClustering(unittest.TestCase):

    def test_python_kmeans(self):
        """ Test the python implementation of kmeans """
        ds = datasets.SyntheticDataset(32, 10000, 0, 0)
        x = ds.get_train()

        # bad distribution to stress-test split code
        xt = x[:10000].copy()
        xt[:5000] = x[0]

        km_ref = faiss.Kmeans(ds.d, 100, niter=10)
        km_ref.train(xt)
        err = faiss.knn(xt, km_ref.centroids, 1)[0].sum()

        data = clustering.DatasetAssign(xt)
        centroids = clustering.kmeans(100, data, 10)
        err2 = faiss.knn(xt, centroids, 1)[0].sum()

        # err=33498.332 err2=33380.477
        self.assertLess(err2, err * 1.1)

    def test_2level(self):
        " verify that 2-level clustering is not too sub-optimal "
        ds = datasets.SyntheticDataset(32, 10000, 0, 0)
        xt = ds.get_train()
        km_ref = faiss.Kmeans(ds.d, 100)
        km_ref.train(xt)
        err = faiss.knn(xt, km_ref.centroids, 1)[0].sum()

        centroids2, _ = clustering.two_level_clustering(xt, 10, 100)
        err2 = faiss.knn(xt, centroids2, 1)[0].sum()

        self.assertLess(err2, err * 1.1)

    def test_ivf_train_2level(self):
        " check 2-level clustering with IVF training "
        ds = datasets.SyntheticDataset(32, 10000, 1000, 200)
        index = faiss.index_factory(ds.d, "PCA16,IVF100,SQ8")
        faiss.extract_index_ivf(index).nprobe = 10
        index.train(ds.get_train())
        index.add(ds.get_database())
        Dref, Iref = index.search(ds.get_queries(), 1)

        index = faiss.index_factory(ds.d, "PCA16,IVF100,SQ8")
        faiss.extract_index_ivf(index).nprobe = 10
        clustering.train_ivf_index_with_2level(
            index, ds.get_train(), verbose=True, rebalance=False)
        index.add(ds.get_database())
        Dnew, Inew = index.search(ds.get_queries(), 1)

        # normally 47 / 200 differences
        ndiff = (Iref != Inew).sum()
        self.assertLess(ndiff, 51)


class TestBigBatchSearch(unittest.TestCase):

    def do_test(self, factory_string, metric=faiss.METRIC_L2):
        # ds = datasets.SyntheticDataset(32, 2000, 4000, 1000)
        ds = datasets.SyntheticDataset(32, 2000, 400, 500)
        k = 10
        index = faiss.index_factory(ds.d, factory_string, metric)
        assert index.metric_type == metric
        index.train(ds.get_train())
        index.add(ds.get_database())
        index.nprobe = 5
        Dref, Iref = index.search(ds.get_queries(), k)
        # faiss.omp_set_num_threads(1)
        for method in ("pairwise_distances", "knn_function", "index"):
            for threaded in 0, 1, 2:
                Dnew, Inew = big_batch_search.big_batch_search(
                    index, ds.get_queries(),
                    k, method=method,
                    threaded=threaded
                )
                self.assertLess((Inew != Iref).sum() / Iref.size, 1e-4)
                np.testing.assert_almost_equal(Dnew, Dref, decimal=4)

    def test_Flat(self):
        self.do_test("IVF64,Flat")

    def test_Flat_IP(self):
        self.do_test("IVF64,Flat", metric=faiss.METRIC_INNER_PRODUCT)

    def test_PQ(self):
        self.do_test("IVF64,PQ4np")

    def test_SQ(self):
        self.do_test("IVF64,SQ8")

    def test_checkpoint(self):
        ds = datasets.SyntheticDataset(32, 2000, 400, 500)
        k = 10
        index = faiss.index_factory(ds.d, "IVF64,SQ8")
        index.train(ds.get_train())
        index.add(ds.get_database())
        index.nprobe = 5
        Dref, Iref = index.search(ds.get_queries(), k)

        checkpoint = tempfile.mktemp()
        try:
            # First big batch search
            try:
                Dnew, Inew = big_batch_search.big_batch_search(
                    index, ds.get_queries(),
                    k, method="knn_function",
                    threaded=2,
                    checkpoint=checkpoint, checkpoint_freq=0.1,
                    crash_at=20
                )
            except ZeroDivisionError:
                pass
            else:
                self.assertFalse("should have crashed")
            # Second big batch search
            Dnew, Inew = big_batch_search.big_batch_search(
                index, ds.get_queries(),
                k, method="knn_function",
                threaded=2,
                checkpoint=checkpoint, checkpoint_freq=5
            )
            self.assertLess((Inew != Iref).sum() / Iref.size, 1e-4)
            np.testing.assert_almost_equal(Dnew, Dref, decimal=4)
        finally:
            if os.path.exists(checkpoint):
                os.unlink(checkpoint)


class TestInvlistSort(unittest.TestCase):

    def test_sort(self):
        """ make sure that the search results do not change
        after sorting the inverted lists """
        ds = datasets.SyntheticDataset(32, 2000, 200, 20)
        index = faiss.index_factory(ds.d, "IVF50,SQ8")
        index.train(ds.get_train())
        index.add(ds.get_database())
        index.nprobe = 5
        Dref, Iref = index.search(ds.get_queries(), 5)

        ivf_tools.sort_invlists_by_size(index)
        list_sizes = ivf_tools.get_invlist_sizes(index.invlists)
        assert np.all(list_sizes[1:] >= list_sizes[:-1])

        Dnew, Inew = index.search(ds.get_queries(), 5)
        np.testing.assert_equal(Dnew, Dref)
        np.testing.assert_equal(Inew, Iref)

    def test_hnsw_permute(self):
        """ make sure HNSW permutation works (useful when used as coarse quantizer) """
        ds = datasets.SyntheticDataset(32, 0, 1000, 50)
        index = faiss.index_factory(ds.d, "HNSW32,Flat")
        index.add(ds.get_database())
        Dref, Iref = index.search(ds.get_queries(), 5)
        rs = np.random.RandomState(1234)
        perm = rs.permutation(index.ntotal)
        index.permute_entries(perm)
        Dnew, Inew = index.search(ds.get_queries(), 5)
        np.testing.assert_equal(Dnew, Dref)
        Inew_remap = perm[Inew]
        np.testing.assert_equal(Inew_remap, Iref)


class TestCodeSet(unittest.TestCase):

    def test_code_set(self):
        """ CodeSet and np.unique should produce the same output """
        d = 8
        n = 1000  # > 256 and using only 0 or 1 so there must be duplicates
        codes = np.random.randint(0, 2, (n, d), dtype=np.uint8)
        s = faiss.CodeSet(d)
        inserted = s.insert(codes)
        np.testing.assert_equal(
            np.sort(np.unique(codes, axis=0), axis=None),
            np.sort(codes[inserted], axis=None))


@unittest.skipIf(platform.system() == 'Windows',
                'OnDiskInvertedLists is unsupported on Windows.')
class TestMerge(unittest.TestCase):
    @contextmanager
    def temp_directory(self):
        temp_dir = tempfile.mkdtemp()
        try:
            yield temp_dir
        finally:
            shutil.rmtree(temp_dir)

    def do_test_ondisk_merge(self, shift_ids=False):
        with self.temp_directory() as tmpdir:
            # only train and add index to disk without adding elements.
            # this will create empty inverted lists.
            ds = datasets.SyntheticDataset(32, 2000, 200, 20)
            index = faiss.index_factory(ds.d, "IVF32,Flat")
            index.train(ds.get_train())
            faiss.write_index(index, tmpdir + "/trained.index")

            # create 4 shards and add elements to them
            ns = 4  # number of shards

            for bno in range(ns):
                index = faiss.read_index(tmpdir + "/trained.index")
                i0, i1 = int(bno * ds.nb / ns), int((bno + 1) * ds.nb / ns)
                if shift_ids:
                    index.add_with_ids(ds.xb[i0:i1], np.arange(0, ds.nb / ns))
                else:
                    index.add_with_ids(ds.xb[i0:i1], np.arange(i0, i1))
                faiss.write_index(index, tmpdir + "/block_%d.index" % bno)

            # construct the output index and merge them on disk
            index = faiss.read_index(tmpdir + "/trained.index")
            block_fnames = [tmpdir + "/block_%d.index" % bno for bno in range(4)]

            merge_ondisk(
                index, block_fnames, tmpdir + "/merged_index.ivfdata", shift_ids
            )
            faiss.write_index(index, tmpdir + "/populated.index")

            # perform a search from index on disk
            index = faiss.read_index(tmpdir + "/populated.index")
            index.nprobe = 5
            D, I = index.search(ds.xq, 5)

            # ground-truth
            gtI = ds.get_groundtruth(5)

            recall_at_1 = (I[:, :1] == gtI[:, :1]).sum() / float(ds.xq.shape[0])
            self.assertGreaterEqual(recall_at_1, 0.5)

    def test_ondisk_merge(self):
        self.do_test_ondisk_merge()

    def test_ondisk_merge_with_shift_ids(self):
        # verified that recall is same for test_ondisk_merge and
        self.do_test_ondisk_merge(True)
