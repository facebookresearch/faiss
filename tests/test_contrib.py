# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import faiss
import unittest
import numpy as np
import platform

from faiss.contrib import datasets
from faiss.contrib import inspect_tools
from faiss.contrib import evaluation
from faiss.contrib import ivf_tools

from common_faiss_tests import get_dataset_2
try:
    from faiss.contrib.exhaustive_search import knn_ground_truth, knn, range_ground_truth
    from faiss.contrib.exhaustive_search import range_search_max_results

except:
    pass  # Submodule import broken in python 2.



@unittest.skipIf(platform.python_version_tuple()[0] < '3', \
                 'Submodule import broken in python 2.')
class TestComputeGT(unittest.TestCase):

    def test_compute_GT(self):
        d = 64
        xt, xb, xq = get_dataset_2(d, 0, 10000, 100)

        index = faiss.IndexFlatL2(d)
        index.add(xb)
        Dref, Iref = index.search(xq, 10)

        # iterator function on the matrix

        def matrix_iterator(xb, bs):
            for i0 in range(0, xb.shape[0], bs):
                yield xb[i0:i0 + bs]

        Dnew, Inew = knn_ground_truth(xq, matrix_iterator(xb, 1000), 10)

        np.testing.assert_array_equal(Iref, Inew)
        # decimal = 4 required when run on GPU
        np.testing.assert_almost_equal(Dref, Dnew, decimal=4)


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

        evaluation.test_ref_range_results(
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
        print(threshold)

        index = faiss.IndexFlat(32, metric)
        index.add(xb)
        ref_lims, ref_D, ref_I = index.range_search(xq, threshold)

        def matrix_iterator(xb, bs):
            for i0 in range(0, xb.shape[0], bs):
                yield xb[i0:i0 + bs]

        # check repro OK
        _, new_lims, new_D, new_I = range_search_max_results(
            index, matrix_iterator(xq, 100), threshold)

        evaluation.test_ref_range_results(
            ref_lims, ref_D, ref_I,
            new_lims, new_D, new_I
        )

        max_res = ref_lims[-1] // 2

        new_threshold, new_lims, new_D, new_I = range_search_max_results(
            index, matrix_iterator(xq, 100), threshold, max_results=max_res)

        self.assertLessEqual(new_lims[-1], max_res)

        ref_lims, ref_D, ref_I = index.range_search(xq, new_threshold)

        evaluation.test_ref_range_results(
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
        print(precision, recall)

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

        # search elements xq, increase nprobe, check 4 first results w/ groundtruth
        prev_inter_perf = 0
        for nprobe in 1, 10, 20:

            index.nprobe = nprobe
            a = alt_quantizer.search(xq[:, :20].copy(), index.nprobe)[1]
            D, I = ivf_tools.search_preassigned(index, xq, 4, a)
            inter_perf = (I == ds.get_groundtruth()[:, :4]).sum() / I.size
            self.assertTrue(inter_perf >= prev_inter_perf)
            prev_inter_perf = inter_perf

        # test range search

        index.nprobe = 20

        a = alt_quantizer.search(xq[:, :20].copy(), index.nprobe)[1]

        # just to find a reasonable radius
        D, I = ivf_tools.search_preassigned(index, xq, 4, a)
        radius = D.max() * 1.01

        lims, DR, IR = ivf_tools.range_search_preassigned(index, xq, radius, a)

        # with that radius the k-NN results are a subset of the range search results
        for q in range(len(xq)):
            l0, l1 = lims[q], lims[q + 1]
            self.assertTrue(set(I[q]) <= set(IR[l0:l1]))

    def test_binary(self):
        ds = datasets.SyntheticDataset(128, 2000, 2000, 200)

        d = ds.d
        xt = ds.get_train()
        xq = ds.get_queries()
        xb = ds.get_database()

        # define alternative quantizer on the 20 first dims of vectors (will be in float)
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

        # search elements xq, increase nprobe, check 4 first results w/ groundtruth
        prev_inter_perf = 0
        for nprobe in 1, 10, 20:

            index.nprobe = nprobe
            a = alt_quantizer.search(xq[:, :20].copy(), index.nprobe)[1]
            D, I = ivf_tools.search_preassigned(index, xq_bin, 4, a)
            inter_perf = (I == ds.get_groundtruth()[:, :4]).sum() / I.size
            self.assertTrue(inter_perf >= prev_inter_perf)
            prev_inter_perf = inter_perf

        # test range search

        index.nprobe = 20

        a = alt_quantizer.search(xq[:, :20].copy(), index.nprobe)[1]

        # just to find a reasonable radius
        D, I = ivf_tools.search_preassigned(index, xq_bin, 4, a)
        radius = int(D.max() + 1)

        lims, DR, IR = ivf_tools.range_search_preassigned(index, xq_bin, radius, a)

        # with that radius the k-NN results are a subset of the range search results
        for q in range(len(xq)):
            l0, l1 = lims[q], lims[q + 1]
            self.assertTrue(set(I[q]) <= set(IR[l0:l1]))
