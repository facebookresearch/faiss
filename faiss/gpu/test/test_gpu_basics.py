# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function, unicode_literals

import unittest
import numpy as np
import faiss
import random
from common_faiss_tests import get_dataset_2


class ReferencedObject(unittest.TestCase):

    d = 16
    xb = np.random.rand(256, d).astype('float32')
    nlist = 128

    d_bin = 256
    xb_bin = np.random.randint(256, size=(10000, d_bin // 8)).astype('uint8')
    xq_bin = np.random.randint(256, size=(1000, d_bin // 8)).astype('uint8')

    def test_proxy(self):
        index = faiss.IndexReplicas()
        for _i in range(3):
            sub_index = faiss.IndexFlatL2(self.d)
            sub_index.add(self.xb)
            index.addIndex(sub_index)
        assert index.d == self.d
        index.search(self.xb, 10)

    def test_resources(self):
        # this used to crash!
        index = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0,
                                       faiss.IndexFlatL2(self.d))
        index.add(self.xb)

    def test_flat(self):
        index = faiss.GpuIndexFlat(faiss.StandardGpuResources(),
                                   self.d, faiss.METRIC_L2)
        index.add(self.xb)

    def test_ivfflat(self):
        index = faiss.GpuIndexIVFFlat(
            faiss.StandardGpuResources(),
            self.d, self.nlist, faiss.METRIC_L2)
        index.train(self.xb)

    def test_ivfpq(self):
        index_cpu = faiss.IndexIVFPQ(
            faiss.IndexFlatL2(self.d),
            self.d, self.nlist, 2, 8)
        # speed up test
        index_cpu.pq.cp.niter = 2
        index_cpu.do_polysemous_training = False
        index_cpu.train(self.xb)

        index = faiss.GpuIndexIVFPQ(
            faiss.StandardGpuResources(), index_cpu)
        index.add(self.xb)

    def test_binary_flat(self):
        k = 10

        index_ref = faiss.IndexBinaryFlat(self.d_bin)
        index_ref.add(self.xb_bin)
        D_ref, I_ref = index_ref.search(self.xq_bin, k)

        index = faiss.GpuIndexBinaryFlat(faiss.StandardGpuResources(),
                                         self.d_bin)
        index.add(self.xb_bin)
        D, I = index.search(self.xq_bin, k)

        for d_ref, i_ref, d_new, i_new in zip(D_ref, I_ref, D, I):
            # exclude max distance
            assert d_ref.max() == d_new.max()
            dmax = d_ref.max()

            # sort by (distance, id) pairs to be reproducible
            ref = [(d, i) for d, i in zip(d_ref, i_ref) if d < dmax]
            ref.sort()

            new = [(d, i) for d, i in zip(d_new, i_new) if d < dmax]
            new.sort()

            assert ref == new

    def test_stress(self):
        # a mixture of the above, from issue #631
        target = np.random.rand(50, 16).astype('float32')

        index = faiss.IndexReplicas()
        size, dim = target.shape
        num_gpu = 4
        for _i in range(num_gpu):
            config = faiss.GpuIndexFlatConfig()
            config.device = 0   # simulate on a single GPU
            sub_index = faiss.GpuIndexFlatIP(faiss.StandardGpuResources(), dim, config)
            index.addIndex(sub_index)

        index = faiss.IndexIDMap(index)
        ids = np.arange(size)
        index.add_with_ids(target, ids)



class TestGPUKmeans(unittest.TestCase):

    def test_kmeans(self):
        d = 32
        nb = 1000
        k = 10
        rs = np.random.RandomState(123)
        xb = rs.rand(nb, d).astype('float32')

        km1 = faiss.Kmeans(d, k)
        obj1 = km1.train(xb)

        km2 = faiss.Kmeans(d, k, gpu=True)
        obj2 = km2.train(xb)

        print(obj1, obj2)
        assert np.allclose(obj1, obj2)

    def test_progressive_dim(self):
        d = 32
        n = 10000
        k = 50
        xt, _, _ = get_dataset_2(d, n, 0, 0)

        # basic kmeans
        kmeans = faiss.Kmeans(d, k, gpu=True)
        kmeans.train(xt)

        pca = faiss.PCAMatrix(d, d)
        pca.train(xt)
        xt_pca = pca.apply(xt)

        # same test w/ Kmeans wrapper
        kmeans2 = faiss.Kmeans(d, k, progressive_dim_steps=5, gpu=True)
        kmeans2.train(xt_pca)
        self.assertLess(kmeans2.obj[-1], kmeans.obj[-1])


class TestAlternativeDistances(unittest.TestCase):

    def do_test(self, metric, metric_arg=0):
        res = faiss.StandardGpuResources()
        d = 32
        nb = 1000
        nq = 100

        rs = np.random.RandomState(123)
        xb = rs.rand(nb, d).astype('float32')
        xq = rs.rand(nq, d).astype('float32')

        index_ref = faiss.IndexFlat(d, metric)
        index_ref.metric_arg = metric_arg
        index_ref.add(xb)
        Dref, Iref = index_ref.search(xq, 10)

        # build from other index
        index = faiss.GpuIndexFlat(res, index_ref)
        Dnew, Inew = index.search(xq, 10)
        np.testing.assert_array_equal(Inew, Iref)
        np.testing.assert_allclose(Dnew, Dref, rtol=1e-6)

        #  build from scratch
        index = faiss.GpuIndexFlat(res, d, metric)
        index.metric_arg = metric_arg
        index.add(xb)

        Dnew, Inew = index.search(xq, 10)
        np.testing.assert_array_equal(Inew, Iref)

    @unittest.skipIf(
        "CUVS" in faiss.get_compile_options(),
        "only if CUVS is not compiled in")
    def do_test_gower(self):
        """Special test for Gower distance with mixed numeric/categorical data"""
        res = faiss.StandardGpuResources()
        d = 32
        nb = 1000
        nq = 100

        rs = np.random.RandomState(123)

        # Create mixed data: first half numeric [0,1], second half categorical (negative)
        xb = np.zeros((nb, d), dtype='float32')
        xq = np.zeros((nq, d), dtype='float32')

        # Numeric features in [0,1] range
        xb[:, :d//2] = rs.rand(nb, d//2)
        xq[:, :d//2] = rs.rand(nq, d//2)

        # Categorical features (negative integers)
        xb[:, d//2:] = -rs.randint(1, 5, size=(nb, d//2)).astype('float32')
        xq[:, d//2:] = -rs.randint(1, 5, size=(nq, d//2)).astype('float32')

        # CPU reference
        index_ref = faiss.IndexFlat(d, faiss.METRIC_GOWER)
        index_ref.add(xb)
        Dref, Iref = index_ref.search(xq, 10)

        # GPU test: build from CPU index
        index = faiss.GpuIndexFlat(res, index_ref)
        Dnew, Inew = index.search(xq, 10)
        np.testing.assert_array_equal(Inew, Iref)
        np.testing.assert_allclose(Dnew, Dref, rtol=1e-6)

        # GPU test: build from scratch
        index = faiss.GpuIndexFlat(res, d, faiss.METRIC_GOWER)
        index.add(xb)
        Dnew, Inew = index.search(xq, 10)
        np.testing.assert_array_equal(Inew, Iref)
        np.testing.assert_allclose(Dnew, Dref, rtol=1e-6)

    def test_L1(self):
        self.do_test(faiss.METRIC_L1)

    def test_Linf(self):
        self.do_test(faiss.METRIC_Linf)

    def test_Lp(self):
        self.do_test(faiss.METRIC_Lp, 0.7)

    def test_gower(self):
        self.do_test_gower()


class TestGpuRef(unittest.TestCase):

    def test_gpu_ref(self):
        # this crashes
        dim = 256
        training_data = np.random.randint(256, size=(10000, dim // 8)).astype('uint8')
        centroids = 330

        def create_cpu(dim):
            quantizer = faiss.IndexBinaryFlat(dim)
            return faiss.IndexBinaryIVF(quantizer, dim, centroids)

        def create_gpu(dim):
            gpu_quantizer = faiss.index_cpu_to_all_gpus(faiss.IndexFlatL2(dim))

            index = create_cpu(dim)
            index.clustering_index = gpu_quantizer
            index.dont_dealloc_me = gpu_quantizer
            return index

        index = create_gpu(dim)

        index.verbose = True
        index.cp.verbose = True

        index.train(training_data)

def make_t(num, d, clamp=False, seed=None):
    rs = None
    if seed is None:
        rs = np.random.RandomState(123)
    else:
        rs = np.random.RandomState(seed)

    x = rs.rand(num, d).astype(np.float32)
    if clamp:
        x = (x * 255).astype('uint8').astype('float32')
    return x

class TestKnn(unittest.TestCase):
    def test_input_types(self):
        self.do_test_input_types(0, 0)

    def test_input_types_tiling(self):
        self.do_test_input_types(0, 500)
        self.do_test_input_types(1000, 0)
        self.do_test_input_types(1000, 500)

    def do_test_input_types(self, vectorsMemoryLimit, queriesMemoryLimit):
        d = 33
        k = 5
        nb = 1000
        nq = 10

        xs = make_t(nb, d)
        qs = make_t(nq, d)

        res = faiss.StandardGpuResources()

        # Get ground truth using IndexFlat
        index = faiss.IndexFlatL2(d)
        index.add(xs)
        ref_d, ref_i = index.search(qs, k)

        out_d = np.empty((nq, k), dtype=np.float32)
        out_i = np.empty((nq, k), dtype=np.int64)

        gpu_id = random.randrange(0, faiss.get_num_gpus())

        # Try f32 data/queries, i64 out indices
        params = faiss.GpuDistanceParams()
        params.k = k
        params.dims = d
        params.vectors = faiss.swig_ptr(xs)
        params.numVectors = nb
        params.queries = faiss.swig_ptr(qs)
        params.numQueries = nq
        params.outDistances = faiss.swig_ptr(out_d)
        params.outIndices = faiss.swig_ptr(out_i)
        params.device = gpu_id

        if vectorsMemoryLimit > 0 or queriesMemoryLimit > 0:
            faiss.bfKnn_tiling(
                res,
                params,
                vectorsMemoryLimit,
                queriesMemoryLimit)
        else:
            faiss.bfKnn(res, params)

        np.testing.assert_allclose(ref_d, out_d, atol=1e-5)
        np.testing.assert_array_equal(out_i, ref_i)

        faiss.knn_gpu(
            res, qs, xs, k, out_d, out_i, device=gpu_id,
            vectorsMemoryLimit=vectorsMemoryLimit,
            queriesMemoryLimit=queriesMemoryLimit)

        np.testing.assert_allclose(ref_d, out_d, atol=1e-5)
        np.testing.assert_array_equal(out_i, ref_i)

        # Try int32 out indices
        out_i32 = np.empty((nq, k), dtype=np.int32)
        params.outIndices = faiss.swig_ptr(out_i32)
        params.outIndicesType = faiss.IndicesDataType_I32

        faiss.bfKnn(res, params)

        np.testing.assert_allclose(ref_d, out_d, atol=1e-5)
        np.testing.assert_array_equal(out_i32, ref_i)

        # Try float16 data/queries, i64 out indices
        xs_f16 = xs.astype(np.float16)
        qs_f16 = qs.astype(np.float16)
        xs_f16_f32 = xs_f16.astype(np.float32)
        qs_f16_f32 = qs_f16.astype(np.float32)
        index.reset()
        index.add(xs_f16_f32)
        ref_d_f16, ref_i_f16 = index.search(qs_f16_f32, k)

        params.vectors = faiss.swig_ptr(xs_f16)
        params.vectorType = faiss.DistanceDataType_F16
        params.queries = faiss.swig_ptr(qs_f16)
        params.queryType = faiss.DistanceDataType_F16
        params.device = random.randrange(0, faiss.get_num_gpus())

        out_d_f16 = np.empty((nq, k), dtype=np.float32)
        out_i_f16 = np.empty((nq, k), dtype=np.int64)

        params.outDistances = faiss.swig_ptr(out_d_f16)
        params.outIndices = faiss.swig_ptr(out_i_f16)
        params.outIndicesType = faiss.IndicesDataType_I64
        params.device = random.randrange(0, faiss.get_num_gpus())

        faiss.bfKnn(res, params)

        self.assertGreaterEqual((out_i_f16 == ref_i_f16).sum(), ref_i_f16.size - 5)
        np.testing.assert_allclose(ref_d_f16, out_d_f16, atol = 2e-3)

class TestAllPairwiseDistance(unittest.TestCase):
    def test_dist(self):
        metrics = [
            faiss.METRIC_L2,
            faiss.METRIC_INNER_PRODUCT,
            faiss.METRIC_L1,
            faiss.METRIC_Linf,
            faiss.METRIC_Canberra,
            faiss.METRIC_BrayCurtis,
            faiss.METRIC_JensenShannon,
            faiss.METRIC_Jaccard
        ]

        for metric in metrics:
            print(metric)
            d = 33
            k = 500

            # all pairwise distance should be the same as nb = k
            nb = k
            nq = 20

            xs = make_t(nb, d)
            qs = make_t(nq, d)

            res = faiss.StandardGpuResources()

            # Get ground truth using IndexFlat
            index = faiss.IndexFlat(d, metric)
            index.add(xs)
            ref_d, _ = index.search(qs, k)

            out_d = np.empty((nq, k), dtype=np.float32)

            # Try f32 data/queries
            params = faiss.GpuDistanceParams()
            params.metric = metric
            params.k = -1 # all pairwise
            params.dims = d
            params.vectors = faiss.swig_ptr(xs)
            params.numVectors = nb
            params.queries = faiss.swig_ptr(qs)
            params.numQueries = nq
            params.outDistances = faiss.swig_ptr(out_d)
            params.device = random.randrange(0, faiss.get_num_gpus())

            faiss.bfKnn(res, params)

            # IndexFlat will sort the results, so we need to
            # do the same on our end
            out_d = np.sort(out_d, axis=1)

            # INNER_PRODUCT is in descending order, make sure it is the same
            # order
            if faiss.is_similarity_metric(metric):
                ref_d = np.sort(ref_d, axis=1)

            print('f32', np.abs(ref_d - out_d).max())

            np.testing.assert_allclose(ref_d, out_d, atol=1e-5)

            # Try float16 data/queries
            xs_f16 = xs.astype(np.float16)
            qs_f16 = qs.astype(np.float16)
            xs_f16_f32 = xs_f16.astype(np.float32)
            qs_f16_f32 = qs_f16.astype(np.float32)
            index.reset()
            index.add(xs_f16_f32)
            ref_d_f16, _ = index.search(qs_f16_f32, k)

            params.vectors = faiss.swig_ptr(xs_f16)
            params.vectorType = faiss.DistanceDataType_F16
            params.queries = faiss.swig_ptr(qs_f16)
            params.queryType = faiss.DistanceDataType_F16

            out_d_f16 = np.empty((nq, k), dtype=np.float32)
            params.outDistances = faiss.swig_ptr(out_d_f16)
            params.device = random.randrange(0, faiss.get_num_gpus())

            faiss.bfKnn(res, params)

            # IndexFlat will sort the results, so we need to
            # do the same on our end
            out_d_f16 = np.sort(out_d_f16, axis=1)

            # INNER_PRODUCT is in descending order, make sure it is the same
            # order
            if faiss.is_similarity_metric(metric):
                ref_d_f16 = np.sort(ref_d_f16, axis=1)

            print('f16', np.abs(ref_d_f16 - out_d_f16).max())

            np.testing.assert_allclose(ref_d_f16, out_d_f16, atol = 4e-3)

    def test_gower_pairwise(self):
        """Test Gower distance with GPU pairwise distance computation"""
        d = 16
        k = 100
        nb = k  # all pairwise distance should be the same as nb = k
        nq = 10

        rs = np.random.RandomState(123)
        
        # Create mixed data: first half numeric [0,1], second half categorical (negative)
        xs = np.zeros((nb, d), dtype=np.float32)
        qs = np.zeros((nq, d), dtype=np.float32)
        
        # Numeric features in [0,1] range
        xs[:, :d//2] = rs.rand(nb, d//2)
        qs[:, :d//2] = rs.rand(nq, d//2)
        
        # Categorical features (negative integers)
        xs[:, d//2:] = -rs.randint(1, 4, size=(nb, d//2)).astype('float32')
        qs[:, d//2:] = -rs.randint(1, 4, size=(nq, d//2)).astype('float32')

        res = faiss.StandardGpuResources()

        # Get ground truth using IndexFlat
        index = faiss.IndexFlat(d, faiss.METRIC_GOWER)
        index.add(xs)
        ref_d, _ = index.search(qs, k)

        out_d = np.empty((nq, k), dtype=np.float32)

        # Test GPU pairwise computation
        params = faiss.GpuDistanceParams()
        params.metric = faiss.METRIC_GOWER
        params.k = -1  # all pairwise
        params.dims = d
        params.vectors = faiss.swig_ptr(xs)
        params.numVectors = nb
        params.queries = faiss.swig_ptr(qs)
        params.numQueries = nq
        params.outDistances = faiss.swig_ptr(out_d)
        params.device = 0  # Use GPU 0

        faiss.bfKnn(res, params)

        # IndexFlat will sort the results, so we need to do the same on our end
        out_d = np.sort(out_d, axis=1)

        print('Gower max diff:', np.abs(ref_d - out_d).max())
        np.testing.assert_allclose(ref_d, out_d, atol=1e-5)



def eval_codec(q, xb):
    codes = q.compute_codes(xb)
    decoded = q.decode(codes)
    return ((xb - decoded) ** 2).sum()


class TestResidualQuantizer(unittest.TestCase):

    # This test is disabled due to memory corruption in some dependency.
    # It only happens in CUDA 11.4.4 after switching from  defaults
    # to conda-forge for dependencies.
    # GpuProgressiveDimIndexFactory is partially overwritten, and ncall
    # ends up with garbage data when checking it in Python. However,
    # the C++ side prints the right values. This is likely a compiler bug.
    # This test is left in the codebase for now but skipped so that we
    # know there is a problem with it.
    @unittest.skip("Skipped due to ncall memory corruption.")
    def test_with_gpu(self):
        """ check that we get the same results with a GPU quantizer and a CPU quantizer """
        d = 32
        nt = 3000
        nb = 1000
        xt, xb, _ = get_dataset_2(d, nt, nb, 0)

        rq0 = faiss.ResidualQuantizer(d, 4, 6)
        rq0.train(xt)
        err_rq0 = eval_codec(rq0, xb)
        # codes0 = rq0.compute_codes(xb)
        rq1 = faiss.ResidualQuantizer(d, 4, 6)
        fac = faiss.GpuProgressiveDimIndexFactory(1)
        rq1.assign_index_factory = fac
        rq1.train(xt)
        self.assertGreater(fac.ncall, 0)
        ncall_train = fac.ncall
        err_rq1 = eval_codec(rq1, xb)
        # codes1 = rq1.compute_codes(xb)
        self.assertGreater(fac.ncall, ncall_train)

        print(err_rq0, err_rq1)

        self.assertTrue(0.9 * err_rq0 < err_rq1 < 1.1 * err_rq0)

        # np.testing.assert_array_equal(codes0, codes1)


class TestGpuFlags(unittest.TestCase):

    def test_gpu_flag(self):
        assert "GPU" in faiss.get_compile_options().split()
