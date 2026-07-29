# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import (
    absolute_import,
    division,
    print_function,
    unicode_literals,
)

import math
import unittest
import numpy as np
import faiss
from faiss.contrib import datasets
from faiss.contrib import ivf_tools
from faiss.contrib.evaluation import knn_intersection_measure


class TestIVFSearchPreassigned(unittest.TestCase):
    def test_ivfflat_add_preassigned(self):
        res = faiss.StandardGpuResources()
        d = 32
        nb = 5000
        nq = 100
        nlist = 64
        nprobe = 8
        k = 10

        rs = np.random.RandomState(123)
        xt = rs.rand(2000, d).astype("float32")
        xb = rs.rand(nb, d).astype("float32")
        xq = rs.rand(nq, d).astype("float32")

        # The IVF list assignment deliberately comes from a different
        # quantizer over only part of the vector.
        km = faiss.Kmeans(8, nlist, niter=10, verbose=False)
        km.train(xt[:, :8].copy())
        alt_quantizer = km.index

        cpu_quantizer = faiss.IndexFlatL2(d)
        idx_cpu = faiss.IndexIVFFlat(cpu_quantizer, d, nlist, faiss.METRIC_L2)

        config = faiss.GpuIndexIVFFlatConfig()
        config.use_cuvs = False
        idx_gpu = faiss.GpuIndexIVFFlat(res, d, nlist, faiss.METRIC_L2, config)

        fake_centroids = np.zeros((nlist, d), dtype="float32")
        cpu_quantizer.add(fake_centroids)
        idx_gpu.quantizer.add(fake_centroids)

        idx_cpu.train(xt)
        idx_gpu.train(xt)

        assign = alt_quantizer.search(xb[:, :8].copy(), 1)[1].ravel()
        ivf_tools.add_preassigned(idx_cpu, xb, assign)
        ivf_tools.add_preassigned(idx_gpu, xb, assign)

        self.assertEqual(idx_gpu.ntotal, nb)

        idx_cpu.nprobe = nprobe
        idx_gpu.nprobe = nprobe
        query_assign = alt_quantizer.search(xq[:, :8].copy(), nprobe)[1]

        cpu_d, cpu_i = ivf_tools.search_preassigned(idx_cpu, xq, k, query_assign)
        gpu_d, gpu_i = ivf_tools.search_preassigned(idx_gpu, xq, k, query_assign)

        np.testing.assert_allclose(gpu_d, cpu_d, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(gpu_i, cpu_i)

    def test_ivfpq_add_preassigned(self):
        res = faiss.StandardGpuResources()
        d = 32
        nb = 500
        nq = 20
        nlist = 16
        nprobe = 8
        k = 10

        rs = np.random.RandomState(123)
        xt = rs.rand(10000, d).astype("float32")
        xb = rs.rand(nb, d).astype("float32")
        xq = rs.rand(nq, d).astype("float32")

        idx_cpu = faiss.IndexIVFPQ(
            faiss.IndexFlatL2(d), d, nlist, 4, 8
        )
        self.assertTrue(idx_cpu.by_residual)
        idx_cpu.train(xt)

        config = faiss.GpuIndexIVFPQConfig()
        config.use_cuvs = False
        idx_gpu = faiss.GpuIndexIVFPQ(res, idx_cpu, config)

        # Force assignments away from the coarse quantizer result so this
        # exercises residual encoding against the preassigned centroid.
        assign = idx_cpu.quantizer.search(xb, 1)[1].ravel()
        assign = (assign + 1) % nlist

        ivf_tools.add_preassigned(idx_cpu, xb, assign)
        ivf_tools.add_preassigned(idx_gpu, xb, assign)

        self.assertEqual(idx_gpu.ntotal, nb)
        expected_sizes = np.bincount(assign, minlength=nlist)
        for list_no, expected_size in enumerate(expected_sizes):
            self.assertEqual(idx_gpu.getListLength(list_no), expected_size)

        idx_cpu.nprobe = nprobe
        idx_gpu.nprobe = nprobe
        query_dis, query_assign = idx_cpu.quantizer.search(xq, nprobe)

        cpu_d, cpu_i = ivf_tools.search_preassigned(
            idx_cpu, xq, k, query_assign, query_dis
        )
        gpu_d, gpu_i = ivf_tools.search_preassigned(
            idx_gpu, xq, k, query_assign, query_dis
        )

        np.testing.assert_allclose(gpu_d, cpu_d, rtol=1e-5, atol=1e-5)
        np.testing.assert_array_equal(gpu_i, cpu_i)

    def test_ivfflat_search_preassigned(self):
        res = faiss.StandardGpuResources()
        d = 50
        nb = 50000
        nq = 100
        nlist = 128
        nprobe = 10
        k = 50

        config = faiss.GpuIndexIVFFlatConfig()
        config.use_cuvs = False
        idx_gpu = faiss.GpuIndexIVFFlat(res, d, nlist, faiss.METRIC_L2, config)
        idx_gpu.nprobe = nprobe

        rs = np.random.RandomState(567)
        xb = rs.rand(nb, d).astype("float32")
        xq = rs.rand(nq, d).astype("float32")

        idx_gpu.train(xb)
        idx_gpu.add(xb)

        # Search separately using the same quantizer
        q_d, q_i = idx_gpu.quantizer.search(xq, nprobe)

        preassigned_d, preassigned_i = ivf_tools.search_preassigned(
            idx_gpu, xq, k, q_i, q_d
        )

        # Search using the standard API
        d, i = idx_gpu.search(xq, k)

        # The two results should be exactly the same
        self.assertEqual((d == preassigned_d).sum(), d.size)
        self.assertEqual((i == preassigned_i).sum(), i.size)

    def test_ivfpq_search_preassigned(self):
        res = faiss.StandardGpuResources()
        d = 64
        nb = 50000
        nq = 100
        nlist = 128
        nprobe = 5
        k = 50

        config = faiss.GpuIndexIVFPQConfig()
        config.use_cuvs = False
        idx_gpu = faiss.GpuIndexIVFPQ(
            res, d, nlist, 4, 8, faiss.METRIC_L2, config
        )
        idx_gpu.nprobe = nprobe

        rs = np.random.RandomState(567)
        xb = rs.rand(nb, d).astype("float32")
        xq = rs.rand(nq, d).astype("float32")

        idx_gpu.train(xb)
        idx_gpu.add(xb)

        # Search separately using the same quantizer
        q_d, q_i = idx_gpu.quantizer.search(xq, nprobe)

        preassigned_d, preassigned_i = ivf_tools.search_preassigned(
            idx_gpu, xq, k, q_i, q_d
        )

        # Search using the standard API
        d, i = idx_gpu.search(xq, k)

        # The two results should be exactly the same
        self.assertEqual((d == preassigned_d).sum(), d.size)
        self.assertEqual((i == preassigned_i).sum(), i.size)

    def test_ivfsq_search_preassigned(self):
        res = faiss.StandardGpuResources()
        d = 64
        nb = 50000
        nq = 100
        nlist = 128
        nprobe = 5
        k = 50

        idx_gpu = faiss.GpuIndexIVFScalarQuantizer(
            res,
            d,
            nlist,
            faiss.ScalarQuantizer.QT_6bit,
            faiss.METRIC_INNER_PRODUCT,
        )
        idx_gpu.nprobe = nprobe

        rs = np.random.RandomState(567)
        xb = rs.rand(nb, d).astype("float32")
        xq = rs.rand(nq, d).astype("float32")

        idx_gpu.train(xb)
        idx_gpu.add(xb)

        # Search separately using the same quantizer
        q_d, q_i = idx_gpu.quantizer.search(xq, nprobe)

        preassigned_d, preassigned_i = ivf_tools.search_preassigned(
            idx_gpu, xq, k, q_i, q_d
        )

        # Search using the standard API
        d, i = idx_gpu.search(xq, k)

        # The two results should be exactly the same
        self.assertEqual((d == preassigned_d).sum(), d.size)
        self.assertEqual((i == preassigned_i).sum(), i.size)


class TestIVFPluggableCoarseQuantizer(unittest.TestCase):
    def test_ivfflat_cpu_coarse(self):
        res = faiss.StandardGpuResources()
        d = 128
        nb = 5000
        nq = 100
        nlist = 10
        nprobe = 3

        q = faiss.IndexFlatL2(d)
        idx_cpu = faiss.IndexIVFFlat(q, d, nlist)

        rs = np.random.RandomState(567)
        xb = rs.rand(nb, d).astype("float32")
        xq = rs.rand(nq, d).astype("float32")

        idx_cpu.train(xb)
        idx_cpu.add(xb)

        # construct a GPU index using the same trained coarse quantizer
        # from the CPU index
        config = faiss.GpuIndexIVFFlatConfig()
        idx_gpu = faiss.GpuIndexIVFFlat(
            res, q, d, nlist, faiss.METRIC_L2, config
        )
        assert idx_gpu.is_trained
        idx_gpu.add(xb)

        k = 20

        idx_cpu.nprobe = nprobe
        idx_gpu.nprobe = nprobe

        d_g, i_g = idx_gpu.search(xq, k)
        d_c, i_c = idx_cpu.search(xq, k)
        self.assertGreaterEqual((i_g == i_c).sum(), i_g.size * 0.9)
        self.assertTrue(np.allclose(d_g, d_c, rtol=5e-5, atol=5e-5))

    def test_ivfsq_pu_coarse(self):
        res = faiss.StandardGpuResources()
        d = 128
        nb = 5000
        nq = 100
        nlist = 10
        nprobe = 3
        use_residual = True
        qtype = faiss.ScalarQuantizer.QT_8bit

        q = faiss.IndexFlatL2(d)
        idx_cpu = faiss.IndexIVFScalarQuantizer(
            q, d, nlist, qtype, faiss.METRIC_L2, use_residual
        )

        rs = np.random.RandomState(567)
        xb = rs.rand(nb, d).astype("float32")
        xq = rs.rand(nq, d).astype("float32")

        idx_cpu.train(xb)
        idx_cpu.add(xb)

        # construct a GPU index using the same trained coarse quantizer
        # from the CPU index
        idx_gpu = faiss.GpuIndexIVFScalarQuantizer(
            res, q, d, nlist, qtype, faiss.METRIC_L2, use_residual
        )
        assert not idx_gpu.is_trained
        idx_gpu.train(xb)
        idx_gpu.add(xb)

        k = 20

        idx_cpu.nprobe = nprobe
        idx_gpu.nprobe = nprobe

        d_g, i_g = idx_gpu.search(xq, k)
        d_c, i_c = idx_cpu.search(xq, k)

        self.assertGreaterEqual(knn_intersection_measure(i_c, i_g), 0.9)

        self.assertTrue(np.allclose(d_g, d_c, rtol=5e-4, atol=5e-4))

    def test_ivfpq_cpu_coarse(self):
        res = faiss.StandardGpuResources()
        d = 32
        nb = 50000
        nq = 20
        nlist_lvl_1 = 10
        nlist_lvl_2 = 1000
        nprobe_lvl_1 = 3
        nprobe_lvl_2 = 10

        rs = np.random.RandomState(567)
        coarse_centroids = rs.rand(nlist_lvl_2, d).astype("float32")

        # Construct an IVFFlat index for usage as a coarse quantizer
        idx_coarse_cpu = faiss.IndexIVFFlat(
            faiss.IndexFlatL2(d), d, nlist_lvl_1
        )
        idx_coarse_cpu.set_direct_map_type(faiss.DirectMap.Hashtable)
        idx_coarse_cpu.nprobe = nprobe_lvl_1

        idx_coarse_cpu.train(coarse_centroids)
        idx_coarse_cpu.add(coarse_centroids)
        idx_coarse_cpu.make_direct_map()

        assert idx_coarse_cpu.ntotal == nlist_lvl_2

        idx_cpu = faiss.IndexIVFPQ(idx_coarse_cpu, d, nlist_lvl_2, 4, 8)

        xb = rs.rand(nb, d).astype("float32")
        idx_cpu.train(xb)
        idx_cpu.add(xb)
        idx_cpu.nprobe = nprobe_lvl_2

        # construct a GPU index using the same trained coarse quantizer
        # from the CPU index
        config = faiss.GpuIndexIVFPQConfig()
        config.use_cuvs = False
        idx_gpu = faiss.GpuIndexIVFPQ(
            res, idx_coarse_cpu, d, nlist_lvl_2, 4, 8, faiss.METRIC_L2, config
        )
        assert not idx_gpu.is_trained

        idx_gpu.train(xb)
        idx_gpu.add(xb)
        idx_gpu.nprobe = nprobe_lvl_2

        k = 10

        # precomputed codes also utilize the coarse quantizer
        for use_precomputed in [False, True]:
            idx_gpu.setPrecomputedCodes(use_precomputed)

            xq = rs.rand(nq, d).astype("float32")
            d_g, i_g = idx_gpu.search(xq, k)
            d_c, i_c = idx_cpu.search(xq, k)

            self.assertGreaterEqual(knn_intersection_measure(i_c, i_g), 0.9)


class TestInterleavedIVFPQLayout(unittest.TestCase):
    def test_interleaved(self):
        res = faiss.StandardGpuResources()

        for bits_per_code in [4, 5, 6, 8]:
            d = 128
            nb = 10000
            nq = 20

            rs = np.random.RandomState(123)
            xb = rs.rand(nb, d).astype("float32")
            xq = rs.rand(nq, d).astype("float32")

            nlist = int(math.sqrt(nb))
            sub_q = 16
            nprobe = 16

            config = faiss.GpuIndexIVFPQConfig()
            config.interleavedLayout = True
            idx_gpu = faiss.GpuIndexIVFPQ(
                res, d, nlist, sub_q, bits_per_code, faiss.METRIC_L2, config
            )
            q = faiss.IndexFlatL2(d)
            idx_cpu = faiss.IndexIVFPQ(
                q, d, nlist, sub_q, bits_per_code, faiss.METRIC_L2
            )

            idx_gpu.train(xb)
            idx_gpu.add(xb)
            idx_gpu.copyTo(idx_cpu)

            idx_gpu.nprobe = nprobe
            idx_cpu.nprobe = nprobe

            k = 20

            # Try without precomputed codes
            d_g, i_g = idx_gpu.search(xq, k)
            d_c, i_c = idx_cpu.search(xq, k)
            self.assertGreaterEqual((i_g == i_c).sum(), i_g.size * 0.9)
            self.assertTrue(np.allclose(d_g, d_c, rtol=5e-5, atol=5e-5))

            # Try with precomputed codes (different kernel)
            idx_gpu.setPrecomputedCodes(True)
            d_g, i_g = idx_gpu.search(xq, k)
            d_c, i_c = idx_cpu.search(xq, k)
            self.assertGreaterEqual((i_g == i_c).sum(), i_g.size * 0.9)
            self.assertTrue(np.allclose(d_g, d_c, rtol=5e-5, atol=5e-5))

    def test_copy_to_cpu(self):
        res = faiss.StandardGpuResources()

        for bits_per_code in [4, 5, 6, 8]:
            d = 128
            nb = 10000
            nq = 20

            rs = np.random.RandomState(234)
            xb = rs.rand(nb, d).astype("float32")
            xq = rs.rand(nq, d).astype("float32")

            nlist = int(math.sqrt(nb))
            sub_q = 16
            bits_per_code = 8
            nprobe = 4

            config = faiss.GpuIndexIVFPQConfig()
            config.interleavedLayout = True
            idx_gpu = faiss.GpuIndexIVFPQ(
                res, d, nlist, sub_q, bits_per_code, faiss.METRIC_L2, config
            )
            q = faiss.IndexFlatL2(d)
            idx_cpu = faiss.IndexIVFPQ(
                q, d, nlist, sub_q, bits_per_code, faiss.METRIC_L2
            )

            idx_gpu.train(xb)
            idx_gpu.add(xb)

            idx_gpu.copyTo(idx_cpu)

            idx_gpu.nprobe = nprobe
            idx_cpu.nprobe = nprobe

            # Try without precomputed codes
            d_g, i_g = idx_gpu.search(xq, 10)
            d_c, i_c = idx_cpu.search(xq, 10)
            self.assertGreaterEqual((i_g == i_c).sum(), i_g.size * 0.9)
            self.assertTrue(np.allclose(d_g, d_c))

            # Try with precomputed codes (different kernel)
            idx_gpu.setPrecomputedCodes(True)
            d_g, i_g = idx_gpu.search(xq, 10)
            d_c, i_c = idx_cpu.search(xq, 10)
            self.assertGreaterEqual((i_g == i_c).sum(), i_g.size * 0.9)
            self.assertTrue(np.allclose(d_g, d_c))

    def test_copy_to_gpu(self):
        res = faiss.StandardGpuResources()

        for bits_per_code in [4, 5, 6, 8]:
            d = 128
            nb = 10000
            nq = 20

            rs = np.random.RandomState(567)
            xb = rs.rand(nb, d).astype("float32")
            xq = rs.rand(nq, d).astype("float32")

            nlist = int(math.sqrt(nb))
            sub_q = 16
            bits_per_code = 8
            nprobe = 4

            config = faiss.GpuIndexIVFPQConfig()
            config.interleavedLayout = True
            idx_gpu = faiss.GpuIndexIVFPQ(
                res, d, nlist, sub_q, bits_per_code, faiss.METRIC_L2, config
            )
            q = faiss.IndexFlatL2(d)
            idx_cpu = faiss.IndexIVFPQ(
                q, d, nlist, sub_q, bits_per_code, faiss.METRIC_L2
            )

            idx_cpu.train(xb)
            idx_cpu.add(xb)

            idx_gpu.copyFrom(idx_cpu)

            idx_gpu.nprobe = nprobe
            idx_cpu.nprobe = nprobe

            # Try without precomputed codes
            d_g, i_g = idx_gpu.search(xq, 10)
            d_c, i_c = idx_cpu.search(xq, 10)
            self.assertGreaterEqual((i_g == i_c).sum(), i_g.size * 0.9)
            self.assertTrue(np.allclose(d_g, d_c))

            # Try with precomputed codes (different kernel)
            idx_gpu.setPrecomputedCodes(True)
            d_g, i_g = idx_gpu.search(xq, 10)
            d_c, i_c = idx_cpu.search(xq, 10)
            self.assertGreaterEqual((i_g == i_c).sum(), i_g.size * 0.9)
            self.assertTrue(np.allclose(d_g, d_c))


# Make sure indices are properly stored in the IVF lists
class TestIVFIndices(unittest.TestCase):
    def test_indices_ivfflat(self):
        res = faiss.StandardGpuResources()
        d = 128
        nb = 5000
        nlist = 10

        rs = np.random.RandomState(567)
        xb = rs.rand(nb, d).astype("float32")
        xb_indices_base = np.arange(nb, dtype=np.int64)

        # Force values to not be representable in int32
        xb_indices = (xb_indices_base + 4294967296).astype("int64")

        config = faiss.GpuIndexIVFFlatConfig()
        idx = faiss.GpuIndexIVFFlat(res, d, nlist, faiss.METRIC_L2, config)
        idx.train(xb)
        idx.add_with_ids(xb, xb_indices)

        _, I = idx.search(xb[10:20], 5)
        self.assertTrue(np.array_equal(xb_indices[10:20], I[:, 0]))

        # Store values using 32-bit indices instead
        config.indicesOptions = faiss.INDICES_32_BIT
        config.use_cuvs = False
        idx = faiss.GpuIndexIVFFlat(res, d, nlist, faiss.METRIC_L2, config)
        idx.train(xb)
        idx.add_with_ids(xb, xb_indices)

        _, I = idx.search(xb[10:20], 5)
        # This will strip the high bit
        self.assertTrue(np.array_equal(xb_indices_base[10:20], I[:, 0]))

    def test_indices_ivfpq(self):
        res = faiss.StandardGpuResources()
        d = 128
        nb = 5000
        nlist = 10
        M = 4
        nbits = 8

        rs = np.random.RandomState(567)
        xb = rs.rand(nb, d).astype("float32")
        xb_indices_base = np.arange(nb, dtype=np.int64)

        # Force values to not be representable in int32
        xb_indices = (xb_indices_base + 4294967296).astype("int64")

        config = faiss.GpuIndexIVFPQConfig()
        idx = faiss.GpuIndexIVFPQ(
            res, d, nlist, M, nbits, faiss.METRIC_L2, config
        )
        idx.train(xb)
        idx.add_with_ids(xb, xb_indices)

        _, I = idx.search(xb[10:20], 5)
        self.assertTrue(np.array_equal(xb_indices[10:20], I[:, 0]))

        # Store values using 32-bit indices instead
        config.indicesOptions = faiss.INDICES_32_BIT
        # 32-bit indices are not supported with cuVS
        config.use_cuvs = False
        idx = faiss.GpuIndexIVFPQ(
            res, d, nlist, M, nbits, faiss.METRIC_L2, config
        )
        idx.train(xb)
        idx.add_with_ids(xb, xb_indices)

        _, I = idx.search(xb[10:20], 5)
        # This will strip the high bit
        self.assertTrue(np.array_equal(xb_indices_base[10:20], I[:, 0]))

    def test_indices_ivfsq(self):
        res = faiss.StandardGpuResources()
        d = 128
        nb = 5000
        nlist = 10
        qtype = faiss.ScalarQuantizer.QT_4bit

        rs = np.random.RandomState(567)
        xb = rs.rand(nb, d).astype("float32")
        xb_indices_base = np.arange(nb, dtype=np.int64)

        # Force values to not be representable in int32
        xb_indices = (xb_indices_base + 4294967296).astype("int64")

        config = faiss.GpuIndexIVFScalarQuantizerConfig()
        idx = faiss.GpuIndexIVFScalarQuantizer(
            res, d, nlist, qtype, faiss.METRIC_L2, True, config
        )
        idx.train(xb)
        idx.add_with_ids(xb, xb_indices)

        _, I = idx.search(xb[10:20], 5)
        self.assertTrue(np.array_equal(xb_indices[10:20], I[:, 0]))

        # Store values using 32-bit indices instead
        config.indicesOptions = faiss.INDICES_32_BIT
        idx = faiss.GpuIndexIVFScalarQuantizer(
            res, d, nlist, qtype, faiss.METRIC_L2, True, config
        )
        idx.train(xb)
        idx.add_with_ids(xb, xb_indices)

        _, I = idx.search(xb[10:20], 5)
        # This will strip the high bit
        self.assertTrue(np.array_equal(xb_indices_base[10:20], I[:, 0]))


class TestSQ_to_gpu(unittest.TestCase):

    def test_sq_cpu_to_gpu(self):
        res = faiss.StandardGpuResources()
        index = faiss.index_factory(32, "SQfp16")
        index.add(np.random.rand(1000, 32).astype(np.float32))
        config = faiss.GpuClonerOptions()
        config.use_cuvs = False
        gpu_index = faiss.index_cpu_to_gpu(res, 0, index, config)
        self.assertIsInstance(gpu_index, faiss.GpuIndexFlat)


class TestInvalidParams(unittest.TestCase):

    def test_indices_ivfpq(self):
        res = faiss.StandardGpuResources()
        d = 128
        nb = 5000
        nlist = 10
        M = 4
        nbits = 8

        rs = np.random.RandomState(567)
        xb = rs.rand(nb, d).astype("float32")
        xb_indices_base = np.arange(nb, dtype=np.int64)

        # Force values to not be representable in int32
        xb_indices = (xb_indices_base + 4294967296).astype("int64")

        config = faiss.GpuIndexIVFPQConfig()
        idx = faiss.GpuIndexIVFPQ(
            res, d, nlist, M, nbits, faiss.METRIC_L2, config
        )
        idx.train(xb)
        idx.add_with_ids(xb, xb_indices)

        # invalid k (should be > 0)
        k = -5
        idx.nprobe = 3
        self.assertRaises(AssertionError, idx.search, xb[10:20], k)

        # nprobe is unsigned now, so this is caught before reaching C++
        # k = 5
        # idx.nprobe = -3
        # self.assertRaises(RuntimeError, idx.search, xb[10:20], k)

        # valid params
        k = 5
        idx.nprobe = 3
        _, I = idx.search(xb[10:20], k)
        self.assertTrue(np.array_equal(xb_indices[10:20], I[:, 0]))


class TestLSQIcmEncoder(unittest.TestCase):

    @staticmethod
    def eval_codec(q, xb):
        codes = q.compute_codes(xb)
        decoded = q.decode(codes)
        return ((xb - decoded) ** 2).sum()

    def subtest_gpu_encoding(self, ngpus):
        """check that the error is in the same as cpu."""
        ds = datasets.SyntheticDataset(32, 1000, 1000, 0)

        xt = ds.get_train()
        xb = ds.get_database()

        M = 4
        nbits = 8

        lsq = faiss.LocalSearchQuantizer(ds.d, M, nbits)
        lsq.train(xt)
        err_cpu = self.eval_codec(lsq, xb)

        lsq = faiss.LocalSearchQuantizer(ds.d, M, nbits)
        lsq.train(xt)
        lsq.icm_encoder_factory = faiss.GpuIcmEncoderFactory(ngpus)
        err_gpu = self.eval_codec(lsq, xb)

        # 13804.411 vs 13814.794, 1 gpu
        print(err_gpu, err_cpu)
        self.assertLess(err_gpu, err_cpu * 1.05)

    def test_one_gpu(self):
        self.subtest_gpu_encoding(1)

    def test_multiple_gpu(self):
        ngpu = faiss.get_num_gpus()
        self.subtest_gpu_encoding(ngpu)


class TestGpuAutoTune(unittest.TestCase):

    def test_params(self):
        index = faiss.index_factory(32, "IVF65536_HNSW,PQ16")
        res = faiss.StandardGpuResources()
        options = faiss.GpuClonerOptions()
        options.allowCpuCoarseQuantizer = True
        index = faiss.index_cpu_to_gpu(res, 0, index, options)
        ps = faiss.GpuParameterSpace()
        ps.initialize(index)
        for i in range(ps.parameter_ranges.size()):
            pr = ps.parameter_ranges.at(i)
            if pr.name == "quantizer_efSearch":
                break
        else:
            self.fail("should include efSearch")
        ps.set_index_parameter(index, "quantizer_efSearch", 123)
        quantizer = faiss.downcast_index(index.quantizer)
        self.assertEqual(quantizer.hnsw.efSearch, 123)
