# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for IVF index behavior."""

import contextlib
import unittest

import numpy as np

import faiss

faiss.omp_set_num_threads(4)


def encode_fp16(x):
    return np.asarray(x, dtype=np.float32).astype(np.float16)


def decode_fp16(x):
    return x.astype(np.float32)


def make_training_data(n, d):
    i = np.arange(n * d, dtype=np.int64)
    x = ((i * 29) % 131).astype(np.float32) / np.float32(17) - np.float32(3)
    return x.reshape(n, d)


def make_queries(n, d):
    i = np.arange(n * d, dtype=np.int64)
    x = ((i * 37) % 113).astype(np.float32) / np.float32(19) - np.float32(3)
    return x.reshape(n, d)


def small_clustering_params(niter):
    cp = faiss.ClusteringParameters()
    cp.niter = niter
    cp.seed = 1234
    cp.min_points_per_centroid = 1
    return cp


@contextlib.contextmanager
def scoped_blas_tiles(query_bs):
    threshold = faiss.cvar.distance_compute_blas_threshold
    saved_query_bs = faiss.cvar.distance_compute_blas_query_bs
    faiss.cvar.distance_compute_blas_threshold = 0
    faiss.cvar.distance_compute_blas_query_bs = query_bs
    try:
        yield
    finally:
        faiss.cvar.distance_compute_blas_threshold = threshold
        faiss.cvar.distance_compute_blas_query_bs = saved_query_bs


def iteration_objs(clus):
    stats = clus.iteration_stats
    return [stats.at(i).obj for i in range(stats.size())]


class TestIVFIndex(unittest.TestCase):
    def test_sorted_range_selector_rejects_store_pairs(self):
        d = 4
        nb = 100
        k = 50

        quantizer = faiss.IndexFlatL2(d)
        centroid = np.zeros((1, d), dtype=np.float32)
        quantizer.add(centroid)
        index = faiss.IndexIVFFlat(quantizer, d, 1)
        index.is_trained = True

        database = np.zeros((nb, d), dtype=np.float32)
        ids = np.zeros(nb, dtype=np.int64)
        for i in range(nb):
            ids[i] = 1000 + i
            for j in range(d):
                database[i, j] = float(i + j)
        index.add_with_ids(database, ids)

        selector = faiss.IDSelectorRange(1020, 1040, True)
        params = faiss.SearchParametersIVF()
        params.nprobe = 1
        params.sel = selector
        keys = np.zeros((1, 1), dtype=np.int64)
        coarse_distance = np.zeros((1, 1), dtype=np.float32)
        D = np.empty((1, k), dtype=np.float32)
        I = np.empty((1, k), dtype=np.int64)
        with self.assertRaises(RuntimeError):
            index.search_preassigned_c(
                1,
                faiss.swig_ptr(centroid),
                k,
                faiss.swig_ptr(keys),
                faiss.swig_ptr(coarse_distance),
                faiss.swig_ptr(D),
                faiss.swig_ptr(I),
                True,
                params,
            )

    def test_jaccard_search_returns_most_similar_vector(self):
        d = 3
        xb = np.array(
            [
                [1.0, 0.0, 1.0],
                [1.0, 1.0, 0.0],
                [0.0, 1.0, 1.0],
            ],
            dtype=np.float32,
        )

        quantizer = faiss.IndexFlatL2(d)
        quantizer.add(xb[:1])
        index = faiss.IndexIVFFlat(quantizer, d, 1, faiss.METRIC_Jaccard)
        index.add(xb)

        for parallel_mode in range(4):
            index.parallel_mode = parallel_mode
            D, I = index.search(xb[:1], 1)
            self.assertEqual(I[0][0], 0, f"parallel mode {parallel_mode}")
            self.assertAlmostEqual(
                D[0][0], 1.0, msg=f"parallel mode {parallel_mode}")

    def test_preassigned_out_of_range_key(self):
        d = 4
        nlist = 2
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, nlist)

        train_data = np.arange(nlist * d, dtype=np.float32).reshape(
            nlist, d)
        index.train(train_data)
        index.add(train_data)

        xq = np.ones((1, d), dtype=np.float32)
        bad_keys = np.array([[nlist]], dtype=np.int64)
        coarse_dis = np.zeros((1, 1), dtype=np.float32)
        with self.assertRaises(RuntimeError):
            index.search_preassigned(xq, 1, bad_keys, coarse_dis)
        with self.assertRaises(RuntimeError):
            index.range_search_preassigned(
                xq, np.finfo(np.float32).max, bad_keys, coarse_dis
            )

    def test_train_float16_matches_float32_on_rounded_input(self):
        d = 4
        n = 64
        nlist = 4

        i = np.arange(n * d, dtype=np.int64)
        x = ((i * 17) % 101).astype(np.float32) / np.float32(13)
        encoded = encode_fp16(x.reshape(n, d))
        rounded = decode_fp16(encoded)

        float_quantizer = faiss.IndexFlatL2(d)
        float_index = faiss.IndexIVFFlat(float_quantizer, d, nlist)
        float_index.cp.seed = 1234
        float_index.cp.niter = 4
        float_index.cp.min_points_per_centroid = 1
        float_index.train(rounded)

        half_quantizer = faiss.IndexFlatL2(d)
        half_index = faiss.IndexIVFFlat(half_quantizer, d, nlist)
        half_index.cp.seed = 1234
        half_index.cp.niter = 4
        half_index.cp.min_points_per_centroid = 1
        half_index.train(encoded, numeric_type=faiss.Float16)

        self.assertTrue(half_index.is_trained)
        np.testing.assert_array_equal(
            half_quantizer.reconstruct_n(0, nlist),
            float_quantizer.reconstruct_n(0, nlist),
        )

    def test_train_float16_trains_scalar_quantizer_encoder(self):
        d = 4
        n = 128
        nlist = 4

        i = np.arange(n * d, dtype=np.int64)
        x = ((i * 19) % 113).astype(np.float32) / np.float32(23)
        encoded = encode_fp16(x.reshape(n, d))

        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFScalarQuantizer(
            quantizer, d, nlist, faiss.ScalarQuantizer.QT_8bit)
        index.cp.seed = 1234
        index.cp.niter = 4
        index.cp.min_points_per_centroid = 1
        index.train(encoded, numeric_type=faiss.Float16)

        self.assertTrue(index.is_trained)
        self.assertEqual(quantizer.ntotal, nlist)
        self.assertGreater(index.sq.trained.size(), 0)

    def test_train_float16_with_flat_assigner_matches_float32(self):
        d = 10
        n = 120
        nlist = 4
        encoded = encode_fp16(make_training_data(n, d))
        rounded = decode_fp16(encoded)

        float_quantizer = faiss.IndexFlatL2(d)
        float_index = faiss.IndexIVFFlat(float_quantizer, d, nlist)
        float_index.quantizer_trains_alone = 2
        float_index.cp = small_clustering_params(4)
        float_index.train(rounded)

        half_quantizer = faiss.IndexFlatL2(d)
        half_index = faiss.IndexIVFFlat(half_quantizer, d, nlist)
        half_index.quantizer_trains_alone = 2
        half_index.cp = small_clustering_params(4)
        half_index.train(encoded, numeric_type=faiss.Float16)

        np.testing.assert_array_equal(
            half_quantizer.reconstruct_n(0, nlist),
            float_quantizer.reconstruct_n(0, nlist),
        )

    def test_train_float16_rejected_by_dedup(self):
        d = 4
        n = 16
        encoded = encode_fp16(make_training_data(n, d))
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlatDedup(quantizer, d, 2)
        with self.assertRaises(RuntimeError):
            index.train(encoded, numeric_type=faiss.Float16)

    def test_train_ex_float32_preserves_deduplication(self):
        d = 1
        n = 4
        nlist = 1
        x = np.array([[0.0], [0.0], [0.0], [10.0]], dtype=np.float32)

        train_ex_quantizer = faiss.IndexFlatL2(d)
        train_ex = faiss.IndexIVFFlatDedup(train_ex_quantizer, d, nlist)
        train_ex.cp = small_clustering_params(1)
        train_ex.train_ex(n, faiss.swig_ptr(x), faiss.Float32)

        train_ex_centroid = train_ex_quantizer.reconstruct(0)
        np.testing.assert_array_max_ulp(
            train_ex_centroid, np.float32([5.0]), maxulp=4)


class TestClustering(unittest.TestCase):
    def test_train_ex_float16_matches_float32_on_rounded_input(self):
        n = 200
        k = 5
        # 16 exercises only the SIMD kernels, 13 and 37 also the scalar tails
        for d in (13, 16, 37):
            encoded = encode_fp16(make_training_data(n, d))
            rounded = decode_fp16(encoded)
            cp = small_clustering_params(6)

            full = faiss.Clustering(d, k, cp)
            full_index = faiss.IndexFlatL2(d)
            full.train(rounded, full_index)

            half = faiss.Clustering(d, k, cp)
            half_index = faiss.IndexFlatL2(d)
            half.train_ex(
                n, faiss.swig_ptr(encoded), faiss.Float16, half_index)

            np.testing.assert_array_equal(
                faiss.vector_to_array(half.centroids),
                faiss.vector_to_array(full.centroids),
                err_msg=f"d={d}",
            )
            full_objs = iteration_objs(full)
            half_objs = iteration_objs(half)
            self.assertEqual(half_objs, full_objs, f"d={d}")

    def test_train_ex_float16_weighted_matches_float32(self):
        n = 150
        d = 20
        k = 4
        encoded = encode_fp16(make_training_data(n, d))
        rounded = decode_fp16(encoded)
        weights = np.float32(0.25) + (np.arange(n) % 7).astype(np.float32)
        cp = small_clustering_params(1)

        full = faiss.Clustering(d, k, cp)
        full_index = faiss.IndexFlatL2(d)
        full.train(rounded, full_index, weights)

        half = faiss.Clustering(d, k, cp)
        half_index = faiss.IndexFlatL2(d)
        half.train_ex(
            n,
            faiss.swig_ptr(encoded),
            faiss.Float16,
            half_index,
            faiss.swig_ptr(weights),
        )

        full_centroids = faiss.vector_to_array(full.centroids)
        half_centroids = faiss.vector_to_array(half.centroids)
        np.testing.assert_allclose(
            half_centroids, full_centroids, rtol=0, atol=1e-4)

    def test_train_ex_float16_subsampling_initializers_match_float32(self):
        n = 100
        d = 12
        k = 4
        encoded = encode_fp16(make_training_data(n, d))
        rounded = decode_fp16(encoded)

        for method in (
            faiss.ClusteringInitMethod_KMEANS_PLUS_PLUS,
            faiss.ClusteringInitMethod_AFK_MC2,
        ):
            cp = small_clustering_params(4)
            cp.max_points_per_centroid = 8
            cp.init_method = method
            cp.afkmc2_chain_length = 7

            full = faiss.Clustering(d, k, cp)
            full_index = faiss.IndexFlatL2(d)
            full.train(rounded, full_index)

            half = faiss.Clustering(d, k, cp)
            half_index = faiss.IndexFlatL2(d)
            half.train_ex(
                n, faiss.swig_ptr(encoded), faiss.Float16, half_index)

            np.testing.assert_array_equal(
                faiss.vector_to_array(half.centroids),
                faiss.vector_to_array(full.centroids),
                err_msg=f"method={method}",
            )

    def test_train_ex_float16_rejects_non_finite_values(self):
        n = 8
        d = 3
        k = 2
        max_finite = 0x7BFF  # 65504
        for bad in (0x7E00, 0x7C00, 0xFC00):
            encoded = encode_fp16(make_training_data(n, d)).view(np.uint16)
            encoded.flat[0] = max_finite
            encoded.flat[n * d - 1] = bad
            clus = faiss.Clustering(d, k, small_clustering_params(2))
            index = faiss.IndexFlatL2(d)
            with self.assertRaises(RuntimeError, msg=f"bits={bad}"):
                clus.train_ex(
                    n, faiss.swig_ptr(encoded), faiss.Float16, index)

        encoded = encode_fp16(make_training_data(n, d)).view(np.uint16)
        encoded.flat[0] = max_finite
        clus = faiss.Clustering(d, k, small_clustering_params(2))
        index = faiss.IndexFlatL2(d)
        clus.train_ex(n, faiss.swig_ptr(encoded), faiss.Float16, index)

    def test_train_ex_rejects_unsupported_numeric_type(self):
        n = 8
        d = 4
        x = np.ones(n * d, dtype=np.uint8)
        clus = faiss.Clustering(d, 2, small_clustering_params(2))
        index = faiss.IndexFlatL2(d)
        with self.assertRaises(RuntimeError):
            clus.train_ex(n, faiss.swig_ptr(x), faiss.UInt8, index)


class TestIndex(unittest.TestCase):
    def test_search_ex_float16_default_matches_float32(self):
        d = 12
        nb = 300
        nq = 40
        k = 5
        base = decode_fp16(encode_fp16(make_training_data(nb, d)))
        queries = encode_fp16(make_queries(nq, d))
        rounded = decode_fp16(queries)

        # IndexIVFFlat does not override search_ex: exercises the default
        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, 4)
        index.cp = small_clustering_params(3)
        index.train(base)
        index.add(base)
        index.nprobe = 2

        float_distances, float_labels = index.search(rounded, k)
        half_distances, half_labels = index.search(
            queries, k, numeric_type=faiss.Float16)

        np.testing.assert_array_equal(half_labels, float_labels)
        np.testing.assert_array_equal(half_distances, float_distances)
        with self.assertRaises(RuntimeError):
            index.search_ex(
                nq,
                faiss.swig_ptr(queries),
                faiss.UInt8,
                k,
                faiss.swig_ptr(half_distances),
                faiss.swig_ptr(half_labels),
            )


class TestIndexFlat(unittest.TestCase):
    def test_search_ex_float16_native_matches_float32(self):
        nb = 500
        nq = 300
        with scoped_blas_tiles(64):  # several query tiles per search
            for metric in (faiss.METRIC_L2, faiss.METRIC_INNER_PRODUCT):
                for d in (37, 64):
                    for k in (1, 5):
                        msg = f"metric={metric} d={d} k={k}"
                        base = make_training_data(nb, d)
                        queries = encode_fp16(make_queries(nq, d))
                        rounded = decode_fp16(queries)
                        index = faiss.IndexFlat(d, metric)
                        index.add(base)

                        fd, fi = index.search(rounded, k)
                        hd, hi = index.search(
                            queries, k, numeric_type=faiss.Float16)
                        np.testing.assert_array_equal(hi, fi, err_msg=msg)
                        tol = np.float32(1e-5) * np.maximum(
                            np.float32(1), np.abs(fd))
                        err = np.abs(
                            hd.astype(np.float64) - fd.astype(np.float64))
                        self.assertTrue(np.all(err <= tol), msg)

    def test_search_ex_float16_with_selector_matches_float32(self):
        d = 48
        nb = 400
        nq = 50
        k = 3
        with scoped_blas_tiles(16):
            base = make_training_data(nb, d)
            queries = encode_fp16(make_queries(nq, d))
            rounded = decode_fp16(queries)
            index = faiss.IndexFlatL2(d)
            index.add(base)

            sel = faiss.IDSelectorRange(100, 250)
            params = faiss.SearchParameters()
            params.sel = sel
            fd, fi = index.search(rounded, k, params=params)
            hd, hi = index.search(
                queries, k, params=params, numeric_type=faiss.Float16)
        np.testing.assert_array_equal(hi, fi)
        np.testing.assert_array_equal(hd, fd)

    def test_search_ex_float16_keeps_flat1d_search(self):
        nb = 100
        nq = 20
        k = 4
        base = make_training_data(nb, 1)
        queries = encode_fp16(make_queries(nq, 1))
        rounded = decode_fp16(queries)
        index = faiss.IndexFlat1D()
        index.add(base)

        fd, fi = index.search(rounded, k)
        hd, hi = index.search(queries, k, numeric_type=faiss.Float16)
        np.testing.assert_array_equal(hi, fi)
        np.testing.assert_array_equal(hd, fd)  # L1 distances from IndexFlat1D
