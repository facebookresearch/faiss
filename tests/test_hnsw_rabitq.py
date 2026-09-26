# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import faiss
import numpy as np


class TestHNSWRaBitQ(unittest.TestCase):
    def make_data(self, d=32, nt=400, nb=600, nq=20):
        rs = np.random.RandomState(123)
        xt = rs.randn(nt, d).astype("float32")
        xb = rs.randn(nb, d).astype("float32")
        xq = rs.randn(nq, d).astype("float32")
        return xt, xb, xq

    def make_index(self, nb_bits=3):
        xt, xb, xq = self.make_data()
        index = faiss.IndexHNSWRaBitQ(
            xt.shape[1], 8, nb_bits, faiss.METRIC_L2
        )
        index.hnsw.efConstruction = 40
        index.hnsw.efSearch = 64
        index.train(xt)
        index.add(xb)
        return index, xb, xq

    @staticmethod
    def recall_at_k(actual, expected):
        matches = sum(
            len(set(actual[i]).intersection(expected[i]))
            for i in range(actual.shape[0])
        )
        return matches / actual.size

    def test_staged_search_quality(self):
        index, xb, xq = self.make_index(nb_bits=3)

        faiss.cvar.rabitq_stats.reset()
        D, I = index.search(xq, 10)
        stats = faiss.cvar.rabitq_stats
        self.assertGreater(stats.n_1bit, 0)
        self.assertGreater(stats.n_refine, 0)
        self.assertLess(stats.n_refine, stats.n_1bit)

        exact = faiss.IndexFlatL2(xb.shape[1])
        exact.add(xb)
        _, Iexact = exact.search(xq, 10)
        self.assertGreaterEqual(self.recall_at_k(I, Iexact), 0.8)
        self.assertTrue(np.all(I >= 0))
        self.assertTrue(np.all(np.isfinite(D)))

    def test_symmetric_distance_uses_sign_plane(self):
        d = 35  # Exercise padding bits in the final code byte.
        rs = np.random.RandomState(456)
        xt = np.full((80, d), 2.0, dtype="float32")
        residuals = rs.randn(3, d).astype("float32") * 1e10
        residuals[2] = 0
        xb = (residuals + 2.0).astype("float32")
        residuals = xb - 2.0

        norms = np.sum(residuals * residuals, axis=1)
        alphas = np.sum(np.abs(residuals), axis=1) / d
        signs = np.where(residuals > 0, 1.0, -1.0)
        pairs = ((0, 1), (0, 2))
        expected = [
            norms[i]
            + norms[j]
            - 2 * (alphas[i] * alphas[j] * np.dot(signs[i], signs[j]))
            for i, j in pairs
        ]

        distances = []
        for nb_bits in (1, 4):
            storage = faiss.IndexRaBitQ(d, faiss.METRIC_L2, nb_bits)
            storage.train(xt)
            storage.add(xb)
            dc = storage.get_FlatCodesDistanceComputer()
            distances.append([dc.symmetric_dis(i, j) for i, j in pairs])

        for row in distances:
            np.testing.assert_allclose(row, expected, rtol=2e-5, atol=2e-5)
        np.testing.assert_array_equal(distances[0], distances[1])

    def test_clone_and_io_remain_mutable(self):
        index, xb, xq = self.make_index(nb_bits=3)
        storage = faiss.downcast_index(index.storage)
        Dref, Iref = index.search(xq, 10)

        cloned = faiss.clone_index(index)
        self.assertIsInstance(cloned, faiss.IndexHNSWRaBitQ)
        Dclone, Iclone = cloned.search(xq, 10)
        np.testing.assert_array_equal(Iclone, Iref)
        np.testing.assert_array_equal(Dclone, Dref)
        cloned.add(xb[:1])
        self.assertEqual(cloned.ntotal, len(xb) + 1)
        self.assertEqual(index.ntotal, len(xb))
        self.assertEqual(storage.ntotal, len(xb))

        loaded = faiss.deserialize_index(faiss.serialize_index(index))
        self.assertIsInstance(loaded, faiss.IndexHNSWRaBitQ)
        loaded_storage = faiss.downcast_index(loaded.storage)
        self.assertEqual(loaded_storage.rabitq.nb_bits, 3)
        Dloaded, Iloaded = loaded.search(xq, 10)
        np.testing.assert_array_equal(Iloaded, Iref)
        np.testing.assert_array_equal(Dloaded, Dref)
        loaded.add(xb[:1])
        self.assertEqual(loaded.ntotal, len(xb) + 1)

    def test_fp32_graph_batch_build_lifecycle(self):
        xt, xb, xq = self.make_data()
        native = faiss.IndexHNSWRaBitQ(xt.shape[1], 8, 3, faiss.METRIC_L2)
        fp32_graph = faiss.IndexHNSWRaBitQ(
            xt.shape[1], 8, 3, faiss.METRIC_L2
        )
        native.train(xt)
        fp32_graph.train(xt)

        native.add(xb)
        fp32_graph.add_with_fp32_graph(xb)
        self.assertTrue(fp32_graph.fp32_graph_built)

        native_storage = faiss.downcast_index(native.storage)
        fp32_storage = faiss.downcast_index(fp32_graph.storage)
        np.testing.assert_array_equal(
            faiss.vector_to_array(fp32_storage.codes),
            faiss.vector_to_array(native_storage.codes),
        )
        np.testing.assert_array_equal(
            faiss.vector_to_array(fp32_graph.hnsw.levels),
            faiss.vector_to_array(native.hnsw.levels),
        )
        self.assertEqual(fp32_graph.hnsw.entry_point, native.hnsw.entry_point)
        self.assertEqual(fp32_graph.hnsw.max_level, native.hnsw.max_level)
        np.testing.assert_array_equal(
            faiss.vector_to_array(fp32_graph.hnsw.offsets),
            faiss.vector_to_array(native.hnsw.offsets),
        )

        with self.assertRaisesRegex(RuntimeError, "cannot append"):
            fp32_graph.add(xb[:1])
        with self.assertRaisesRegex(RuntimeError, "requires an empty"):
            fp32_graph.add_with_fp32_graph(xb[:1])

        cloned = faiss.clone_index(fp32_graph)
        self.assertTrue(cloned.fp32_graph_built)
        with self.assertRaisesRegex(RuntimeError, "cannot append"):
            cloned.add(xb[:1])

        loaded = faiss.deserialize_index(faiss.serialize_index(fp32_graph))
        self.assertIsInstance(loaded, faiss.IndexHNSWRaBitQ)
        self.assertTrue(loaded.fp32_graph_built)
        loaded_storage = faiss.downcast_index(loaded.storage)
        np.testing.assert_array_equal(
            faiss.vector_to_array(loaded_storage.codes),
            faiss.vector_to_array(fp32_storage.codes),
        )
        with self.assertRaisesRegex(RuntimeError, "cannot append"):
            loaded.add(xb[:1])

        metadata = faiss.deserialize_index(
            faiss.serialize_index(fp32_graph, faiss.IO_FLAG_SKIP_STORAGE)
        )
        self.assertTrue(metadata.fp32_graph_built)
        self.assertIsNone(metadata.storage)
        faiss.serialize_index(metadata, faiss.IO_FLAG_SKIP_STORAGE)

        loaded.reset()
        self.assertFalse(loaded.fp32_graph_built)
        loaded.add(xb)
        self.assertEqual(loaded.ntotal, len(xb))
        D, I = loaded.search(xq, 10)
        self.assertTrue(np.all(I >= 0))
        self.assertTrue(np.all(np.isfinite(D)))

    def test_fp32_graph_matches_deterministic_flat_topology(self):
        xt, xb, _ = self.make_data(nb=160)
        fp32_graph = faiss.IndexHNSWRaBitQ(
            xt.shape[1], 8, 3, faiss.METRIC_L2
        )
        flat_graph = faiss.IndexHNSWFlat(xt.shape[1], 8, faiss.METRIC_L2)
        fp32_graph.train(xt)

        fp32_graph.add_with_fp32_graph(xb)
        flat_graph.add(xb)

        self.assertEqual(fp32_graph.hnsw.entry_point, flat_graph.hnsw.entry_point)
        self.assertEqual(fp32_graph.hnsw.max_level, flat_graph.hnsw.max_level)
        for field in ("levels", "offsets", "neighbors"):
            np.testing.assert_array_equal(
                faiss.vector_to_array(getattr(fp32_graph.hnsw, field)),
                faiss.vector_to_array(getattr(flat_graph.hnsw, field)),
            )

    def test_expanded_full_code_scalar_semantics(self):
        d = 65
        rs = np.random.RandomState(987)
        xt = rs.randn(80, d).astype("float32")
        xb = rs.randn(17, d).astype("float32")
        query = rs.randn(d).astype("float32")
        arbitrary_ids = (16, 0, 7, 2)

        for bits in (2, 4, 7, 8):
            storage = faiss.IndexRaBitQ(d, faiss.METRIC_L2, bits)
            storage.train(xt)
            storage.add(xb)
            center_before = faiss.vector_to_array(storage.center).copy()
            packed = faiss.vector_to_array(storage.codes).reshape(
                len(xb), storage.code_size
            )

            storage.qb = 0
            reference = storage.get_FlatCodesDistanceComputer()
            reference.set_query(faiss.swig_ptr(query))
            reference_scores = np.array(
                [reference(i) for i in arbitrary_ids], dtype="float32"
            )

            storage.set_full_code_mode(faiss.RABITQ_FULL_CODE_EXPANDED)
            np.testing.assert_array_equal(
                faiss.vector_to_array(storage.center), center_before
            )
            expanded = faiss.vector_to_array(storage.expanded_codes).reshape(
                len(xb), d + 8
            )
            ex_bits = bits - 1
            ex_offset = (d + 7) // 8 + 12
            ex_bytes = (d * ex_bits + 7) // 8
            for row in range(len(xb)):
                expected = np.empty(d, dtype="int8")
                for j in range(d):
                    sign = (int(packed[row, j // 8]) >> (j % 8)) & 1
                    low = 0
                    for bit in range(ex_bits):
                        pos = j * ex_bits + bit
                        low |= (
                            (int(packed[row, ex_offset + pos // 8]) >> (pos % 8))
                            & 1
                        ) << bit
                    expected[j] = (sign << ex_bits) + low - (1 << ex_bits)
                np.testing.assert_array_equal(
                    expanded[row, :d].view("int8"), expected
                )
                np.testing.assert_array_equal(
                    expanded[row, d:],
                    packed[row, ex_offset + ex_bytes : ex_offset + ex_bytes + 8],
                )

            control = storage.get_FlatCodesDistanceComputer()
            control.set_query(faiss.swig_ptr(query))
            control_scores = np.array(
                [control(i) for i in arbitrary_ids], dtype="float32"
            )
            np.testing.assert_allclose(
                control_scores, reference_scores, rtol=1e-5, atol=1e-5
            )

            storage.set_full_code_mode(faiss.RABITQ_FULL_CODE_INT8)
            integer = storage.get_FlatCodesDistanceComputer()
            integer.set_query(faiss.swig_ptr(query))
            residual = query - center_before
            scale = np.max(np.abs(residual)) / 127.0
            if scale == 0:
                scale = 1.0
            quantized = np.clip(np.rint(residual / scale), -127, 127).astype(
                "int8"
            )
            for doc_id in arbitrary_ids:
                factors = np.frombuffer(
                    expanded[doc_id, d:].tobytes(), dtype="float32", count=2
                )
                dot = np.dot(
                    quantized.astype("int64"),
                    expanded[doc_id, :d].view("int8").astype("int64"),
                )
                expected = max(
                    0.0,
                    np.dot(residual, residual)
                    + factors[0]
                    + factors[1]
                    * (scale * dot + 0.5 * np.sum(residual, dtype="float32")),
                )
                np.testing.assert_allclose(
                    integer(doc_id), expected, rtol=1e-6, atol=1e-6
                )

            cached_scores = np.array(
                [integer(doc_id) for doc_id in arbitrary_ids], dtype="float32"
            )
            storage.set_full_code_mode(faiss.RABITQ_FULL_CODE_INT8_PACKED)
            self.assertEqual(
                len(faiss.vector_to_array(storage.expanded_codes)), 0
            )
            packed_integer = storage.get_FlatCodesDistanceComputer()
            packed_integer.set_query(faiss.swig_ptr(query))
            packed_scores = np.array(
                [packed_integer(doc_id) for doc_id in arbitrary_ids],
                dtype="float32",
            )
            np.testing.assert_array_equal(packed_scores, cached_scores)

    def test_packed_integer_adc_matches_cached_hnsw(self):
        # 2-bit and 4-bit exercise fused packed ARM kernels. 7-bit remains a
        # useful fallback control for the generic scratch decoder.
        for bits in (2, 4, 7):
            index, _, xq = self.make_index(nb_bits=bits)
            storage = faiss.downcast_index(index.storage)

            index.set_full_code_mode(faiss.RABITQ_FULL_CODE_INT8)
            cached_distances, cached_labels = index.search(xq, 10)
            self.assertGreater(
                len(faiss.vector_to_array(storage.expanded_codes)), 0
            )

            index.set_full_code_mode(faiss.RABITQ_FULL_CODE_INT8_PACKED)
            self.assertEqual(
                len(faiss.vector_to_array(storage.expanded_codes)), 0
            )
            packed_distances, packed_labels = index.search(xq, 10)

            np.testing.assert_array_equal(packed_labels, cached_labels)
            np.testing.assert_array_equal(packed_distances, cached_distances)

    def test_split4_layout_matches_packed_and_one_bit(self):
        d = 65
        rs = np.random.RandomState(8642)
        xt = rs.randn(80, d).astype("float32")
        xb = rs.randn(19, d).astype("float32")
        query = rs.randn(d).astype("float32")

        storage = faiss.IndexRaBitQ(d, faiss.METRIC_L2, 4)
        storage.train(xt)
        storage.add(xb)
        signs = faiss.IndexRaBitQ(d, faiss.METRIC_L2, 1)
        storage.derive_1bit_prefix(signs)

        storage.set_full_code_mode(faiss.RABITQ_FULL_CODE_INT8_PACKED)
        packed = storage.get_FlatCodesDistanceComputer()
        packed.set_query(faiss.swig_ptr(query))
        packed_scores = np.array([packed(i) for i in range(len(xb))])

        sign_dc = signs.get_FlatCodesDistanceComputer()
        sign_dc.set_query(faiss.swig_ptr(query))
        sign_scores = np.array([sign_dc(i) for i in range(len(xb))])

        storage.prepare_split4_layout()
        self.assertEqual(len(faiss.vector_to_array(storage.codes)), 0)
        self.assertEqual(
            len(faiss.vector_to_array(storage.split4_sign_codes)),
            len(xb) * ((d + 7) // 8 + 8),
        )
        self.assertEqual(
            len(faiss.vector_to_array(storage.split4_tail_codes)),
            len(xb) * (3 * ((d + 7) // 8) + 8),
        )

        storage.set_full_code_mode(faiss.RABITQ_SPLIT4_ADC)
        split = storage.get_FlatCodesDistanceComputer()
        split.set_query(faiss.swig_ptr(query))
        np.testing.assert_array_equal(
            [split(i) for i in range(len(xb))], packed_scores
        )

        storage.set_full_code_mode(faiss.RABITQ_SPLIT4_NAVIGATION)
        split_sign = storage.get_FlatCodesDistanceComputer()
        split_sign.set_query(faiss.swig_ptr(query))
        np.testing.assert_array_equal(
            [split_sign(i) for i in range(len(xb))], sign_scores
        )

    def test_nested_progressive_prefix_matches_native_rq2(self):
        d = 65
        rs = np.random.RandomState(2468)
        xt = rs.randn(80, d).astype("float32")
        xb = rs.randn(17, d).astype("float32")
        query = rs.randn(d).astype("float32")

        native = faiss.IndexRaBitQ(d, faiss.METRIC_L2, 2)
        native.train(xt)
        native.add(xb)
        native.set_full_code_mode(faiss.RABITQ_FULL_CODE_INT8_PACKED)

        native_full = faiss.IndexRaBitQ(d, faiss.METRIC_L2, 6)
        native_full.train(xt)
        native_full.add(xb)
        native_full.set_full_code_mode(faiss.RABITQ_FULL_CODE_INT8_PACKED)

        progressive = faiss.IndexRaBitQ(d, faiss.METRIC_L2, 6)
        progressive.train(xt)
        progressive.prepare_progressive_layout()
        progressive.add(xb)
        self.assertEqual(
            progressive.code_size,
            (2 * d + 7) // 8 + 12 + (5 * d + 7) // 8,
        )

        native_dc = native.get_FlatCodesDistanceComputer()
        nested_dc = progressive.get_FlatCodesDistanceComputer()
        native_dc.set_query(faiss.swig_ptr(query))
        nested_dc.set_query(faiss.swig_ptr(query))
        np.testing.assert_allclose(
            [nested_dc(i) for i in range(len(xb))],
            [native_dc(i) for i in range(len(xb))],
            rtol=2e-6,
            atol=2e-6,
        )

        labels = np.arange(len(xb), dtype="int64").reshape(1, -1)
        progressive_scores = np.empty(labels.shape, dtype="float32")
        progressive.compute_distance_subset(
            1,
            faiss.swig_ptr(query.reshape(1, -1)),
            len(xb),
            faiss.swig_ptr(progressive_scores),
            faiss.swig_ptr(labels),
        )
        native_full_dc = native_full.get_FlatCodesDistanceComputer()
        native_full_dc.set_query(faiss.swig_ptr(query))
        expected_full = np.array(
            [native_full_dc(i) for i in range(len(xb))], dtype="float32"
        )
        np.testing.assert_allclose(
            progressive_scores[0], expected_full, rtol=2e-6, atol=2e-6
        )

    def test_progressive_7bit_matches_native_full_scores(self):
        # d=65 exercises sixteen-coordinate SIMD blocks and the scalar tail.
        d = 65
        rs = np.random.RandomState(8642)
        xt = rs.randn(80, d).astype("float32")
        xb = rs.randn(19, d).astype("float32")
        query = rs.randn(3, d).astype("float32")

        native = faiss.IndexRaBitQ(d, faiss.METRIC_L2, 7)
        native.train(xt)
        native.add(xb)
        native.set_full_code_mode(faiss.RABITQ_FULL_CODE_INT8_PACKED)

        progressive = faiss.IndexRaBitQ(d, faiss.METRIC_L2, 7)
        progressive.train(xt)
        progressive.prepare_progressive_layout()
        progressive.add(xb)
        self.assertEqual(
            progressive.code_size,
            (2 * d + 7) // 8 + 12 + (6 * d + 7) // 8,
        )

        labels = np.tile(np.arange(len(xb), dtype="int64"), (len(query), 1))
        actual = np.empty(labels.shape, dtype="float32")
        expected = np.empty(labels.shape, dtype="float32")
        progressive.compute_distance_subset(
            len(query),
            faiss.swig_ptr(query),
            len(xb),
            faiss.swig_ptr(actual),
            faiss.swig_ptr(labels),
        )
        native.compute_distance_subset(
            len(query),
            faiss.swig_ptr(query),
            len(xb),
            faiss.swig_ptr(expected),
            faiss.swig_ptr(labels),
        )
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)

        loaded = faiss.deserialize_index(faiss.serialize_index(progressive))
        self.assertEqual(
            loaded.full_code_mode, faiss.RABITQ_FULL_CODE_PROGRESSIVE
        )
        self.assertEqual(loaded.code_size, progressive.code_size)
        loaded_scores = np.empty(labels.shape, dtype="float32")
        loaded.compute_distance_subset(
            len(query),
            faiss.swig_ptr(query),
            len(xb),
            faiss.swig_ptr(loaded_scores),
            faiss.swig_ptr(labels),
        )
        np.testing.assert_array_equal(loaded_scores, actual)

    def test_nested_lut7_scalar_oracle_and_serialization(self):
        # d=65 covers one scalar coordinate after four full NEON blocks.
        d = 65
        rs = np.random.RandomState(97531)
        xt = rs.randn(96, d).astype("float32")
        xb = rs.randn(19, d).astype("float32")
        xq = rs.randn(3, d).astype("float32")

        index = faiss.IndexRaBitQ(d, faiss.METRIC_L2, 7)
        index.train(xt)
        self.assertEqual(len(faiss.vector_to_array(index.nested_lut7)), 64)
        index.prepare_nested_lut7_layout()
        index.add(xb)
        prefix_bytes = (2 * d + 7) // 8
        low4_bytes = (4 * d + 7) // 8
        high1_bytes = (d + 7) // 8
        self.assertEqual(
            index.code_size, prefix_bytes + 12 + low4_bytes + high1_bytes
        )

        native_rq2 = faiss.IndexRaBitQ(d, faiss.METRIC_L2, 2)
        native_rq2.train(xt)
        native_rq2.add(xb)
        native_rq2.set_full_code_mode(faiss.RABITQ_FULL_CODE_INT8_PACKED)
        nested_navigation = index.get_FlatCodesDistanceComputer()
        native_navigation = native_rq2.get_FlatCodesDistanceComputer()
        nested_navigation.set_query(faiss.swig_ptr(xq[0]))
        native_navigation.set_query(faiss.swig_ptr(xq[0]))
        np.testing.assert_allclose(
            [nested_navigation(i) for i in range(len(xb))],
            [native_navigation(i) for i in range(len(xb))],
            rtol=2e-6,
            atol=2e-6,
        )

        codes = faiss.vector_to_array(index.codes).reshape(
            len(xb), index.code_size
        )
        lut = faiss.vector_to_array(index.nested_lut7)
        center = faiss.vector_to_array(index.center)
        labels = np.tile(np.arange(len(xb), dtype="int64"), (len(xq), 1))
        actual = np.empty(labels.shape, dtype="float32")
        index.compute_distance_subset(
            len(xq),
            faiss.swig_ptr(xq),
            len(xb),
            faiss.swig_ptr(actual),
            faiss.swig_ptr(labels),
        )

        expected = np.empty_like(actual)
        for qi, query in enumerate(xq):
            residual = query - center
            query_norm = np.dot(residual, residual)
            half_sum = 0.5 * residual.sum()
            scale = np.max(np.abs(residual)) / 127.0
            if scale == 0:
                scale = 1.0
            quantized = np.clip(np.rint(residual / scale), -127, 127).astype(
                "int8"
            )
            for row, code in enumerate(codes):
                levels = np.empty(d, dtype="int8")
                for j in range(d):
                    prefix = (code[j // 4] >> (2 * (j % 4))) & 3
                    positive = prefix >> 1
                    coarse = (prefix & 1) ^ (0 if positive else 1)
                    local_low = (
                        code[prefix_bytes + 12 + j // 2]
                        >> (4 * (j % 2))
                    ) & 15
                    local_high = (
                        code[prefix_bytes + 12 + low4_bytes + j // 8]
                        >> (j % 8)
                    ) & 1
                    magnitude = int(lut[coarse * 32 + local_low + 16 * local_high])
                    levels[j] = magnitude if positive else -1 - magnitude
                factors = np.frombuffer(
                    code[prefix_bytes : prefix_bytes + 12].tobytes(),
                    dtype="float32",
                )
                dot = np.dot(
                    quantized.astype("int64"), levels.astype("int64")
                )
                expected[qi, row] = max(
                    0.0,
                    query_norm
                    + factors[0]
                    + factors[2] * (scale * dot + half_sum),
                )
        np.testing.assert_allclose(actual, expected, rtol=2e-6, atol=2e-6)

        index.nested_lut7_full_navigation = True
        full_navigation = index.get_FlatCodesDistanceComputer()
        full_navigation.set_query(faiss.swig_ptr(xq[0]))
        np.testing.assert_array_equal(
            [full_navigation(i) for i in range(len(xb))], actual[0]
        )
        index.nested_lut7_full_navigation = False

        loaded = faiss.deserialize_index(faiss.serialize_index(index))
        self.assertEqual(
            loaded.full_code_mode, faiss.RABITQ_FULL_CODE_NESTED_LUT7
        )
        np.testing.assert_array_equal(
            faiss.vector_to_array(loaded.nested_lut7), lut
        )
        loaded_scores = np.empty_like(actual)
        loaded.compute_distance_subset(
            len(xq),
            faiss.swig_ptr(xq),
            len(xb),
            faiss.swig_ptr(loaded_scores),
            faiss.swig_ptr(labels),
        )
        np.testing.assert_array_equal(loaded_scores, actual)

    def test_nested_lut4_nibble_matches_staged_full_and_serializes(self):
        # Odd d exercises the scalar tail after full 16-coordinate blocks.
        d = 65
        rs = np.random.RandomState(24680)
        xt = rs.randn(96, d).astype("float32")
        xb = rs.randn(19, d).astype("float32")
        xq = rs.randn(3, d).astype("float32")

        staged = faiss.IndexRaBitQ(d, faiss.METRIC_L2, 4)
        staged.train(xt)
        self.assertEqual(len(faiss.vector_to_array(staged.nested_lut4)), 8)
        staged.prepare_nested_lut4_layout()
        staged.add(xb)

        nibble = faiss.IndexRaBitQ(d, faiss.METRIC_L2, 4)
        nibble.train(xt)
        nibble.prepare_nibble_lut4_layout()
        nibble.add(xb)

        prefix_bytes = (2 * d + 7) // 8
        nibble_bytes = (4 * d + 7) // 8
        self.assertEqual(staged.code_size, 2 * prefix_bytes + 12)
        self.assertEqual(nibble.code_size, nibble_bytes + 8)
        np.testing.assert_array_equal(
            faiss.vector_to_array(staged.nested_lut4),
            faiss.vector_to_array(nibble.nested_lut4),
        )

        labels = np.tile(np.arange(len(xb), dtype="int64"), (len(xq), 1))
        staged_scores = np.empty(labels.shape, dtype="float32")
        nibble_scores = np.empty_like(staged_scores)
        staged.compute_distance_subset(
            len(xq),
            faiss.swig_ptr(xq),
            len(xb),
            faiss.swig_ptr(staged_scores),
            faiss.swig_ptr(labels),
        )
        nibble.compute_distance_subset(
            len(xq),
            faiss.swig_ptr(xq),
            len(xb),
            faiss.swig_ptr(nibble_scores),
            faiss.swig_ptr(labels),
        )
        np.testing.assert_array_equal(nibble_scores, staged_scores)

        codes = faiss.vector_to_array(nibble.codes).reshape(
            len(xb), nibble.code_size
        )
        lut = faiss.vector_to_array(nibble.nested_lut4)
        center = faiss.vector_to_array(nibble.center)
        oracle = np.empty_like(nibble_scores)
        for qi, query in enumerate(xq):
            residual = query - center
            query_norm = np.dot(residual, residual)
            half_sum = 0.5 * residual.sum()
            scale = np.max(np.abs(residual)) / 127.0
            if scale == 0:
                scale = 1.0
            quantized = np.clip(np.rint(residual / scale), -127, 127).astype(
                "int8"
            )
            for row, code in enumerate(codes):
                levels = np.empty(d, dtype="int8")
                for j in range(d):
                    symbol = (code[j // 2] >> (4 * (j % 2))) & 15
                    magnitude = int(lut[symbol & 7])
                    levels[j] = magnitude if symbol & 8 else -1 - magnitude
                factors = np.frombuffer(
                    code[nibble_bytes : nibble_bytes + 8].tobytes(),
                    dtype="float32",
                )
                dot = np.dot(
                    quantized.astype("int64"), levels.astype("int64")
                )
                oracle[qi, row] = max(
                    0.0,
                    query_norm
                    + factors[0]
                    + factors[1] * (scale * dot + half_sum),
                )
        np.testing.assert_allclose(nibble_scores, oracle, rtol=2e-6, atol=2e-6)

        for index, expected_mode in (
            (staged, faiss.RABITQ_FULL_CODE_NESTED_LUT4),
            (nibble, faiss.RABITQ_FULL_CODE_NIBBLE_LUT4),
        ):
            loaded = faiss.deserialize_index(faiss.serialize_index(index))
            self.assertEqual(loaded.full_code_mode, expected_mode)
            loaded_scores = np.empty_like(staged_scores)
            loaded.compute_distance_subset(
                len(xq),
                faiss.swig_ptr(xq),
                len(xb),
                faiss.swig_ptr(loaded_scores),
                faiss.swig_ptr(labels),
            )
            np.testing.assert_array_equal(loaded_scores, staged_scores)

    def test_nested_lut_simd_dispatch_equivalence(self):
        """Every selectable SIMD level must preserve Nested-LUT scores.

        On an SPR dynamic-dispatch build this exercises the direct VNNI path
        at AVX512_SPR and the portable fallback at every lower x86 level.
        """
        d = 65
        rs = np.random.RandomState(13579)
        xt = rs.randn(96, d).astype("float32")
        # Two eight-way batches plus a seven-way tail exercise every fixed
        # power-of-two AVX-512 kernel used by the compact dispatcher.
        xb = rs.randn(23, d).astype("float32")
        xq = rs.randn(3, d).astype("float32")
        labels = np.tile(np.arange(len(xb), dtype="int64"), (len(xq), 1))

        indexes = []
        for nb_bits, prepare in (
            (4, "prepare_nested_lut4_layout"),
            (4, "prepare_nibble_lut4_layout"),
            (7, "prepare_nested_lut7_layout"),
        ):
            index = faiss.IndexRaBitQ(d, faiss.METRIC_L2, nb_bits)
            index.train(xt)
            getattr(index, prepare)()
            index.add(xb)
            indexes.append(index)

        levels = [
            faiss.SIMDLevel_NONE,
            faiss.SIMDLevel_AVX2,
            faiss.SIMDLevel_AVX512,
            faiss.SIMDLevel_AVX512_VPOPCNT,
            faiss.SIMDLevel_AVX512_SPR,
            faiss.SIMDLevel_ARM_NEON,
            faiss.SIMDLevel_ARM_SVE,
            faiss.SIMDLevel_RISCV_RVV,
        ]
        levels = [
            level
            for level in levels
            if faiss.SIMDConfig.is_simd_level_available(level)
        ]
        original_level = faiss.SIMDConfig.get_level()
        scores_by_level = []
        try:
            for level in levels:
                faiss.SIMDConfig.set_level(level)
                self.assertEqual(
                    faiss.SIMDConfig.get_dispatched_level(), level
                )
                level_scores = []
                for index in indexes:
                    scores = np.empty(labels.shape, dtype="float32")
                    index.compute_distance_subset(
                        len(xq),
                        faiss.swig_ptr(xq),
                        len(xb),
                        faiss.swig_ptr(scores),
                        faiss.swig_ptr(labels),
                    )
                    level_scores.append(scores)
                scores_by_level.append(level_scores)
        finally:
            faiss.SIMDConfig.set_level(original_level)

        for level_scores in scores_by_level[1:]:
            for actual, expected in zip(level_scores, scores_by_level[0]):
                np.testing.assert_array_equal(actual, expected)

    def test_nested_adaptive_simd_dispatch_equivalence(self):
        d = 65
        rs = np.random.RandomState(97531)
        xt = rs.randn(128, d).astype("float32")
        xb = rs.randn(256, d).astype("float32")
        xq = rs.randn(8, d).astype("float32")

        index = faiss.IndexHNSWRaBitQ(d, 16, 7, faiss.METRIC_L2)
        index.train(xt)
        storage = faiss.downcast_index(index.storage)
        storage.prepare_nested_lut7_layout()
        index.set_full_code_mode(faiss.RABITQ_FULL_CODE_NESTED_LUT7)
        index.add_with_fp32_graph(xb)
        index.prepare_nested_adaptive_navigation()
        self.assertEqual(storage.nested_adaptive_sigma, 2.0)
        index.hnsw.efSearch = 64

        original_level = faiss.SIMDConfig.get_level()
        comparison_levels = [
            level
            for level in (
                faiss.SIMDLevel_NONE,
                faiss.SIMDLevel_AVX2,
                faiss.SIMDLevel_AVX512,
                faiss.SIMDLevel_AVX512_VPOPCNT,
                faiss.SIMDLevel_AVX512_SPR,
                faiss.SIMDLevel_ARM_NEON,
                faiss.SIMDLevel_ARM_SVE,
                faiss.SIMDLevel_RISCV_RVV,
            )
            if level != original_level
            and faiss.SIMDConfig.is_simd_level_available(level)
        ]
        try:
            expected_distances, expected_labels = index.search(xq, 10)
            for level in comparison_levels:
                faiss.SIMDConfig.set_level(level)
                actual_distances, actual_labels = index.search(xq, 10)
                np.testing.assert_array_equal(actual_labels, expected_labels)
                np.testing.assert_array_equal(
                    actual_distances, expected_distances
                )
        finally:
            faiss.SIMDConfig.set_level(original_level)

    def test_nested_adaptive_mutation_resets_dispatch(self):
        d = 65
        rs = np.random.RandomState(24680)
        xt = rs.randn(128, d).astype("float32")
        xb = rs.randn(256, d).astype("float32")
        xq = rs.randn(8, d).astype("float32")

        index = faiss.IndexHNSWRaBitQ(d, 16, 7, faiss.METRIC_L2)
        index.train(xt)
        storage = faiss.downcast_index(index.storage)
        storage.prepare_nested_lut7_layout()
        index.set_full_code_mode(faiss.RABITQ_FULL_CODE_NESTED_LUT7)
        index.add_with_fp32_graph(xb)
        index.prepare_nested_adaptive_navigation()
        self.assertEqual(
            index.hnsw.search_method, faiss.HNSW.SM_RABITQ_ADAPTIVE
        )

        perm = np.arange(len(xb) - 1, -1, -1, dtype=np.int64)
        index.permute_entries(perm)
        self.assertFalse(storage.nested_adaptive_navigation)
        self.assertEqual(index.hnsw.search_method, faiss.HNSW.SM_DEFAULT)
        distances, labels = index.search(xq, 10)
        self.assertTrue(np.all(np.isfinite(distances)))
        self.assertTrue(np.all(labels >= 0))

        index.prepare_nested_adaptive_navigation()
        self.assertEqual(
            index.hnsw.search_method, faiss.HNSW.SM_RABITQ_ADAPTIVE
        )
        index.reset()
        self.assertFalse(storage.nested_adaptive_navigation)
        self.assertEqual(index.hnsw.search_method, faiss.HNSW.SM_DEFAULT)
        self.assertFalse(index.fp32_graph_built)

    def test_nested_lut4_hnsw_serialization(self):
        d = 65
        rs = np.random.RandomState(86420)
        xt = rs.randn(128, d).astype("float32")
        xb = rs.randn(200, d).astype("float32")
        xq = rs.randn(5, d).astype("float32")

        index = faiss.IndexHNSWRaBitQ(d, 16, 4, faiss.METRIC_L2)
        index.train(xt)
        storage = faiss.downcast_index(index.storage)
        storage.prepare_nested_lut4_layout()
        index.set_full_code_mode(faiss.RABITQ_FULL_CODE_NESTED_LUT4)
        index.add_with_fp32_graph(xb)
        index.hnsw.efSearch = 64

        expected_distances, expected_labels = index.search(xq, 10)
        self.assertTrue(np.all(np.isfinite(expected_distances)))
        self.assertTrue(np.all(expected_labels >= 0))

        loaded = faiss.deserialize_index(faiss.serialize_index(index))
        loaded_storage = faiss.downcast_index(loaded.storage)
        self.assertEqual(
            loaded_storage.full_code_mode,
            faiss.RABITQ_FULL_CODE_NESTED_LUT4,
        )
        self.assertEqual(loaded_storage.code_size, storage.code_size)
        self.assertEqual(loaded.hnsw.search_method, faiss.HNSW.SM_DEFAULT)
        actual_distances, actual_labels = loaded.search(xq, 10)
        np.testing.assert_array_equal(actual_labels, expected_labels)
        np.testing.assert_array_equal(actual_distances, expected_distances)

    def test_expanded_full_code_hnsw_lifecycle(self):
        index, xb, xq = self.make_index(nb_bits=7)
        storage = faiss.downcast_index(index.storage)
        packed_codes = faiss.vector_to_array(storage.codes).copy()

        index.set_full_code_mode(faiss.RABITQ_FULL_CODE_INT8)
        self.assertEqual(index.hnsw.search_method, faiss.HNSW.SM_DEFAULT)
        self.assertEqual(
            len(faiss.vector_to_array(storage.expanded_codes)),
            len(xb) * (xb.shape[1] + 8),
        )
        D, I = index.search(xq, 10)
        self.assertTrue(np.all(np.isfinite(D)))
        self.assertTrue(np.all(I >= 0))

        index.add(xb[:3])
        self.assertEqual(index.hnsw.search_method, faiss.HNSW.SM_DEFAULT)
        self.assertEqual(storage.full_code_mode, faiss.RABITQ_FULL_CODE_INT8)
        self.assertEqual(
            len(faiss.vector_to_array(storage.expanded_codes)),
            (len(xb) + 3) * (xb.shape[1] + 8),
        )

        cloned = faiss.clone_index(index)
        cloned_storage = faiss.downcast_index(cloned.storage)
        self.assertEqual(
            cloned_storage.full_code_mode, faiss.RABITQ_FULL_CODE_INT8
        )
        self.assertEqual(cloned.hnsw.search_method, faiss.HNSW.SM_DEFAULT)

        loaded = faiss.deserialize_index(faiss.serialize_index(index))
        loaded_storage = faiss.downcast_index(loaded.storage)
        self.assertEqual(
            loaded_storage.full_code_mode, faiss.RABITQ_FULL_CODE_PACKED
        )
        self.assertEqual(
            len(faiss.vector_to_array(loaded_storage.expanded_codes)), 0
        )
        self.assertEqual(loaded.hnsw.search_method, faiss.HNSW.SM_RABITQ)
        np.testing.assert_array_equal(
            faiss.vector_to_array(loaded_storage.codes)[: len(packed_codes)],
            packed_codes,
        )

        index.set_full_code_mode(faiss.RABITQ_FULL_CODE_PACKED)
        self.assertEqual(index.hnsw.search_method, faiss.HNSW.SM_RABITQ)
        self.assertEqual(len(faiss.vector_to_array(storage.expanded_codes)), 0)

    def test_io_and_reset(self):
        index, xb, xq = self.make_index(nb_bits=4)
        cloned = faiss.clone_index(index)
        cloned.add(xb[:1])
        self.assertEqual(cloned.ntotal, len(xb) + 1)

        loaded = faiss.deserialize_index(faiss.serialize_index(index))
        loaded.add(xb[:1])
        self.assertEqual(loaded.ntotal, len(xb) + 1)

        loaded.reset()
        loaded.add(xb)
        self.assertEqual(loaded.ntotal, len(xb))
        D, I = loaded.search(xq, 10)
        self.assertTrue(np.all(I >= 0))
        self.assertTrue(np.all(np.isfinite(D)))

    def test_one_bit_fallback(self):
        index, _, xq = self.make_index(nb_bits=1)
        self.assertEqual(index.hnsw.search_method, 0)
        faiss.cvar.rabitq_stats.reset()
        D, I = index.search(xq, 10)
        self.assertTrue(np.all(I >= 0))
        self.assertTrue(np.all(np.isfinite(D)))
        self.assertEqual(faiss.cvar.rabitq_stats.n_1bit, 0)
        self.assertEqual(faiss.cvar.rabitq_stats.n_refine, 0)

        loaded = faiss.deserialize_index(faiss.serialize_index(index))
        loaded_storage = faiss.downcast_index(loaded.storage)
        self.assertEqual(loaded_storage.rabitq.nb_bits, 1)
        self.assertEqual(loaded.hnsw.search_method, 0)
        Dloaded, Iloaded = loaded.search(xq, 10)
        np.testing.assert_array_equal(Dloaded, D)
        np.testing.assert_array_equal(Iloaded, I)

    def test_staged_search_requires_bounded_queue(self):
        index, _, xq = self.make_index(nb_bits=3)
        index.hnsw.search_bounded_queue = False
        with self.assertRaises(RuntimeError):
            index.search(xq, 10)

    def test_permute_then_add(self):
        index, xb, xq = self.make_index(nb_bits=3)
        index.set_full_code_mode(faiss.RABITQ_FULL_CODE_INT8)
        perm = np.arange(len(xb) - 1, -1, -1, dtype=np.int64)
        index.permute_entries(perm)
        index.add(xb[:2])
        self.assertEqual(index.ntotal, len(xb) + 2)
        self.assertEqual(index.hnsw.search_method, faiss.HNSW.SM_DEFAULT)
        D, I = index.search(xq, 10)
        self.assertTrue(np.all(I >= 0))
        self.assertTrue(np.all(np.isfinite(D)))

    def test_factory(self):
        default_index = faiss.index_factory(32, "HNSW8,RaBitQ")
        self.assertIsInstance(default_index, faiss.IndexHNSWRaBitQ)
        self.assertFalse(default_index.is_trained)
        default_storage = faiss.downcast_index(default_index.storage)
        self.assertIsInstance(default_storage, faiss.IndexRaBitQ)
        self.assertFalse(default_storage.is_trained)
        self.assertEqual(default_storage.rabitq.nb_bits, 1)
        self.assertEqual(default_index.hnsw.search_method, 0)
        with self.assertRaises(RuntimeError):
            default_index.add(np.zeros((1, 32), dtype="float32"))

        four_bit = faiss.index_factory(32, "HNSW8,RaBitQ4")
        four_bit_storage = faiss.downcast_index(four_bit.storage)
        self.assertEqual(four_bit_storage.rabitq.nb_bits, 4)
        self.assertNotEqual(four_bit.hnsw.search_method, 0)

    def test_skip_storage_can_be_reserialized(self):
        index, _, _ = self.make_index(nb_bits=3)
        data = faiss.serialize_index(index, faiss.IO_FLAG_SKIP_STORAGE)
        metadata_only = faiss.deserialize_index(data)
        self.assertIsNone(metadata_only.storage)
        self.assertNotEqual(metadata_only.hnsw.search_method, 0)

        data_again = faiss.serialize_index(
            metadata_only, faiss.IO_FLAG_SKIP_STORAGE
        )
        metadata_again = faiss.deserialize_index(data_again)
        self.assertIsNone(metadata_again.storage)
        self.assertEqual(
            metadata_again.hnsw.search_method,
            metadata_only.hnsw.search_method,
        )

    def test_unsupported_metric_throws(self):
        for metric in (faiss.METRIC_INNER_PRODUCT, faiss.METRIC_L1):
            with self.assertRaises(RuntimeError):
                faiss.IndexHNSWRaBitQ(32, 8, 1, metric)


if __name__ == "__main__":
    unittest.main()
