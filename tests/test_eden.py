# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import faiss

from common_faiss_tests import NoneSIMDLevel


EDEN_QTYPES = {
    1: faiss.ScalarQuantizer.QT_1bit_eden,
    2: faiss.ScalarQuantizer.QT_2bit_eden,
    3: faiss.ScalarQuantizer.QT_3bit_eden,
    4: faiss.ScalarQuantizer.QT_4bit_eden,
    5: faiss.ScalarQuantizer.QT_5bit_eden,
    6: faiss.ScalarQuantizer.QT_6bit_eden,
    7: faiss.ScalarQuantizer.QT_7bit_eden,
    8: faiss.ScalarQuantizer.QT_8bit_eden,
}


def lloyd_max_codebook(bits):
    quantizer = faiss.ScalarQuantizer(1, EDEN_QTYPES[bits])
    quantizer.train(np.zeros((1, 1), dtype="float32"))
    trained = faiss.vector_to_array(quantizer.trained)
    k = 1 << bits
    return trained[:k], trained[k:]


def eden_reference_reconstruct(x, bits, scale_type="unbiased"):
    residual = np.asarray(x, dtype="float32")

    centroids, boundaries = lloyd_max_codebook(bits)
    d = residual.shape[1]

    out = np.empty_like(residual)
    for i, r in enumerate(residual):
        norm_sqr = float(np.dot(r, r))
        if norm_sqr == 0:
            out[i] = 0
            continue
        z = r * np.sqrt(d / norm_sqr)
        assignments = np.searchsorted(boundaries, z, side="right")
        q = centroids[assignments]
        if scale_type == "unbiased":
            scale = norm_sqr / float(np.dot(q, r))
        else:
            scale = float(np.dot(q, r)) / float(np.dot(q, q))
        out[i] = scale * q
    return out


def eden_reference_l2_distances(x, query, bits, center):
    center = np.asarray(center, dtype="float32")
    residual = np.asarray(x, dtype="float32") - center
    query_residual = np.asarray(query, dtype="float32") - center

    centroids, boundaries = lloyd_max_codebook(bits)
    d = residual.shape[1]
    query_base = float(np.dot(query_residual, query_residual))
    distances = np.empty(residual.shape[0], dtype="float32")

    for i, r in enumerate(residual):
        norm_sqr = float(np.dot(r, r))
        if norm_sqr == 0:
            distances[i] = query_base
            continue
        z = r * np.sqrt(d / norm_sqr)
        assignments = np.searchsorted(boundaries, z, side="right")
        q = centroids[assignments]
        scale = norm_sqr / float(np.dot(q, r))
        distances[i] = (
            query_base
            + norm_sqr
            - 2.0 * scale * float(np.dot(q, query_residual))
        )
    # The unbiased scale reports an estimator of the squared distance that can
    # go negative; the C++ distance computer clamps it to zero (METRIC_L2
    # distances are never negative), so the reference must match.
    return np.maximum(distances, 0.0)


class TestEDENScalarQuantizer(unittest.TestCase):
    def test_eden_quantizer_uses_scalar_quantizer_qtypes(self):
        for bits, qtype in EDEN_QTYPES.items():
            with self.subTest(bits=bits):
                quantizer = faiss.ScalarQuantizer(64, qtype)
                quantizer.train(np.zeros((1, 64), dtype="float32"))
                self.assertEqual(quantizer.bits, bits)
                self.assertEqual(quantizer.code_size, (64 * bits + 7) // 8)

    def test_decode_matches_eden_scale_reference(self):
        rs = np.random.RandomState(123)
        xb = rs.randn(16, 64).astype("float32")

        for scale_type, scale_name in [
            (faiss.EDENScaleType_UNBIASED, "unbiased"),
            (faiss.EDENScaleType_BIASED, "biased"),
        ]:
            for bits in [1, 2, 4]:
                with self.subTest(scale_type=scale_name, bits=bits):
                    index = faiss.IndexEDEN(
                        64, faiss.METRIC_L2, bits, scale_type
                    )
                    index.train(np.zeros((1, 64), dtype="float32"))
                    codes = index.sa_encode(xb)
                    decoded = index.sa_decode(codes)
                    reference = eden_reference_reconstruct(
                        xb, bits, scale_type=scale_name
                    )
                    np.testing.assert_allclose(decoded, reference, rtol=1e-5)

    def test_distance_computer_matches_reference_l2(self):
        rs = np.random.RandomState(456)
        xt = rs.randn(200, 32).astype("float32")
        xb = rs.randn(40, 32).astype("float32")
        xq = rs.randn(1, 32).astype("float32")

        for bits in [1, 2, 4, 8]:
            with self.subTest(bits=bits):
                index = faiss.IndexEDEN(32, faiss.METRIC_L2, bits)
                index.train(xt)
                index.add(xb)

                reference = eden_reference_l2_distances(
                    xb, xq[0], bits, faiss.vector_to_array(index.center)
                )

                dc = index.get_distance_computer()
                dc.set_query(faiss.swig_ptr(xq[0]))
                actual = np.array([dc(i) for i in range(index.ntotal)])

                np.testing.assert_allclose(
                    actual, reference, rtol=1e-5, atol=1e-5
                )

    def test_factory(self):
        for description, index_class, scale_type in [
            ("EDEN4", faiss.IndexEDEN, faiss.EDENScaleType_UNBIASED),
            ("IVF16,EDEN4", faiss.IndexIVFEDEN, faiss.EDENScaleType_UNBIASED),
            ("EDEN4BIASED", faiss.IndexEDEN, faiss.EDENScaleType_BIASED),
            (
                "IVF16,EDEN4BIASED",
                faiss.IndexIVFEDEN,
                faiss.EDENScaleType_BIASED,
            ),
        ]:
            with self.subTest(description=description):
                index = faiss.index_factory(64, description)
                self.assertIsInstance(index, index_class)
                self.assertEqual(
                    index.sq.qtype, faiss.ScalarQuantizer.QT_4bit_eden
                )
                self.assertEqual(index.scale_type, scale_type)

    def test_search_matches_none_simd_level(self):
        if not faiss.SIMDConfig.is_simd_level_available(faiss.SIMDLevel_NONE):
            self.skipTest("SIMDLevel.NONE not available")

        levels = [
            faiss.SIMDLevel_AVX2,
            faiss.SIMDLevel_AVX512,
            faiss.SIMDLevel_ARM_NEON,
            faiss.SIMDLevel_RISCV_RVV,
        ]
        levels = [
            level
            for level in levels
            if faiss.SIMDConfig.is_simd_level_available(level)
        ]
        if not levels:
            self.skipTest("No non-NONE SIMD levels available")

        rs = np.random.RandomState(2468)
        previous = faiss.SIMDConfig.get_level()
        try:
            for bits, d in [(1, 256), (2, 64), (4, 256), (8, 128)]:
                xt = rs.randn(400, d).astype("float32")
                xb = rs.randn(128, d).astype("float32")
                xq = rs.randn(5, d).astype("float32")
                index = faiss.IndexEDEN(d, faiss.METRIC_L2, bits)
                index.train(xt)
                index.add(xb)

                with NoneSIMDLevel():
                    D_ref, I_ref = index.search(xq, 10)

                for level in levels:
                    with self.subTest(bits=bits, d=d, level=level):
                        faiss.SIMDConfig.set_level(level)
                        D, I = index.search(xq, 10)
                        np.testing.assert_allclose(
                            D, D_ref, rtol=1e-5, atol=1e-4
                        )
                        np.testing.assert_array_equal(I, I_ref)
        finally:
            faiss.SIMDConfig.set_level(previous)

    def test_high_dimensional_search_matches_distance_computer(self):
        rs = np.random.RandomState(2468)
        d, nt, nb, nq, k = 1024, 128, 23, 3, 5
        xt = rs.randn(nt, d).astype("float32")
        xb = rs.randn(nb, d).astype("float32")
        xq = rs.randn(nq, d).astype("float32")

        for bits in [1, 2, 4, 8]:
            with self.subTest(bits=bits):
                index = faiss.IndexEDEN(d, faiss.METRIC_L2, bits)
                index.train(xt)
                index.add(xb)
                distances, labels = index.search(xq, k)

                dc = index.get_distance_computer()
                for q in range(nq):
                    dc.set_query(faiss.swig_ptr(xq[q]))
                    reference = np.array(
                        [dc(i) for i in range(index.ntotal)], dtype="float32"
                    )
                    expected_labels = np.argsort(reference)[:k]
                    expected_distances = reference[expected_labels]

                    np.testing.assert_array_equal(labels[q], expected_labels)
                    np.testing.assert_allclose(
                        distances[q],
                        expected_distances,
                        rtol=1e-5,
                        atol=1e-4,
                    )

    def test_unbiased_l2_distances_are_non_negative(self):
        # Self-queries drive the unbiased estimator below zero; METRIC_L2
        # distances must still be non-negative.
        rs = np.random.RandomState(2468)
        d, nb = 128, 256
        xt = rs.randn(400, d).astype("float32")
        xb = rs.randn(nb, d).astype("float32")
        xq = xb.copy()

        index = faiss.IndexEDEN(d, faiss.METRIC_L2, 1)
        index.train(xt)
        index.add(xb)

        dc = index.get_distance_computer()
        for q in range(xq.shape[0]):
            dc.set_query(faiss.swig_ptr(xq[q]))
            vals = np.array(
                [dc(i) for i in range(index.ntotal)], dtype="float32"
            )
            self.assertTrue(np.all(vals >= 0.0))

    def test_serde(self):
        rs = np.random.RandomState(987)
        d, nt, nb, nq = 32, 200, 300, 10
        xt = rs.randn(nt, d).astype("float32")
        xb = rs.randn(nb, d).astype("float32")
        xq = rs.randn(nq, d).astype("float32")

        for description in [
            "EDEN4",
            "IVF16,EDEN4",
            "EDEN4BIASED",
            "IVF16,EDEN4BIASED",
        ]:
            with self.subTest(description=description):
                index = faiss.index_factory(d, description)
                if hasattr(index, "nprobe"):
                    index.nprobe = 4
                index.train(xt)
                index.add(xb)
                Dref, Iref = index.search(xq, 5)
                self.assertTrue(np.all(np.isfinite(Dref)))
                self.assertTrue(np.all(Iref >= 0))

                payload = faiss.serialize_index(index)
                index2 = faiss.deserialize_index(payload)
                if hasattr(index2, "nprobe"):
                    index2.nprobe = 4
                Dnew, Inew = index2.search(xq, 5)

                np.testing.assert_array_equal(Iref, Inew)
                np.testing.assert_allclose(Dref, Dnew, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
