# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np
import faiss


HALF_LLOYD_MAX_CENTROIDS = {
    1: [0.7978845608028654],
    2: [0.4527800398860679, 1.5104176087114887],
    3: [
        0.24509416307340598,
        0.7560052489539643,
        1.3439092613750225,
        2.151945669890335,
    ],
    4: [
        0.12839501671105813,
        0.38804823445328507,
        0.6567589957631145,
        0.9423402689122875,
        1.2562309480263467,
        1.6180460517130526,
        2.069016730231837,
        2.732588804065177,
    ],
}


def lloyd_max_centroids(bits):
    positive = np.asarray(HALF_LLOYD_MAX_CENTROIDS[bits], dtype="float32")
    return np.concatenate((-positive[::-1], positive))


def eden_reference_reconstruct(x, bits, center=None, scale_type="unbiased"):
    x = np.asarray(x, dtype="float32")
    if center is None:
        residual = x.copy()
        base = 0
    else:
        center = np.asarray(center, dtype="float32")
        residual = x - center
        base = center

    centroids = lloyd_max_centroids(bits)
    boundaries = (centroids[:-1] + centroids[1:]) * 0.5
    d = residual.shape[1]

    out = np.empty_like(residual)
    for i, r in enumerate(residual):
        norm_sqr = float(np.dot(r, r))
        if norm_sqr == 0:
            out[i] = base
            continue
        z = r * np.sqrt(d / norm_sqr)
        assignments = np.searchsorted(boundaries, z, side="right")
        q = centroids[assignments]
        if scale_type == "unbiased":
            scale = norm_sqr / float(np.dot(q, r))
        elif scale_type == "biased":
            scale = float(np.dot(q, r)) / float(np.dot(q, q))
        else:
            raise ValueError(f"unknown EDEN scale type: {scale_type}")
        out[i] = base + scale * q
    return out


def eden_reference_l2_distances(x, query, bits, center=None, scale_type="unbiased"):
    x = np.asarray(x, dtype="float32")
    query = np.asarray(query, dtype="float32")
    if center is None:
        residual = x.copy()
        query_residual = query
    else:
        center = np.asarray(center, dtype="float32")
        residual = x - center
        query_residual = query - center

    centroids = lloyd_max_centroids(bits)
    boundaries = (centroids[:-1] + centroids[1:]) * 0.5
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
        code_dot_residual = float(np.dot(q, r))
        code_norm_sqr = float(np.dot(q, q))
        if scale_type == "unbiased":
            scale = norm_sqr / code_dot_residual
            l2_norm_term = norm_sqr
        elif scale_type == "biased":
            scale = code_dot_residual / code_norm_sqr
            l2_norm_term = scale * scale * code_norm_sqr
        else:
            raise ValueError(f"unknown EDEN scale type: {scale_type}")
        distances[i] = (
            query_base
            + l2_norm_term
            - 2.0 * scale * float(np.dot(q, query_residual))
        )
    return distances


class TestEDENQuantizer(unittest.TestCase):
    def test_decode_matches_original_eden_scale_reference(self):
        rs = np.random.RandomState(123)
        xb = rs.randn(16, 64).astype("float32")

        for bits in [1, 2, 4]:
            with self.subTest(bits=bits):
                quantizer = faiss.EDENQuantizer(64, faiss.METRIC_L2, bits)
                codes = quantizer.compute_codes(xb)
                decoded = quantizer.decode(codes)
                reference = eden_reference_reconstruct(xb, bits)
                np.testing.assert_allclose(decoded, reference, rtol=1e-5)

    def test_decode_matches_biased_eden_scale_reference(self):
        rs = np.random.RandomState(321)
        xb = rs.randn(16, 64).astype("float32")

        for bits in [1, 2, 4]:
            with self.subTest(bits=bits):
                quantizer = faiss.EDENQuantizer(
                    64,
                    faiss.METRIC_L2,
                    bits,
                    faiss.EDENScaleType_BIASED,
                )
                codes = quantizer.compute_codes(xb)
                decoded = quantizer.decode(codes)
                reference = eden_reference_reconstruct(
                    xb, bits, scale_type="biased"
                )
                np.testing.assert_allclose(decoded, reference, rtol=1e-5)

    def test_distance_computer_matches_reference_l2(self):
        rs = np.random.RandomState(456)
        xt = rs.randn(200, 32).astype("float32")
        xb = rs.randn(40, 32).astype("float32")
        xq = rs.randn(1, 32).astype("float32")

        for bits in [1, 2, 4]:
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

    def test_one_bit_distance_matches_rabitq(self):
        rs = np.random.RandomState(1357)
        d, nt, nb, nq = 64, 128, 40, 3
        xt = rs.randn(nt, d).astype("float32")
        xb = rs.randn(nb, d).astype("float32")
        xq = rs.randn(nq, d).astype("float32")

        rabitq = faiss.IndexRaBitQ(d, faiss.METRIC_L2, 1)
        eden = faiss.IndexEDEN(d, faiss.METRIC_L2, 1)
        rabitq.qb = 0
        for index in (rabitq, eden):
            index.train(xt)
            index.add(xb)

        np.testing.assert_allclose(
            faiss.vector_to_array(rabitq.center),
            faiss.vector_to_array(eden.center),
            rtol=0,
            atol=0,
        )
        rabitq_reconstructed = np.vstack(
            [rabitq.reconstruct(i) for i in range(nb)]
        )
        eden_reconstructed = np.vstack([eden.reconstruct(i) for i in range(nb)])
        np.testing.assert_allclose(
            rabitq_reconstructed, eden_reconstructed, rtol=1e-5, atol=1e-5
        )

        rabitq_dc = rabitq.get_distance_computer()
        eden_dc = eden.get_distance_computer()
        for query in xq:
            rabitq_dc.set_query(faiss.swig_ptr(query))
            eden_dc.set_query(faiss.swig_ptr(query))
            rabitq_distances = np.array([rabitq_dc(i) for i in range(nb)])
            eden_distances = np.array([eden_dc(i) for i in range(nb)])
            np.testing.assert_allclose(
                rabitq_distances, eden_distances, rtol=1e-5, atol=1e-4
            )

    def test_factory_flat_and_ivf(self):
        rs = np.random.RandomState(789)
        d, nt, nb, nq = 64, 300, 500, 20
        xt = rs.randn(nt, d).astype("float32")
        xb = rs.randn(nb, d).astype("float32")
        xq = rs.randn(nq, d).astype("float32")

        index = faiss.index_factory(d, "EDEN4")
        self.assertIsInstance(index, faiss.IndexEDEN)
        self.assertEqual(index.eden.nb_bits, 4)
        self.assertEqual(
            index.eden.scale_type, faiss.EDENScaleType_UNBIASED
        )
        index.train(xt)
        index.add(xb)
        D, I = index.search(xq, 10)
        self.assertEqual(D.shape, (nq, 10))
        self.assertTrue(np.all(np.isfinite(D)))
        self.assertTrue(np.all(I >= 0))

        index_ivf = faiss.index_factory(d, "IVF16,EDEN4")
        self.assertIsInstance(index_ivf, faiss.IndexIVFEDEN)
        self.assertEqual(index_ivf.eden.nb_bits, 4)
        self.assertEqual(
            index_ivf.eden.scale_type, faiss.EDENScaleType_UNBIASED
        )
        index_ivf.nprobe = 4
        index_ivf.train(xt)
        index_ivf.add(xb)
        D, I = index_ivf.search(xq, 10)
        self.assertEqual(D.shape, (nq, 10))
        self.assertTrue(np.all(np.isfinite(D)))
        self.assertTrue(np.any(I >= 0))

    def test_high_dimensional_search_matches_distance_computer(self):
        rs = np.random.RandomState(2468)
        d, nt, nb, nq, k = 1024, 128, 23, 3, 5
        xt = rs.randn(nt, d).astype("float32")
        xb = rs.randn(nb, d).astype("float32")
        xq = rs.randn(nq, d).astype("float32")

        for bits in [1, 2, 4]:
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

    def test_factory_biased_eden_scale(self):
        index = faiss.index_factory(64, "EDEN4BIASED")
        self.assertIsInstance(index, faiss.IndexEDEN)
        self.assertEqual(index.eden.nb_bits, 4)
        self.assertEqual(index.eden.scale_type, faiss.EDENScaleType_BIASED)

        index_ivf = faiss.index_factory(64, "IVF16,EDEN4BIASED")
        self.assertIsInstance(index_ivf, faiss.IndexIVFEDEN)
        self.assertEqual(index_ivf.eden.nb_bits, 4)
        self.assertEqual(index_ivf.eden.scale_type, faiss.EDENScaleType_BIASED)

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

                payload = faiss.serialize_index(index)
                index2 = faiss.deserialize_index(payload)
                if hasattr(index2, "nprobe"):
                    index2.nprobe = 4
                Dnew, Inew = index2.search(xq, 5)

                np.testing.assert_array_equal(Iref, Inew)
                np.testing.assert_allclose(Dref, Dnew, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
