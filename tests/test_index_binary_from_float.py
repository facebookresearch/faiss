# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from __future__ import absolute_import, division, print_function

import numpy as np
import unittest
import faiss

from common_faiss_tests import for_all_simd_levels


def make_binary_dataset(d, nb, nt, nq):
    assert d % 8 == 0
    rs = np.random.RandomState(123)
    x = rs.randint(256, size=(nb + nq + nt, int(d / 8))).astype("uint8")
    return x[:nt], x[nt:-nq], x[-nq:]


@for_all_simd_levels
class TestIndexBinaryFromFloat(unittest.TestCase):
    def test_index_from_float_inner_product(self):
        d = 256
        nt = 0
        nb = 1500
        nq = 500
        (xt, xb, xq) = make_binary_dataset(d, nb, nt, nq)

        index_ref = faiss.IndexBinaryFlat(d)
        index_ref.add(xb)

        index = faiss.IndexFlat(d, faiss.METRIC_INNER_PRODUCT)
        index_bin = faiss.IndexBinaryFromFloat(index)
        index_bin.add(xb)

        D_ref, I_ref = index_ref.search(xq, 10)
        D, I = index_bin.search(xq, 10)

        np.testing.assert_array_equal(D_ref, D)

    def test_index_from_float_l1(self):
        d = 256
        nt = 0
        nb = 1500
        nq = 500
        (xt, xb, xq) = make_binary_dataset(d, nb, nt, nq)

        index_ref = faiss.IndexBinaryFlat(d)
        index_ref.add(xb)

        index = faiss.IndexFlat(d, faiss.METRIC_L1)
        index_bin = faiss.IndexBinaryFromFloat(index)
        index_bin.add(xb)

        D_ref, I_ref = index_ref.search(xq, 10)
        D, I = index_bin.search(xq, 10)

        np.testing.assert_array_equal(D_ref, D)

    def test_wrapped_quantizer(self):
        d = 256
        nt = 150
        nb = 1500
        nq = 500
        (xt, xb, xq) = make_binary_dataset(d, nb, nt, nq)

        nlist = 16
        quantizer_ref = faiss.IndexBinaryFlat(d)
        index_ref = faiss.IndexBinaryIVF(quantizer_ref, d, nlist)
        index_ref.train(xt)

        index_ref.add(xb)

        unwrapped_quantizer = faiss.IndexFlatL2(d)
        quantizer = faiss.IndexBinaryFromFloat(unwrapped_quantizer)
        index = faiss.IndexBinaryIVF(quantizer, d, nlist)

        index.train(xt)

        index.add(xb)

        D_ref, I_ref = index_ref.search(xq, 10)
        D, I = index.search(xq, 10)

        np.testing.assert_array_equal(D_ref, D)

    def test_wrapped_quantizer_IMI(self):
        d = 256
        nt = 3500
        nb = 10000
        nq = 500
        (xt, xb, xq) = make_binary_dataset(d, nb, nt, nq)

        index_ref = faiss.IndexBinaryFlat(d)

        index_ref.add(xb)

        nlist_exp = 6
        nlist = 2 ** (2 * nlist_exp)
        float_quantizer = faiss.MultiIndexQuantizer(d, 2, nlist_exp)
        wrapped_quantizer = faiss.IndexBinaryFromFloat(float_quantizer)
        wrapped_quantizer.train(xt)

        assert nlist == float_quantizer.ntotal

        index = faiss.IndexBinaryIVF(
            wrapped_quantizer, d, float_quantizer.ntotal
        )
        index.nprobe = 2048

        index.add(xb)

        D_ref, I_ref = index_ref.search(xq, 10)
        D, I = index.search(xq, 10)

        recall = sum(gti[0] in Di[:10] for gti, Di in zip(D_ref, D)) / float(
            D_ref.shape[0]
        )

        assert recall > 0.82, "recall = %g" % recall


class TestOverrideKmeansQuantizer(unittest.TestCase):

    def test_override(self):
        d = 256
        nt = 3500
        nb = 10000
        nq = 500
        (xt, xb, xq) = make_binary_dataset(d, nb, nt, nq)

        def train_and_get_centroids(override_kmeans_index):
            index = faiss.index_binary_factory(d, "BIVF10")
            index.verbose = True

            if override_kmeans_index is not None:
                index.clustering_index = override_kmeans_index

            index.train(xt)

            centroids = faiss.downcast_IndexBinary(index.quantizer).xb
            return faiss.vector_to_array(centroids).reshape(-1, d // 8)

        centroids_ref = train_and_get_centroids(None)

        # should do the exact same thing
        centroids_new = train_and_get_centroids(faiss.IndexFlatL2(d))

        assert np.all(centroids_ref == centroids_new)

        # will do less accurate assignment... Sanity check that the
        # index is indeed used by kmeans
        centroids_new = train_and_get_centroids(faiss.IndexLSH(d, 16))

        assert not np.all(centroids_ref == centroids_new)
