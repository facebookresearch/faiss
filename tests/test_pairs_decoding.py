# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np

import faiss

faiss.omp_set_num_threads(4)

d = 64
nb = 8000
nq = 200

rs = np.random.RandomState(0)


def make_data(n):
    return rs.rand(n, d).astype(np.float32)


def make_index(index_key, x):
    index = faiss.index_factory(d, index_key)
    index.train(x)
    index.add(x)
    return index


class TestPairsDecoding(unittest.TestCase):
    def check_pairs_decoding(self, index_key):
        xb = make_data(nb)
        index = make_index(index_key, xb)

        centroid_ids = np.empty(nb, dtype=np.int64)
        faiss.search_centroid(
            index, faiss.swig_ptr(xb), nb, faiss.swig_ptr(centroid_ids)
        )

        ivf = faiss.extract_index_ivf(index)
        for i in range(nb):
            list_no = int(centroid_ids[i])
            ids = faiss.rev_swig_ptr(
                ivf.invlists.get_ids(list_no),
                ivf.invlists.list_size(list_no),
            )
            self.assertIn(i, set(ids.tolist()))

        ivf.nprobe = 4

        xq = make_data(nq)
        k = 5

        _refD, refI = index.search(xq, k)

        newD = np.empty((nq, k), dtype=np.float32)
        newI = np.empty((nq, k), dtype=np.int64)
        query_centroid_ids = np.empty(nq, dtype=np.int64)
        result_centroid_ids = np.empty(nq * k, dtype=np.int64)
        faiss.search_and_return_centroids(
            index,
            nq,
            faiss.swig_ptr(xq),
            k,
            faiss.swig_ptr(newD),
            faiss.swig_ptr(newI),
            faiss.swig_ptr(query_centroid_ids),
            faiss.swig_ptr(result_centroid_ids),
        )

        np.testing.assert_array_equal(newI, refI)

        flatI = newI.reshape(-1)
        for i in range(nq * k):
            result_no = int(flatI[i])
            if result_no < 0:
                continue
            list_no = int(result_centroid_ids[i])
            ids = faiss.rev_swig_ptr(
                ivf.invlists.get_ids(list_no),
                ivf.invlists.list_size(list_no),
            )
            self.assertIn(result_no, set(ids.tolist()))

    def test_ivf_flat(self):
        self.check_pairs_decoding("IVF32,Flat")

    def test_pca_ivf_flat(self):
        self.check_pairs_decoding("PCA16,IVF32,Flat")
