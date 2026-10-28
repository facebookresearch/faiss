# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Sliding window tests."""

import unittest

import numpy as np

import faiss

faiss.omp_set_num_threads(4)

d = 32
nt = 5000
nb = 1000
nq = 200

total_size = 40
window_size = 10


def make_data(n):
    rs = np.random.RandomState(0)
    return rs.rand(n, d).astype(np.float32)


def make_trained_index(index_key):
    index = faiss.index_factory(d, index_key)
    index.train(make_data(nt))
    faiss.ParameterSpace().set_index_parameter(index, "nprobe", 4)
    return index


def search_index(index, xq):
    _D, I = index.search(xq, 10)
    return I


def make_index_slices(trained_index):
    rs = np.random.RandomState(1)
    sub_indexes = []
    for _i in range(total_size):
        index = faiss.clone_index(trained_index)
        xb = rs.rand(nb, d).astype(np.float32)
        ids = rs.randint(0, 2**31, size=nb).astype(np.int64)
        index.add_with_ids(xb, ids)
        sub_indexes.append(index)
    return sub_indexes


def make_merged_index(trained_index, sub_indexes, i):
    merged_index = faiss.clone_index(trained_index)
    for j in range(i - window_size + 1, i + 1):
        if j < 0 or j >= total_size:
            continue
        sub_index = faiss.clone_index(sub_indexes[j])
        ivf0 = faiss.extract_index_ivf(merged_index)
        ivf1 = faiss.extract_index_ivf(sub_index)
        ivf0.merge_from(ivf1, 0)
        merged_index.ntotal = ivf0.ntotal
    return merged_index


class TestSlidingWindow(unittest.TestCase):
    def check_sliding_window(self, index_key):
        trained_index = make_trained_index(index_key)
        sub_indexes = make_index_slices(trained_index)

        index = faiss.clone_index(trained_index)
        window = faiss.SlidingIndexWindow(index)

        xq = make_data(nq)

        for i in range(total_size + window_size):
            window.step(
                sub_indexes[i] if i < total_size else None,
                i >= window_size,
            )

            new_res = search_index(index, xq)
            merged_index = make_merged_index(trained_index, sub_indexes, i)
            ref_res = search_index(merged_index, xq)

            np.testing.assert_array_equal(ref_res, new_res)

    def test_pca_ivf_flat(self):
        self.check_sliding_window("PCA24,IVF32,Flat")


class TestSlidingInvlists(unittest.TestCase):
    def check_sliding_invlists(self, index_key):
        trained_index = make_trained_index(index_key)
        sub_indexes = make_index_slices(trained_index)

        index = faiss.clone_index(trained_index)
        index_ivf = faiss.extract_index_ivf(index)

        xq = make_data(nq)

        for i in range(total_size + window_size):
            ils = faiss.InvertedListsPtrVector()
            for j in range(i - window_size + 1, i + 1):
                if j < 0 or j >= total_size:
                    continue
                ils.push_back(
                    faiss.extract_index_ivf(sub_indexes[j]).invlists)
            if ils.size() == 0:
                continue

            ci = faiss.ConcatenatedInvertedLists(ils.size(), ils.data())
            index_ivf.replace_invlists(ci, True)
            ci.this.disown()

            new_res = search_index(index, xq)
            merged_index = make_merged_index(trained_index, sub_indexes, i)
            ref_res = search_index(merged_index, xq)

            np.testing.assert_array_equal(ref_res, new_res)

    def test_ivf_flat(self):
        self.check_sliding_invlists("IVF32,Flat")
