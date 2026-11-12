# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Inverted list transfer tests."""

import unittest

import numpy as np

import faiss

faiss.omp_set_num_threads(4)

d = 64
nb = 1000
nq = 100
nt = 500
k = 10


def get_data(n, seed):
    return np.random.RandomState(seed).randn(n, d).astype(np.float32)


class TestTransferInvlists(unittest.TestCase):
    def check_index_type(self, index_key):
        nslice = 3

        trained = faiss.index_factory(d, index_key)
        trained.train(get_data(nt, 123))

        xq = get_data(nq, 818)

        src = faiss.clone_index(trained)
        src.add(get_data(nb, 245))
        faiss.ParameterSpace().set_index_parameter(src, "nprobe", 4)

        Dref, Iref = src.search(xq, k)

        # destination index: filled with garbage, replaced by source index
        dst = faiss.clone_index(trained)
        dst.add(get_data(nb + 10, 366))
        faiss.ParameterSpace().set_index_parameter(dst, "nprobe", 4)

        nlist = faiss.extract_index_ivf(src).nlist

        for sl in range(nslice):
            Dnew, Inew = dst.search(xq, k)
            self.assertFalse(np.array_equal(Iref, Inew))

            i0 = sl * nlist // nslice
            i1 = (sl + 1) * nlist // nslice

            il = faiss.get_invlist_range(src, i0, i1)
            wr = faiss.VectorIOWriter()
            faiss.write_InvertedLists(il, wr)

            # transfer data here from source machine to dest machine

            reader = faiss.VectorIOReader()
            reader.data = wr.data

            il2 = faiss.read_InvertedLists(reader)

            # swap inverted lists. Block searches here!
            faiss.set_invlist_range(dst, i0, i1, il2)

        self.assertEqual(dst.ntotal, src.ntotal)

        Dnew, Inew = dst.search(xq, k)
        np.testing.assert_array_equal(Iref, Inew)
        np.testing.assert_array_equal(Dref, Dnew)

    def test_ivf_flat_preproc(self):
        self.check_index_type("PCAR32,IVF40,Flat")
