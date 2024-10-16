# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import faiss
import numpy as np

from faiss.contrib import datasets

faiss.omp_set_num_threads(4)


class TestClone(unittest.TestCase):
    """
    Test clone_index for various index combinations.
    """

    def do_test_clone(self, factory, with_ids=False):
        """
        Verify that cloning works for a given index type
        """
        d = 32
        ds = datasets.SyntheticDataset(d, 1000, 2000, 10)
        index1 = faiss.index_factory(d, factory)
        index1.train(ds.get_train())
        if with_ids:
            index1.add_with_ids(ds.get_database(),
                                np.arange(ds.nb).astype("int64"))
        else:
            index1.add(ds.get_database())
        k = 5
        Dref1, Iref1 = index1.search(ds.get_queries(), k)

        index2 = faiss.clone_index(index1)
        self.assertEqual(type(index1), type(index2))
        index1 = None

        Dref2, Iref2 = index2.search(ds.get_queries(), k)
        np.testing.assert_array_equal(Dref1, Dref2)
        np.testing.assert_array_equal(Iref1, Iref2)

    def test_RFlat(self):
        self.do_test_clone("SQ4,RFlat")

    def test_Refine(self):
        self.do_test_clone("SQ4,Refine(SQ8)")

    def test_IVF(self):
        self.do_test_clone("IVF16,Flat")

    def test_PCA(self):
        self.do_test_clone("PCA8,Flat")

    def test_IDMap(self):
        self.do_test_clone("IVF16,Flat,IDMap", with_ids=True)

    def test_IDMap2(self):
        self.do_test_clone("IVF16,Flat,IDMap2", with_ids=True)

    def test_NSGPQ(self):
        self.do_test_clone("NSG32,Flat")

    def test_IVFAdditiveQuantizer(self):
        self.do_test_clone("IVF16,LSQ5x6_Nqint8")
        self.do_test_clone("IVF16,RQ5x6_Nqint8")
        self.do_test_clone("IVF16,PLSQ4x3x5_Nqint8")
        self.do_test_clone("IVF16,PRQ4x3x5_Nqint8")

    def test_IVFAdditiveQuantizerFastScan(self):
        self.do_test_clone("IVF16,LSQ3x4fs_32_Nlsq2x4")
        self.do_test_clone("IVF16,RQ3x4fs_32_Nlsq2x4")
        self.do_test_clone("IVF16,PLSQ2x3x4fs_Nlsq2x4")
        self.do_test_clone("IVF16,PRQ2x3x4fs_Nrq2x4")

    def test_AdditiveQuantizer(self):
        self.do_test_clone("LSQ5x6_Nqint8")
        self.do_test_clone("RQ5x6_Nqint8")
        self.do_test_clone("PLSQ4x3x5_Nqint8")
        self.do_test_clone("PRQ4x3x5_Nqint8")

    def test_AdditiveQuantizerFastScan(self):
        self.do_test_clone("LSQ3x4fs_32_Nlsq2x4")
        self.do_test_clone("RQ3x4fs_32_Nlsq2x4")
        self.do_test_clone("PLSQ2x3x4fs_Nlsq2x4")
        self.do_test_clone("PRQ2x3x4fs_Nrq2x4")
