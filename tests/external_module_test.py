# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import faiss

import faiss.faiss_example_external_module as external_module

import numpy as np


class TestCustomIDSelector(unittest.TestCase):

    def test_IDSelector(self):
        ids = external_module.IDSelectorModulo(3)
        self.assertFalse(ids.is_member(1))
        self.assertTrue(ids.is_member(3))


class TestArrayConversions(unittest.TestCase):

    def test_idx_array(self):
        tab = np.arange(10).astype("int64")
        new_sum = external_module.sum_of_idx(len(tab), faiss.swig_ptr(tab))
        self.assertEqual(new_sum, tab.sum())

    def test_sum_all_types(self):
        for ty in (
            "uint8",
            "uint16",
            "uint32",
            "uint64",
            "int8",
            "int16",
            "int32",
            "int64",
            "float32",
            "float64",
        ):
            with self.subTest(ty=ty):
                tab = np.arange(10).astype(ty)
                func = getattr(external_module, "sum_of_" + ty)
                new_sum = func(len(tab), faiss.swig_ptr(tab))
                self.assertEqual(new_sum, tab.sum())
