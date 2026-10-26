# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Tests for the SVS metric mapping. Ported from test_svs_metric.cpp."""

import unittest

import numpy as np

import faiss

faiss.omp_set_num_threads(4)


@unittest.skipIf(
    "SVS" not in faiss.get_compile_options().split(),
    "SVS support not compiled in",
)
class TestSVSMetric(unittest.TestCase):
    def test_index_svs_flat_create_impl_unsupported_metric_throws(self):
        index = faiss.IndexSVSFlat(4, faiss.METRIC_L1)
        data = np.array(
            [[1.0, 2.0, 3.0, 4.0]], dtype=np.float32)
        with self.assertRaises(RuntimeError):
            index.add(data)
