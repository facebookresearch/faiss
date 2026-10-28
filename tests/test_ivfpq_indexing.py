# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import numpy as np

import faiss

faiss.omp_set_num_threads(4)


class TestIVFPQIndexing(unittest.TestCase):
    def test_on_the_fly_distance_to_code_matches_precomputed_table(self):
        d = 4
        nt = 256
        coarse_quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFPQ(coarse_quantizer, d, 2, 2, 4)

        rng = np.random.RandomState(1234)
        training = rng.uniform(-1.0, 1.0, size=(nt, d)).astype(np.float32)
        index.train(training)

        query = np.array([0.2, -0.4, 0.6, -0.8], dtype=np.float32)
        value = np.array([-0.1, 0.3, 0.7, -0.5], dtype=np.float32)
        coarse_D, coarse_I = index.quantizer.search(query.reshape(1, d), 1)
        list_no = int(coarse_I[0, 0])
        coarse_distance = float(coarse_D[0, 0])

        code = np.zeros(index.code_size, dtype=np.uint8)
        index.encode(list_no, faiss.swig_ptr(value), faiss.swig_ptr(code))
        table_scanner = index.get_InvertedListScanner(False, None, None)
        on_the_fly_scanner = index.get_InvertedListScanner(
            False, None, faiss.IndexIVFPQ.ScannerMode_OnTheFly
        )

        for scanner in table_scanner, on_the_fly_scanner:
            scanner.set_query(faiss.swig_ptr(query))
            scanner.set_list(list_no, coarse_distance)

        self.assertAlmostEqual(
            on_the_fly_scanner.distance_to_code(faiss.swig_ptr(code)),
            table_scanner.distance_to_code(faiss.swig_ptr(code)),
            delta=1e-4,
        )
