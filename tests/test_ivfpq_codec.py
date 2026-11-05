# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import unittest

import numpy as np

import faiss

faiss.omp_set_num_threads(4)

d = 64
nb = 8000


def eval_codec_error(ncentroids, m, database):
    coarse_quantizer = faiss.IndexFlatL2(d)
    index = faiss.IndexIVFPQ(coarse_quantizer, d, ncentroids, m, 8)
    index.pq.cp.niter = 10  # speed up train
    index.train(database)

    keys = np.empty(nb, dtype=np.int64)
    codes = np.empty((nb, m), dtype=np.uint8)
    index.encode_multiple(
        nb,
        faiss.swig_ptr(keys),
        faiss.swig_ptr(database),
        faiss.swig_ptr(codes),
        True,
    )

    v2 = np.empty((nb, d), dtype=np.float32)
    index.decode_multiple(
        nb, faiss.swig_ptr(keys), faiss.swig_ptr(codes), faiss.swig_ptr(v2)
    )

    return float(np.sum((database - v2) ** 2))


def runs_on_sandcastle():
    # see discussion here https://fburl.com/qc5kpdo2
    if os.environ.get("SANDCASTLE") == "1":
        return True
    return os.environ.get("TW_JOB_USER") == "sandcastle"


class TestIVFPQCodec(unittest.TestCase):
    def test_codec(self):
        rs = np.random.RandomState(0)
        database = rs.rand(nb, d).astype(np.float32)

        # limit number of threads when running on heavily parallelized test
        # environment
        if runs_on_sandcastle():
            faiss.omp_set_num_threads(2)

        err0 = eval_codec_error(16, 8, database)

        # should be more accurate as there are more coarse centroids
        err1 = eval_codec_error(128, 8, database)
        self.assertGreater(err0, err1)

        # should be more accurate as there are more PQ codes
        err2 = eval_codec_error(16, 16, database)
        self.assertGreater(err0, err2)
