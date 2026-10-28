# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import tempfile
import unittest

import numpy as np

import faiss

faiss.omp_set_num_threads(4)

# The tests rewrite a file while it is memory-mapped. Windows does not permit
# that.
MMAP_IFC_SUPPORTED = sys.platform.startswith(("linux", "freebsd", "darwin"))


def make_data(n, d, seed):
    rs = np.random.RandomState(seed)
    return rs.rand(n, d).astype(np.float32)


def make_binary_data(n, d, seed):
    rs = np.random.RandomState(seed)
    return rs.randint(0, 256, size=(n, d)).astype(np.uint8)


class TestMmap(unittest.TestCase):
    def test_mmap_flatcodes(self):
        if not MMAP_IFC_SUPPORTED:
            self.skipTest("Skipping test on unsupported platform.")
        nt = 1000
        nq = 10
        d = 32
        k = 25

        xt1 = make_data(nt, d, 123)
        xt2 = make_data(nt, d, 456)
        xq = make_data(nq, d, 789)

        index1 = faiss.IndexFlatL2(d)
        index1.add(xt1)
        ref_dis_1, ref_ids_1 = index1.search(xq, k)

        index2 = faiss.IndexFlatL2(d)
        index2.add(xt2)
        ref_dis_2, ref_ids_2 = index2.search(xq, k)

        self.assertFalse(np.array_equal(ref_dis_1, ref_dis_2))
        self.assertFalse(np.array_equal(ref_ids_1, ref_ids_2))

        with tempfile.TemporaryDirectory() as tmpdir:
            f1 = os.path.join(tmpdir, "index1.faiss")
            f2 = os.path.join(tmpdir, "index2.faiss")
            fmm = os.path.join(tmpdir, "indexmm.faiss")
            faiss.write_index(index1, f1)
            faiss.write_index(index2, f2)
            with open(f1, "rb") as f:
                buf1 = f.read()
            with open(f2, "rb") as f:
                buf2 = f.read()
            with open(fmm, "wb") as f:
                f.write(buf1)

            index1mm = faiss.read_index(fmm, faiss.IO_FLAG_MMAP_IFC)

            cand_dis_1, cand_ids_1 = index1mm.search(xq, k)
            np.testing.assert_array_equal(ref_ids_1, cand_ids_1)
            np.testing.assert_array_equal(ref_dis_1, cand_dis_1)

            with open(fmm, "wb") as f:
                f.write(buf2)
            cand_dis_2, cand_ids_2 = index1mm.search(xq, k)
            np.testing.assert_array_equal(ref_ids_2, cand_ids_2)
            np.testing.assert_array_equal(ref_dis_2, cand_dis_2)

    def test_mmap_binary_flatcodes(self):
        if not MMAP_IFC_SUPPORTED:
            self.skipTest("Skipping test on unsupported platform.")
        nt = 1000
        nq = 10
        d = 64
        d8 = (d + 7) // 8
        k = 25

        xt1 = make_binary_data(nt, d8, 123)
        xt2 = make_binary_data(nt, d8, 456)
        xq = make_binary_data(nq, d8, 789)

        index1 = faiss.IndexBinaryFlat(d)
        index1.add(xt1)
        ref_dis_1, ref_ids_1 = index1.search(xq, k)

        index2 = faiss.IndexBinaryFlat(d)
        index2.add(xt2)
        ref_dis_2, ref_ids_2 = index2.search(xq, k)

        self.assertFalse(np.array_equal(ref_dis_1, ref_dis_2))
        self.assertFalse(np.array_equal(ref_ids_1, ref_ids_2))

        with tempfile.TemporaryDirectory() as tmpdir:
            f1 = os.path.join(tmpdir, "index1.faiss")
            f2 = os.path.join(tmpdir, "index2.faiss")
            fmm = os.path.join(tmpdir, "indexmm.faiss")
            faiss.write_index_binary(index1, f1)
            faiss.write_index_binary(index2, f2)
            with open(f1, "rb") as f:
                buf1 = f.read()
            with open(f2, "rb") as f:
                buf2 = f.read()
            with open(fmm, "wb") as f:
                f.write(buf1)

            index1mm = faiss.read_index_binary(fmm, faiss.IO_FLAG_MMAP_IFC)

            cand_dis_1, cand_ids_1 = index1mm.search(xq, k)
            np.testing.assert_array_equal(ref_ids_1, cand_ids_1)
            np.testing.assert_array_equal(ref_dis_1, cand_dis_1)

            with open(fmm, "wb") as f:
                f.write(buf2)
            cand_dis_2, cand_ids_2 = index1mm.search(xq, k)
            np.testing.assert_array_equal(ref_ids_2, cand_ids_2)
            np.testing.assert_array_equal(ref_dis_2, cand_dis_2)
