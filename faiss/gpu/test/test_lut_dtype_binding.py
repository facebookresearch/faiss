# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import unittest

import faiss


@unittest.skipIf(
    "CUVS" not in faiss.get_compile_options(),
    "only if cuVS is compiled in")
class TestIVFPQSearchCagraConfigDtypes(unittest.TestCase):
    """Regression tests for the SWIG int typemap on cudaDataType_t.

    Before the typemap was added, IVFPQSearchCagraConfig.lut_dtype and
    .internal_distance_dtype were exposed as SwigPyObject pointers that
    Python users could neither construct nor assign to, and the
    CUDA_R_* enum values were not exported. These tests pin the
    behavior that they are now plain ints settable from Python.
    """

    def test_constants_exported(self):
        self.assertEqual(faiss.CUDA_R_32F, 0)
        self.assertEqual(faiss.CUDA_R_64F, 1)
        self.assertEqual(faiss.CUDA_R_16F, 2)
        self.assertEqual(faiss.CUDA_R_8I, 3)
        self.assertEqual(faiss.CUDA_R_8U, 8)

    def test_default_lut_dtype_is_fp32(self):
        c = faiss.IVFPQSearchCagraConfig()
        self.assertEqual(c.lut_dtype, faiss.CUDA_R_32F)
        self.assertEqual(c.internal_distance_dtype, faiss.CUDA_R_32F)

    def test_set_lut_dtype_via_constants(self):
        c = faiss.IVFPQSearchCagraConfig()
        for value in (
            faiss.CUDA_R_16F,
            faiss.CUDA_R_8U,
            faiss.CUDA_R_32F,
        ):
            c.lut_dtype = value
            self.assertEqual(c.lut_dtype, value)

    def test_set_lut_dtype_via_raw_int(self):
        c = faiss.IVFPQSearchCagraConfig()
        c.lut_dtype = 2  # CUDA_R_16F
        self.assertEqual(c.lut_dtype, 2)

    def test_dtype_fields_are_independent(self):
        c = faiss.IVFPQSearchCagraConfig()
        c.lut_dtype = faiss.CUDA_R_8U
        c.internal_distance_dtype = faiss.CUDA_R_16F
        self.assertEqual(c.lut_dtype, faiss.CUDA_R_8U)
        self.assertEqual(c.internal_distance_dtype, faiss.CUDA_R_16F)


if __name__ == "__main__":
    unittest.main()
