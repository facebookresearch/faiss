# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import sys
import types
import unittest
from unittest import mock

import numpy as np
from numpy._core._multiarray_umath import __cpu_features__

from faiss.loader import supported_instruction_sets


class TestDisabledCPUFeatures(unittest.TestCase):

    def test_numpy_cpu_features(self):
        features = {"AVX2": True, "AVX512F": True, "SSE2": True}
        for disabled in ["AVX2,AVX512F", "AVX2 AVX512F", " AVX2,\tAVX512F\n"]:
            with self.subTest(disabled=disabled):
                with (
                    mock.patch.dict(__cpu_features__, features, clear=True),
                    mock.patch.dict(
                        os.environ, {"FAISS_DISABLE_CPU_FEATURES": disabled}
                    ),
                    mock.patch("platform.machine", return_value="x86_64"),
                ):
                    self.assertEqual(supported_instruction_sets(), {"SSE2"})

    def test_no_disabled_features(self):
        features = {"AVX2": True, "AVX512F": True, "SSE2": True}
        with (
            mock.patch.dict(__cpu_features__, features, clear=True),
            mock.patch.dict(os.environ, {"FAISS_DISABLE_CPU_FEATURES": ""}),
            mock.patch("platform.machine", return_value="x86_64"),
        ):
            self.assertEqual(supported_instruction_sets(), set(features))

    def test_legacy_linux_cpu_features(self):
        cpuinfo = types.ModuleType("numpy.distutils.cpuinfo")
        cpuinfo.cpu = types.SimpleNamespace(
            info=[{"flags": "avx2 avx512 avx512_fp16"}]
        )
        distutils = types.ModuleType("numpy.distutils")
        distutils.cpuinfo = cpuinfo
        with (
            mock.patch.object(np, "__version__", "1.18.0"),
            mock.patch.object(np, "distutils", distutils, create=True),
            mock.patch.dict(
                sys.modules,
                {
                    "numpy.distutils": distutils,
                    "numpy.distutils.cpuinfo": cpuinfo,
                },
            ),
            mock.patch("platform.machine", return_value="x86_64"),
            mock.patch("platform.system", return_value="Linux"),
            mock.patch.dict(
                os.environ,
                {"FAISS_DISABLE_CPU_FEATURES": "AVX2, AVX512\tAVX512_SPR"},
            ),
        ):
            self.assertEqual(supported_instruction_sets(), set())


if __name__ == "__main__":
    unittest.main()
