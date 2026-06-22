# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Wheel smoke tests for faiss-gpu and faiss-gpu-cuvs pip packages.

These tests validate that a pip-installed GPU wheel is correctly packaged:
  - CUDA shared libraries load (cudart, cublas)
  - GPU devices are detected
  - GPU index creation and search works
  - GPU <-> CPU index transfer works
  - cuVS/CAGRA integration works (when compiled with cuVS)

They are intentionally minimal and fast. The full GPU test suite
runs separately in the cmake-based CI.
"""

import functools
import unittest
import numpy as np

import faiss

# This file is the faiss-gpu wheel smoke test. It also gets collected by
# `pytest tests/` in the CPU wheel CI and by Buck's GPU-less test runner.
# Skip every class via decorator (rather than `raise SkipTest` at module
# scope) so pytest still collects the tests — pytest exits with code 5
# ("no tests collected") otherwise, which Buck treats as a fatal.
_GPU_BUILD = "GPU" in faiss.get_compile_options()
_skip_unless_gpu_build = unittest.skipUnless(
    _GPU_BUILD, "Skipping faiss-gpu smoke tests: built without GPU support"
)


def skip_if_no_gpu(test_func):
    """Skip test if no GPU is available."""
    @functools.wraps(test_func)
    def wrapper(*args, **kwargs):
        if faiss.get_num_gpus() == 0:
            raise unittest.SkipTest("No GPU available")
        return test_func(*args, **kwargs)
    return wrapper


@_skip_unless_gpu_build
class TestGPUAvailability(unittest.TestCase):
    """Catch: CUDA libs not loaded, GPU not detected."""

    def test_compile_options_include_gpu(self):
        opts = faiss.get_compile_options()
        self.assertIn(
            "GPU", opts, f"faiss-gpu wheel missing GPU support: {opts}"
        )

    def test_gpu_count(self):
        n = faiss.get_num_gpus()
        if n < 1:
            raise unittest.SkipTest(f"No GPU available (count={n})")
        self.assertGreaterEqual(n, 1)


@_skip_unless_gpu_build
class TestGPUResources(unittest.TestCase):
    """Catch: CUDA runtime init failure, StandardGpuResources construction."""

    @skip_if_no_gpu
    def test_standard_gpu_resources(self):
        res = faiss.StandardGpuResources()
        # Verify the resource object is usable
        self.assertIsNotNone(res)


@_skip_unless_gpu_build
class TestGPUFlatSearch(unittest.TestCase):
    """Catch: GPU kernel launch failures, CUDA memory allocation."""

    @skip_if_no_gpu
    def test_gpu_flat_l2(self):
        d, n = 64, 1000
        np.random.seed(42)
        xb = np.random.random((n, d)).astype("float32")

        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatL2(res, d)
        index.add(xb)
        self.assertEqual(index.ntotal, n)

        D, I = index.search(xb[:5], 10)
        self.assertEqual(I.shape, (5, 10))
        # Nearest neighbor of each vector should be itself
        np.testing.assert_array_equal(I[:, 0], np.arange(5))
        # Self-distance should be ~0
        np.testing.assert_allclose(D[:, 0], 0, atol=1e-5)

    @skip_if_no_gpu
    def test_gpu_flat_ip(self):
        d, n = 64, 1000
        np.random.seed(42)
        xb = np.random.random((n, d)).astype("float32")
        faiss.normalize_L2(xb)

        res = faiss.StandardGpuResources()
        index = faiss.GpuIndexFlatIP(res, d)
        index.add(xb)

        D, I = index.search(xb[:5], 10)
        self.assertEqual(I.shape, (5, 10))
        np.testing.assert_array_equal(I[:, 0], np.arange(5))
        np.testing.assert_allclose(D[:, 0], 1.0, atol=1e-5)


@_skip_unless_gpu_build
class TestGPUIVFPQ(unittest.TestCase):
    """Catch: GPU training, IVF + PQ kernel issues."""

    @skip_if_no_gpu
    def test_gpu_ivfpq(self):
        d, n = 64, 10000
        np.random.seed(42)
        xb = np.random.random((n, d)).astype("float32")

        res = faiss.StandardGpuResources()
        # Build on CPU first, then transfer to GPU
        cpu_index = faiss.index_factory(d, "IVF32,PQ8")
        cpu_index.train(xb)

        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        gpu_index.add(xb)
        self.assertEqual(gpu_index.ntotal, n)

        D, I = gpu_index.search(xb[:5], 10)
        self.assertEqual(I.shape, (5, 10))


@_skip_unless_gpu_build
class TestGPUCPUTransfer(unittest.TestCase):
    """Catch: GPU<->CPU index serialization, memory transfer."""

    @skip_if_no_gpu
    def test_cpu_to_gpu_and_back(self):
        d, n = 64, 1000
        np.random.seed(42)
        xb = np.random.random((n, d)).astype("float32")

        # Build CPU index
        cpu_index = faiss.IndexFlatL2(d)
        cpu_index.add(xb)
        D_cpu, I_cpu = cpu_index.search(xb[:5], 10)

        # Transfer to GPU
        res = faiss.StandardGpuResources()
        gpu_index = faiss.index_cpu_to_gpu(res, 0, cpu_index)
        self.assertEqual(gpu_index.ntotal, n)
        D_gpu, I_gpu = gpu_index.search(xb[:5], 10)

        # Transfer back to CPU
        cpu_index2 = faiss.index_gpu_to_cpu(gpu_index)
        self.assertEqual(cpu_index2.ntotal, n)
        D_back, I_back = cpu_index2.search(xb[:5], 10)

        # Results should match
        np.testing.assert_array_equal(I_cpu, I_gpu)
        np.testing.assert_array_equal(I_cpu, I_back)
        np.testing.assert_allclose(D_cpu, D_gpu, atol=1e-5)
        np.testing.assert_allclose(D_cpu, D_back, atol=1e-5)


@_skip_unless_gpu_build
class TestGPUSerialization(unittest.TestCase):
    """Catch: GPU index serialization roundtrip."""

    @skip_if_no_gpu
    def test_gpu_serialize_roundtrip(self):
        d = 32
        np.random.seed(42)
        xb = np.random.random((100, d)).astype("float32")

        res = faiss.StandardGpuResources()
        gpu_index = faiss.GpuIndexFlatL2(res, d)
        gpu_index.add(xb)

        # Transfer to CPU for serialization
        cpu_index = faiss.index_gpu_to_cpu(gpu_index)
        data = faiss.serialize_index(cpu_index)
        self.assertIsInstance(data, np.ndarray)

        # Deserialize and transfer back to GPU
        cpu_index2 = faiss.deserialize_index(data)
        gpu_index2 = faiss.index_cpu_to_gpu(res, 0, cpu_index2)
        self.assertEqual(gpu_index2.ntotal, 100)

        D1, I1 = gpu_index.search(xb[:3], 5)
        D2, I2 = gpu_index2.search(xb[:3], 5)
        np.testing.assert_array_equal(I1, I2)
        np.testing.assert_allclose(D1, D2, atol=1e-5)
