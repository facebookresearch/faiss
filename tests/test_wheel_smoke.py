# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Wheel smoke tests for faiss-cpu pip package.

These tests validate that a pip-installed wheel is correctly packaged:
  - Shared libraries load (SWIG, BLAS, OpenMP)
  - Core search functionality works
  - Key index types are registered
  - Serialization roundtrips
  - Python reference counting is safe
  - Contrib submodules are included

They are intentionally minimal and fast (~15s total). The full test suite
runs separately in the cmake-based CI.
"""

import platform
import unittest
import numpy as np

import faiss


class TestImportAndMetadata(unittest.TestCase):
    """Catch: missing shared libs, broken SWIG, wrong Python ABI."""

    def test_version_attribute(self):
        assert hasattr(faiss, "__version__")
        parts = faiss.__version__.split(".")
        assert len(parts) == 3
        assert all(p.isdigit() for p in parts)

    def test_compile_options(self):
        opts = faiss.get_compile_options()
        assert isinstance(opts, str)


class TestOpenMP(unittest.TestCase):
    """Catch: OpenMP not linked."""

    def test_openmp_threads(self):
        n = faiss.omp_get_max_threads()
        assert n >= 1
        faiss.omp_set_num_threads(2)
        assert faiss.omp_get_max_threads() == 2
        # Restore
        faiss.omp_set_num_threads(n)


class TestFlatSearch(unittest.TestCase):
    """Catch: BLAS not linked, SIMD codepath issues."""

    def test_flat_l2(self):
        d, n = 64, 1000
        np.random.seed(42)
        xb = np.random.random((n, d)).astype("float32")
        index = faiss.IndexFlatL2(d)
        index.add(xb)
        D, I = index.search(xb[:5], 10)
        assert I.shape == (5, 10)
        # Nearest neighbor of each vector should be itself
        np.testing.assert_array_equal(I[:, 0], np.arange(5))
        # Self-distance should be ~0
        np.testing.assert_allclose(D[:, 0], 0, atol=1e-5)

    def test_flat_ip(self):
        d, n = 64, 1000
        np.random.seed(42)
        xb = np.random.random((n, d)).astype("float32")
        faiss.normalize_L2(xb)
        index = faiss.IndexFlatIP(d)
        index.add(xb)
        D, I = index.search(xb[:5], 10)
        assert I.shape == (5, 10)
        np.testing.assert_array_equal(I[:, 0], np.arange(5))
        np.testing.assert_allclose(D[:, 0], 1.0, atol=1e-5)


class TestIndexFactory(unittest.TestCase):
    """Catch: missing index implementations, broken factory string parsing."""

    def test_ivf_pq(self):
        d, n = 64, 10000
        np.random.seed(42)
        xb = np.random.random((n, d)).astype("float32")
        index = faiss.index_factory(d, "IVF32,PQ8")
        assert not index.is_trained
        index.train(xb)
        assert index.is_trained
        index.add(xb)
        assert index.ntotal == n
        D, I = index.search(xb[:5], 10)
        assert I.shape == (5, 10)

    def test_hnsw(self):
        d, n = 64, 2000
        np.random.seed(42)
        xb = np.random.random((n, d)).astype("float32")
        index = faiss.IndexHNSWFlat(d, 16)
        index.add(xb)
        assert index.ntotal == n
        D, I = index.search(xb[:5], 10)
        assert I.shape == (5, 10)
        # HNSW should find the vector itself as nearest neighbor
        np.testing.assert_array_equal(I[:, 0], np.arange(5))


class TestSerialization(unittest.TestCase):
    """Catch: broken I/O, buffer handling."""

    def test_serialize_deserialize_roundtrip(self):
        d = 32
        np.random.seed(42)
        xb = np.random.random((100, d)).astype("float32")
        index = faiss.IndexFlatL2(d)
        index.add(xb)

        # Serialize to bytes
        data = faiss.serialize_index(index)
        assert isinstance(data, np.ndarray)
        assert data.dtype == np.uint8

        # Deserialize
        index2 = faiss.deserialize_index(data)
        assert index2.ntotal == 100

        # Results should match
        D1, I1 = index.search(xb[:3], 5)
        D2, I2 = index2.search(xb[:3], 5)
        np.testing.assert_array_equal(I1, I2)
        np.testing.assert_allclose(D1, D2)


class TestGCSafety(unittest.TestCase):
    """Catch: Python reference counting bugs, crashes on GC."""

    def test_ivf_quantizer_lifecycle(self):
        d, n = 32, 5000
        np.random.seed(42)
        xb = np.random.random((n, d)).astype("float32")

        quantizer = faiss.IndexFlatL2(d)
        index = faiss.IndexIVFFlat(quantizer, d, 32)
        index.train(xb)
        index.add(xb)

        # Delete the quantizer reference — the index should still work
        # because faiss.__init__.py adds it to referenced_objects
        del quantizer

        D, I = index.search(xb[:3], 5)
        assert I.shape == (3, 5)


class TestContribImports(unittest.TestCase):
    """Catch: missing Python files in the wheel."""

    def test_contrib_modules(self):
        import faiss.contrib.datasets
        import faiss.contrib.evaluation
        import faiss.contrib.inspect_tools
        import faiss.contrib.ivf_tools
        import faiss.contrib.exhaustive_search
        import faiss.contrib.factory_tools
        import faiss.contrib.vecs_io  # noqa: F401

    def test_contrib_torch_subpackage(self):
        try:
            import torch  # noqa: F401
        except ImportError:
            raise unittest.SkipTest("torch not available")
        import faiss.contrib.torch
        import faiss.contrib.torch.clustering
        import faiss.contrib.torch.quantization  # noqa: F401


class TestSIMD(unittest.TestCase):
    """Catch: Dynamic Dispatch not selecting a SIMD path (x86_64 only)."""

    @unittest.skipIf(
        platform.machine() not in ("x86_64", "AMD64"),
        "SIMD level check only meaningful on x86_64",
    )
    @unittest.skipIf(
        platform.system() == "Windows",
        "Windows wheels are built generic (no SIMD) — see pyproject.toml",
    )
    def test_dd_selects_simd_path(self):
        opts = faiss.get_compile_options()
        if "DD" not in opts:
            raise unittest.SkipTest("Not a Dynamic Dispatch build")
        # DD builds should select at least AVX2 on modern x86_64
        assert "AVX2" in opts or "AVX512" in opts, (
            f"DD build did not select a SIMD path: {opts}"
        )
