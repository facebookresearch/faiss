# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Tests for Dynamic Dispatch (DD) SIMD level selection.

These tests verify that FAISS_SIMD_LEVEL environment variable correctly
controls which SIMD implementation is used at runtime.

NOTE: These tests only work with DD builds (FAISS_ENABLE_DD).
In static builds, the SIMD level is fixed at compile time.
"""

import os
import subprocess
import sys
import unittest


def get_available_simd_levels():
    """
    Returns SIMD levels that are available on the current platform.
    NONE is always available. Others depend on architecture.
    """
    import platform
    arch = platform.machine().lower()

    # SIMDLevel enum names
    levels = ["NONE"]

    if arch in ("x86_64", "amd64"):
        levels.extend(["AVX2", "AVX512"])
        # AVX512_SPR is typically not enabled in DD builds
    elif arch in ("aarch64", "arm64"):
        levels.append("ARM_NEON")

    return levels


class TestSIMDDispatch(unittest.TestCase):
    """Tests for SIMD dispatch via FAISS_SIMD_LEVEL environment variable."""

    def test_dispatch_function_exists(self):
        """Verify SIMDConfig.get_dispatched_level() is available."""
        try:
            import faiss
        except ImportError:
            self.skipTest("faiss not available")

        # Method should exist on SIMDConfig
        self.assertTrue(hasattr(faiss.SIMDConfig, 'get_dispatched_level'))

        # Should return a value (SIMDLevel enum)
        result = faiss.SIMDConfig.get_dispatched_level()
        # In SWIG, enum values are integers
        self.assertIsNotNone(result)

    def test_get_level_equals_get_dispatched_level(self):
        """Verify get_level() and get_dispatched_level() return the same value."""
        try:
            import faiss
        except ImportError:
            self.skipTest("faiss not available")

        level = faiss.SIMDConfig.get_level()
        dispatched = faiss.SIMDConfig.get_dispatched_level()
        self.assertEqual(level, dispatched)

    def test_dispatch_with_env_var(self):
        """
        Test that FAISS_SIMD_LEVEL controls dispatch.

        Runs faiss in subprocesses with different SIMD levels and verifies
        that get_dispatched_level() returns the expected value.
        """
        # Only run in DD mode
        try:
            import faiss
            if "DD" not in faiss.get_compile_options():
                self.skipTest("Not a DD build - SIMD level is fixed at compile time")
        except ImportError:
            self.skipTest("faiss not available")

        levels = get_available_simd_levels()

        # Get the PYTHONPATH that includes faiss module
        python_path = os.pathsep.join(sys.path)

        for level_name in levels:
            with self.subTest(level=level_name):
                # Run a subprocess with FAISS_SIMD_LEVEL set
                env = os.environ.copy()
                env["FAISS_SIMD_LEVEL"] = level_name
                env["PYTHONPATH"] = python_path

                script = f'''
import faiss
level = faiss.SIMDConfig.get_level()
dispatched = faiss.SIMDConfig.get_dispatched_level()
level_name = faiss.SIMDConfig.get_level_name()

# Verify dispatch matches
if level != dispatched:
    print(f"FAIL: get_level() != get_dispatched_level()")
    exit(1)

# Verify it's the expected level
if level_name != "{level_name}":
    print(f"FAIL: expected {level_name}, got {{level_name}}")
    exit(1)

print(f"OK: SIMD level {level_name} dispatched correctly")
'''
                result = subprocess.run(
                    [sys.executable, "-c", script],
                    env=env,
                    capture_output=True,
                    text=True
                )

                self.assertEqual(
                    result.returncode, 0,
                    f"Failed for {level_name}: {result.stdout} {result.stderr}"
                )


if __name__ == "__main__":
    unittest.main()
