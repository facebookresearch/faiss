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


def get_cpu_flags():
    """Return the set of CPU feature flags from /proc/cpuinfo, or None.

    On Linux x86_64 this includes tokens like 'avx512f', 'avx512_bf16' and
    'avx512_fp16' - the OS-reported ground truth for SIMD feature detection.
    """
    try:
        with open("/proc/cpuinfo") as f:
            cpuinfo = f.read()
    except OSError:
        return None

    for line in cpuinfo.splitlines():
        # 'flags' on x86, 'Features' on aarch64.
        if line.startswith("flags") or line.startswith("Features"):
            return set(line.split(":", 1)[1].split())
    return set()


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
    elif arch in ("riscv64", "riscv"):
        levels.append("RISCV_RVV")

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
        self.assertTrue(hasattr(faiss.SIMDConfig, "get_dispatched_level"))

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

                script = f"""
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
"""
                result = subprocess.run(
                    [sys.executable, "-c", script],
                    env=env,
                    capture_output=True,
                    text=True,
                )

                self.assertEqual(
                    result.returncode,
                    0,
                    f"Failed for {level_name}: {result.stdout} {result.stderr}",
                )

    def test_detected_level_does_not_crash_fp16(self):
        """The auto-detected SIMD level must always be safe to execute on the
        running CPU. AVX512_SPR is compiled with -mavx512fp16, so it must
        only be detected on CPUs that actually have AVX512_FP16. AMD Zen 4
        (bergamo) has AVX512_BF16 but NOT AVX512_FP16; before the fix,
        detection classified it as SPR and SQfp16 distance computations
        dispatched to SPR code, which crashed.
        """
        try:
            import faiss

            if "DD" not in faiss.get_compile_options():
                self.skipTest("Not a DD build - SIMD level is fixed at compile time")
        except ImportError:
            self.skipTest("faiss not available")

        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(sys.path)

        script = """
import faiss
import numpy as np

d = 32
rs = np.random.RandomState(123)
xb = rs.rand(2000, d).astype("float32")
xq = rs.rand(20, d).astype("float32")


def exercise_sqfp16():
    index = faiss.index_factory(d, "SQfp16")
    index.train(xb)
    index.add(xb)
    # SQfp16 distance computations are the code path that SIGILL'd when
    # SPR was mis-detected on a CPU without AVX512_FP16.
    index.search(xq, 10)


# Auto-detected level: this is exactly what the detection fix controls.
exercise_sqfp16()
print("OK auto", faiss.SIMDConfig.get_level_name())

# Every level that detection reports as available must also be safe to run.
for lvl in range(int(faiss.SIMDLevel_COUNT)):
    if faiss.SIMDConfig.is_simd_level_available(lvl):
        faiss.SIMDConfig.set_level(lvl)
        exercise_sqfp16()
        print("OK", faiss.SIMDConfig.get_level_name())
"""
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            capture_output=True,
            text=True,
        )

        # A negative return code means the child was killed by a signal;
        # -signal.SIGILL is the regression we are guarding against.
        self.assertEqual(
            result.returncode,
            0,
            "SQfp16 crashed at a detected SIMD level (likely SIGILL from "
            f"SPR mis-detection, D107684495). returncode={result.returncode}\n"
            f"stdout: {result.stdout}\nstderr: {result.stderr}",
        )

    def test_spr_detection_matches_cpu_features(self):
        """SPR detection must agree with the CPU's real feature flags. The SPR
        code path is compiled with -mavx512fp16, so AVX512_SPR must be
        reported available if and only if the CPU actually has the full
        AVX512 core feature set AND AVX512_BF16 AND AVX512_FP16.
        """
        import platform

        try:
            import faiss

            if "DD" not in faiss.get_compile_options():
                self.skipTest("Not a DD build - SIMD level is fixed at compile time")
        except ImportError:
            self.skipTest("faiss not available")

        if platform.machine().lower() not in ("x86_64", "amd64"):
            self.skipTest("x86_64-only test")

        flags = get_cpu_flags()
        if flags is None:
            self.skipTest("/proc/cpuinfo not available")

        # The exact feature set faiss requires before selecting AVX512_SPR.
        avx512_core = {"avx512f", "avx512cd", "avx512vl", "avx512dq", "avx512bw"}
        spr_capable = (
            avx512_core <= flags and "avx512_bf16" in flags and "avx512_fp16" in flags
        )

        spr_detected = faiss.SIMDConfig.is_simd_level_available(
            faiss.SIMDLevel_AVX512_SPR
        )

        self.assertEqual(
            spr_detected,
            spr_capable,
            "AVX512_SPR detection disagrees with /proc/cpuinfo: "
            f"detected={spr_detected}, cpu_spr_capable={spr_capable} "
            f"(avx512_core={avx512_core <= flags}, "
            f"bf16={'avx512_bf16' in flags}, fp16={'avx512_fp16' in flags}). "
            "detected=True with fp16=False is the D107684495 regression "
            "(AMD Zen 4 mis-detected as SPR).",
        )


if __name__ == "__main__":
    unittest.main()
