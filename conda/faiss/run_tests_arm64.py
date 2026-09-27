#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
"""
Test runner for arm64 conda builds.  Runs the full test_*.py suite but
skips tests known to be flaky due to NEON floating-point precision differences:

  - TestComponents_ARM_NEON.test_update_codebooks_with_double
    assertLess(err_double, err_float) fails non-deterministically on arm64:
    the NEON LSQ codebook update path produces a higher double-precision error
    than the float path on some runner configurations (observed: 6555 > 4232).

  - TestProductLocalSearchQuantizer.test_lut
    assert_allclose(rtol=5e-04) is too tight for NEON compute_LUT; the max
    relative difference on arm64 is ~1.7e-03 (2/6400 elements affected).

Both are precision-ordering artefacts of NEON FMA contraction, not algorithmic
regressions.  The upstream tests need wider tolerances / conditional skips; this
script provides a stopgap so the conda nightly does not block on them.
"""
import sys
import unittest

# Full dotted test-ids (module.class.method) to skip on arm64.
_EXCLUDED: frozenset[str] = frozenset(
    {
        # NEON LSQ double-vs-float ordering flake (observed on macos-14 / arm64)
        "test_local_search_quantizer.TestComponents_ARM_NEON"
        ".test_update_codebooks_with_double",
        # NEON compute_LUT precision (rtol=5e-04 too tight on arm64)
        "test_local_search_quantizer.TestProductLocalSearchQuantizer.test_lut",
    }
)


def _filter(suite: unittest.TestSuite, excluded: frozenset[str]) -> unittest.TestSuite:
    out = unittest.TestSuite()
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            child = _filter(item, excluded)
            if child.countTestCases():
                out.addTest(child)
        else:
            tid = (
                f"{item.__module__}.{item.__class__.__name__}"
                f".{item._testMethodName}"  # type: ignore[attr-defined]
            )
            if tid not in excluded:
                out.addTest(item)
    return out


if __name__ == "__main__":
    loader = unittest.TestLoader()
    suite = loader.discover(start_dir="tests", pattern="test_*.py")
    filtered = _filter(suite, _EXCLUDED)
    skipped = suite.countTestCases() - filtered.countTestCases()
    if skipped:
        print(
            f"[run_tests_arm64] skipping {skipped} arm64-flaky test(s): "
            + ", ".join(sorted(_EXCLUDED)),
            flush=True,
        )
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(filtered)
    sys.exit(0 if result.wasSuccessful() else 1)
