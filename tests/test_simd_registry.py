# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Checks the SIMD guard rules in faiss/docs/simd_dynamic_dispatch_migration.md:
per-level files are registered in the build lists and have no whole-file
guard, and only per-level code and a fixed list of headers use SIMD
intrinsics or compiler macros for SIMD features."""

import ast
import os
import re
import unittest
from importlib import resources
from pathlib import Path

# CMake list name suffix -> SIMDLevel name.
CMAKE_LEVELS = {
    "AVX2": "AVX2",
    "AVX512": "AVX512",
    "AVX512_VPOPCNT": "AVX512_VPOPCNT",
    "AVX512_SPR": "AVX512_SPR",
    "NEON": "ARM_NEON",
    "SVE": "ARM_SVE",
    "RVV": "RISCV_RVV",
}

# Directories that are not part of the CPU library.
EXCLUDED_DIRS = {
    "benchs",
    "c_api",
    "contrib",
    "cppcontrib",
    "demos",
    "docs",
    "fb",
    "gpu",
    "gpu_rocm",
    "perf_tests",
    "public_tld",
    "python",
    "tests",
    "tutorial",
}

# A file whose name contains one of these words, between "-", "_" or ".",
# holds per-level code. Only per-level files include such a header ("sse": x86
# code for the AVX2 and AVX512 files).
PER_LEVEL_NAME = re.compile(
    r"(?:^|[-_.])(?:avx2|avx512|neon|sve|rvv|riscv|ppc64|sse)(?=[-_.])"
)

# Headers that common files include and that select SIMD code with compiler
# macros. Add a header here only after review.
COMPILER_MACRO_HEADERS = {
    "impl/Panorama.h",  # __BMI2__ next to FAISS_SIMD_TU(AVX2)
    "impl/platform_macros.h",  # MSVC defines of __F16C__ and __FMA__
    "impl/simdlib/simdlib.h",
    "utils/bf16.h",
    "utils/fp16-fp16c.h",
    "utils/fp16-arm.h",
    "utils/fp16.h",
    "utils/hamming_distance/common.h",  # NEON under __aarch64__
    "utils/prefetch.h",  # architecture helper
    "utils/simd_levels.h",  # defines FAISS_SIMD_TU
}

# Common files that test COMPILE_SIMD_<L>. They only declare, reference or
# dispatch to per-level code. Add a file here only after review.
COMPILE_SIMD_FILES = {
    "impl/ProductQuantizer.cpp",  # selects the dsub2 kernels
    "impl/pq_code_distance/pq_code_distance-generic.h",  # scalar NEON body
    "impl/scalar_quantizer/EDENQuantizer.cpp",
    "impl/simd_dispatch.h",
    "utils/simd_impl/fp16_kernels.h",
    "utils/simd_impl/super_kmeans_kernels.h",
    "utils/simd_levels.cpp",
    "utils/simd_levels.h",
}

# Common headers that include per-level headers under FAISS_SIMD_TU or an
# architecture macro. Add a header here only after review.
PER_LEVEL_INCLUDERS = {
    "impl/fast_scan/accumulate_loops.h",
    "impl/fast_scan/decompose_qbs.h",
    "impl/simdlib/simdlib.h",
    "impl/simdlib/simdlib_dispatch.h",
}

ISA_MACRO = re.compile(
    r"\b__(?:AVX\w*__|F16C__|FMA4?__|BMI2?__|LZCNT__|POPCNT__|PCLMUL__"
    r"|VPCLMULQDQ__|GFNI__|V?AES__|SHA__|SSE\w*__|ARM_FEATURE_\w+"
    r"|ARM_NEON\w*|riscv_vector)\b"
)
# Not <intrin.h>: MSVC also needs it for scalar helpers such as __cpuid.
# INTRINSIC_CODE catches vector intrinsics used through it.
INTRINSIC_INCLUDE = re.compile(
    r"#\s*include\s*<(?:[a-z]mmintrin|x86intrin|x86gprintrin|immintrin|arm_neon"
    r"|arm_sve|arm_acle|arm_fp16|arm_bf16|riscv_vector)\.h>"
)
# Intrinsic calls and vector types (x86, NEON, SVE, RVV). The SVE call
# pattern excludes "svs_", the prefix of the SVS library functions.
INTRINSIC_CODE = re.compile(
    r"\b_mm\w*_\w+\s*\(|\b__m(?:128|256|512)[dhi]?\b|\b__mmask\d+\b"
    r"|\b(?:u?int|float|poly)(?:8|16|32|64)x\d+(?:x\d+)?_t\b"
    r"|\bv[a-z0-9_]+_[supf](?:8|16|32|64)\s*\("
    r"|\bsv(?:float|int|uint|bool)\d*_t\b"
    r"|\bsv(?!s_)\w+_[suf](?:8|16|32|64)(?:_[xmz])?\s*\("
    r"|\b__riscv_v\w+\s*\("
)
COMPILE_SIMD = re.compile(r"\bCOMPILE_SIMD_\w+")
LEVEL_WORD = re.compile(
    r"(?:^|[-_.])(avx2|avx512|neon|sve|rvv|riscv|ppc64|sse)(?=[-_.])"
)
# The levels that a file with this word in its name can belong to.
WORD_LEVELS = {
    "avx2": {"AVX2"},
    "avx512": {"AVX512", "AVX512_VPOPCNT", "AVX512_SPR"},
    "sse": {"AVX2", "AVX512", "AVX512_VPOPCNT", "AVX512_SPR"},
    "neon": {"ARM_NEON"},
    "sve": {"ARM_SVE"},
    "rvv": {"RISCV_RVV"},
    "riscv": {"RISCV_RVV"},
}
CONDITION = re.compile(r"^\s*#\s*(?:if|ifdef|ifndef|elif)\b")
OPEN = re.compile(r"^\s*#\s*(?:if|ifdef|ifndef)\b(.*)$")
CLOSE = re.compile(r"^\s*#\s*endif\b")
CODE = re.compile(r"^\s*(?!#|//|/\*|\*)\S")
INCLUDE = re.compile(r'^\s*#\s*include\s*(?:"([^"]+)"|<(faiss/[^>]+)>)')


# base_module of the test in tests/BUCK.
BUCK_PACKAGE = "faiss_simd_registry"


def find_faiss_root():
    candidates = [os.environ.get("FAISS_SOURCE_DIR")]
    if __package__:
        # Buck packages the sources as a resource of this test.
        try:
            sources = resources.files(__package__) / "faiss_sources"
            candidates.append(str(sources))
        except (ModuleNotFoundError, TypeError):
            pass
    here = Path(__file__).resolve().parent
    candidates += [here.parent / "faiss", here.parent]
    for candidate in candidates:
        if candidate is None:
            continue
        root = Path(candidate)
        if (root / "CMakeLists.txt").is_file() and (
            root / "utils" / "simd_levels.h"
        ).is_file():
            return root
    return None


def cmake_list(cmake_text, name):
    match = re.search(r"set\(" + name + r"\s*\n(.*?)\)", cmake_text, re.S)
    return match.group(1).split() if match else []


def cmake_simd_files(root):
    text = (root / "CMakeLists.txt").read_text()
    files = {}
    for suffix, level in CMAKE_LEVELS.items():
        for path in cmake_list(text, f"FAISS_SIMD_{suffix}_SRC"):
            files[path] = level
    return files


def buck_simd_files(root):
    tree = ast.parse((root / "xplat.bzl").read_text())
    for node in tree.body:
        if (
            isinstance(node, ast.Assign)
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "SIMD_FILES"
        ):
            return {
                key.value: value.elts[1].id
                for key, value in zip(node.value.keys, node.value.values)
            }
    return {}


def logical_lines(path):
    """(line number, line) with each backslash continuation joined."""
    number, parts = 0, []
    for i, line in enumerate(path.read_text(errors="replace").splitlines(), 1):
        if not parts:
            number = i
        if line.endswith("\\"):
            parts.append(line[:-1])
            continue
        parts.append(line)
        yield number, " ".join(parts)
        parts = []
    if parts:
        yield number, " ".join(parts)


def library_files(root):
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if path.suffix not in (".h", ".cpp") or rel.parts[0] in EXCLUDED_DIRS:
            continue
        yield rel.as_posix(), path


def whole_file_guard(path):
    """The condition of an #if that encloses all code of the file, or None."""
    lines = [line for _, line in logical_lines(path)]
    starts = [
        i
        for i, line in enumerate(lines)
        if OPEN.match(line) or CODE.match(line)
    ]
    first = starts[0] if starts else None
    if first is None or not OPEN.match(lines[first]):
        return None
    depth = 0
    for i in range(first, len(lines)):
        if OPEN.match(lines[i]):
            depth += 1
        elif CLOSE.match(lines[i]):
            depth -= 1
            if depth == 0:
                rest = lines[i + 1 :]
                if any(CODE.match(line) for line in rest):
                    return None
                return OPEN.match(lines[first]).group(1).strip()
    return None


def common_line_error(rel, line):
    """The rule that a line of a common file breaks, or None."""
    macro_header = rel in COMPILER_MACRO_HEADERS
    if INTRINSIC_INCLUDE.search(line) and not macro_header:
        return "intrinsic header"
    if CONDITION.match(line):
        if ISA_MACRO.search(line) and not macro_header:
            return "compiler macro for a SIMD feature; use FAISS_SIMD_TU(level)"
        if COMPILE_SIMD.search(line) and rel not in COMPILE_SIMD_FILES:
            return (
                "COMPILE_SIMD_<level> in a common file; level code uses "
                "FAISS_SIMD_TU(level), and a declaration or a dispatch needs "
                "review (COMPILE_SIMD_FILES)"
            )
    include = INCLUDE.match(line)
    if include:
        header = include.group(1) or include.group(2)
        if (
            PER_LEVEL_NAME.search(Path(header).name)
            and rel not in PER_LEVEL_INCLUDERS
        ):
            return f"per-level header {header} in a common file"
    code = line.split("//")[0]
    if INTRINSIC_CODE.search(code) and not macro_header:
        return "intrinsic call or vector type in a common file"
    return None


class SIMDRegistryTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = find_faiss_root()
        if cls.root is None:
            # Buck declares the sources as a resource, and FAISS_SOURCE_DIR
            # names them, so they must exist then. Other runs, such as the
            # conda package tests, may have only the tests directory.
            expected = __package__ == BUCK_PACKAGE or os.environ.get(
                "FAISS_SOURCE_DIR"
            )
            if expected:
                raise RuntimeError(
                    "Faiss source files not found; check FAISS_SOURCE_DIR or "
                    "the //faiss:simd_registry_sources resource"
                )
            raise unittest.SkipTest(
                "Faiss source files not found; set FAISS_SOURCE_DIR"
            )
        cls.from_buck = cls.root.name == "faiss_sources"
        cls.cmake = (cls.root / "CMakeLists.txt").read_text()
        cls.simd_files = cmake_simd_files(cls.root)

    def test_sources_are_complete(self):
        # A parser that misses a list, or a resource that misses files, would
        # make the other checks pass on nothing.
        for suffix in CMAKE_LEVELS:
            self.assertGreater(
                len(cmake_list(self.cmake, f"FAISS_SIMD_{suffix}_SRC")),
                0,
                f"FAISS_SIMD_{suffix}_SRC is empty or not parsed",
            )
        common = cmake_list(self.cmake, "FAISS_SRC")
        self.assertIn("IndexFlat.cpp", common, "FAISS_SRC is not parsed")
        rels = {rel for rel, _ in library_files(self.root)}
        known = ("IndexFlat.cpp", "utils/simd_levels.h", "impl/simd_dispatch.h")
        for rel in known:
            self.assertIn(rel, rels)
        self.assertGreater(len(rels), 300, "too few library files found")

    def test_per_level_files_exist_and_are_not_common(self):
        self.assertGreater(len(self.simd_files), 0)
        common = set(cmake_list(self.cmake, "FAISS_SRC"))
        for path in self.simd_files:
            self.assertTrue((self.root / path).is_file(), path)
            self.assertNotIn(path, common, f"{path} is in FAISS_SRC")
        for path in common:
            self.assertIsNone(
                PER_LEVEL_NAME.search(Path(path).name),
                f"{path}: a per-level file is in FAISS_SRC",
            )

    def test_per_level_files_are_registered(self):
        # A per-level .cpp in no list is never compiled.
        for rel, path in library_files(self.root):
            if path.suffix == ".cpp" and LEVEL_WORD.search(path.name):
                message = f"{rel}: in no FAISS_SIMD_<L>_SRC list"
                self.assertIn(rel, self.simd_files, message)

    def test_file_names_match_levels(self):
        # An avx512 file in the AVX2 list of both build files passes the
        # parity check, but gets the wrong flags.
        for path, level in self.simd_files.items():
            for word in LEVEL_WORD.findall(Path(path).name):
                allowed = WORD_LEVELS.get(word)
                if allowed is not None:
                    self.assertIn(level, allowed, f"{path}: listed as {level}")

    def test_per_level_files_have_no_whole_file_guard(self):
        # The build files select per-level files, so a whole-file guard on a
        # level or on a compiler macro is always true, or always false and
        # empties the file.
        for path in self.simd_files:
            condition = whole_file_guard(self.root / path)
            guarded = condition is not None and bool(
                COMPILE_SIMD.search(condition) or ISA_MACRO.search(condition)
            )
            self.assertFalse(
                guarded, f"{path}: remove the whole-file guard {condition}"
            )

    def test_simd_code_only_in_allowed_files(self):
        errors = []
        for rel, path in library_files(self.root):
            if rel in self.simd_files or PER_LEVEL_NAME.search(path.name):
                continue
            for number, line in logical_lines(path):
                error = common_line_error(rel, line)
                if error:
                    errors.append(f"{rel}:{number}: {error}")
        self.assertEqual(errors, [], "\n".join(errors))

    def test_buck_and_cmake_list_the_same_files(self):
        if not (self.root / "xplat.bzl").is_file():
            # The GitHub checkout has no xplat.bzl; the Buck resource must.
            self.assertFalse(self.from_buck, "xplat.bzl is missing")
            self.skipTest("xplat.bzl is not available")
        buck = buck_simd_files(self.root)
        self.assertGreater(len(buck), 0, "SIMD_FILES is empty or not parsed")
        cmake = {
            path: level
            for path, level in self.simd_files.items()
            if level != "RISCV_RVV"
        }
        self.assertEqual(buck, cmake)
