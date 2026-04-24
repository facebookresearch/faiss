/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef HAMMING_GENERIC_INL_H
#define HAMMING_GENERIC_INL_H

// Scalar backend — no ISA-specific intrinsics.
// All HammingComputer / GenHammingComputer primary templates live in
// common.h (their default bodies are already the scalar fallback).
// This file is a shim kept for symmetry with avx2-inl.h / avx512-inl.h /
// neon-inl.h and for inclusion by hamdis-inl.h's ladder when no SIMD is
// available.

#include <faiss/utils/hamming_distance/common.h>

#endif
