/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

/** Abstractions for 256-bit registers
 *
 * The objective is to separate the different interpretations of the same
 * registers (as a vector of uint8, uint16 or uint32), to provide printing
 * functions.
 */

#ifdef __AVX2__

#include <faiss/utils/simdlib_avx2.h>

#elif defined(__aarch64__)

#include <faiss/utils/simdlib_neon.h>

#else

// emulated = all operations are implemented as scalars
#include <faiss/utils/simdlib_emulated.h>

// FIXME: make a SSE version
// is this ever going to happen? We will probably rather implement AVX512

#endif
