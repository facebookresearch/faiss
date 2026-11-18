/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef COMPILE_SIMD_ARM_SVE

#define THE_SIMD_LEVEL SIMDLevel::ARM_SVE

// NOLINTNEXTLINE(facebook-hte-InlineHeader)
#include <faiss/impl/pq_code_distance/pq_code_distance-generic.h>
// NOLINTNEXTLINE(facebook-hte-InlineHeader)
#include <faiss/utils/hamming_distance/hamming_computer-sve.h>
// pq_scan_impl.h for ARM_SVE is instantiated by pq_code_distance-sve.cpp
// (hand-written SVE kernels); including it here too would duplicate
// pq_scan_*_impl<ARM_SVE> symbols.
// NOLINTNEXTLINE(facebook-hte-InlineHeader)
#include <faiss/impl/pq_code_distance/PQDistanceComputer_impl.h>
// NOLINTNEXTLINE(facebook-hte-InlineHeader)
#include <faiss/impl/pq_code_distance/IVFPQScanner_impl.h>

#endif // COMPILE_SIMD_ARM_SVE
