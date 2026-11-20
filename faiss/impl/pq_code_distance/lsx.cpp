/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#define THE_SIMD_LEVEL SIMDLevel::LOONGARCH_LSX

// NOLINTNEXTLINE(facebook-hte-InlineHeader)
#include <faiss/impl/pq_code_distance/pq_code_distance-generic.h>
// NOLINTNEXTLINE(facebook-hte-InlineHeader)
#include <faiss/utils/hamming_distance/hamming_computer-lsx.h>
// NOLINTNEXTLINE(facebook-hte-InlineHeader)
#include <faiss/impl/pq_code_distance/pq_scan_impl.h>
// NOLINTNEXTLINE(facebook-hte-InlineHeader)
#include <faiss/impl/pq_code_distance/PQDistanceComputer_impl.h>
// NOLINTNEXTLINE(facebook-hte-InlineHeader)
#include <faiss/impl/pq_code_distance/IVFPQScanner_impl.h>
