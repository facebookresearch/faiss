/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

/** Batch construction of an HNSW graph over 8-bit inner-product RaBitQ
 * codes, used by IndexHNSWRaBitQ::add when the index is empty.
 *
 * Instead of inserting the vectors one by one, the exact code-to-code top-k
 * of the whole database is computed with integer matrix products (AMX-INT8,
 * or AVX512-VNNI as a fallback), then each level is pruned with the HNSW
 * heuristic and completed with reverse edges.
 *
 * This header contains no ISA-specific code. The kernels are compiled in the
 * AVX512_SPR SIMD level only (batch_build-avx512-spr.cpp); in other builds
 * select_backend() returns Backend::none and the caller keeps the regular
 * insertion path.
 */

#include <cstddef>
#include <cstdint>

#if defined(COMPILE_SIMD_AVX512_SPR) && !defined(_MSC_VER)
#define FAISS_RABITQ_BATCH_BUILD 1
#else
#define FAISS_RABITQ_BATCH_BUILD 0
#endif

namespace faiss {

struct HNSW;

namespace rabitq_batch {

enum class Backend { none, vnni, amx };

/// Maximum dimension: 65025 * d < 2^31, so that the int32 kernels are exact.
constexpr size_t max_d = 32768;

/** Backend available on this CPU with this build: AMX if allow_amx and the
 * CPU and OS support AMX-INT8, otherwise AVX512-VNNI if available, otherwise
 * none. The first call requests the AMX tile data permission from the OS. */
Backend select_backend(bool allow_amx);

const char* backend_name(Backend backend);

/** Builds all levels of h over n codes of code_size bytes, with candidate
 * lists of length k. The levels and offsets of h must have been set by
 * HNSW::prepare_level_tab; the entry point is chosen as in
 * hnsw_add_vertices_deterministic.
 *
 * @param center_norm_sq  squared norm of the quantizer centroid
 * @param backend         must not be Backend::none
 */
void build(
        HNSW& h,
        const uint8_t* codes,
        size_t n,
        size_t d,
        size_t code_size,
        double center_norm_sq,
        size_t k,
        Backend backend);

#if FAISS_RABITQ_BATCH_BUILD
/// implementation, in batch_build-avx512-spr.cpp
void build_avx512_spr(
        HNSW& h,
        const uint8_t* codes,
        size_t n,
        size_t d,
        size_t code_size,
        double center_norm_sq,
        size_t k,
        bool vnni);
#endif

} // namespace rabitq_batch
} // namespace faiss
