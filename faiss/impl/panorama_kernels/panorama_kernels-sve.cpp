/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// ARM SVE implementations of Panorama kernels.
// TODO(@AlSchlo, @aknayar): implement SVE-optimized panorama kernels.
// Currently delegates to the scalar (NONE) implementation.

#ifdef COMPILE_SIMD_ARM_SVE

#include <faiss/impl/panorama_kernels/panorama_kernels-inl.h>

namespace faiss {
namespace panorama_kernels {

// NOLINTNEXTLINE(facebook-hte-MisplacedTemplateSpecialization)
template <>
void process_level_impl<SIMDLevel::ARM_SVE>(
        size_t level_width_bytes,
        size_t max_batch_size,
        size_t num_active,
        float* sim_table,
        uint8_t* compressed_codes,
        float* exact_distances) {
    process_level_impl<SIMDLevel::NONE>(
            level_width_bytes,
            max_batch_size,
            num_active,
            sim_table,
            compressed_codes,
            exact_distances);
}

// NOLINTNEXTLINE(facebook-hte-MisplacedTemplateSpecialization)
template <>
std::pair<uint8_t*, size_t> process_code_compression_impl<SIMDLevel::ARM_SVE>(
        size_t next_num_active,
        size_t max_batch_size,
        size_t level_width_bytes,
        uint8_t* compressed_codes_begin,
        uint8_t* bitset,
        const uint8_t* codes) {
    return process_code_compression_impl<SIMDLevel::NONE>(
            next_num_active,
            max_batch_size,
            level_width_bytes,
            compressed_codes_begin,
            bitset,
            codes);
}

} // namespace panorama_kernels
} // namespace faiss

#endif // COMPILE_SIMD_ARM_SVE
