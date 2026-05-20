/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef COMPILE_SIMD_AVX512

#include <faiss/impl/simdlib/simdlib_avx2.h>
#include <faiss/impl/simdlib/simdlib_avx512.h>
#include <faiss/impl/EDENQuantizerDistance-optimized.h>

namespace faiss {

namespace eden_distance {

template <>
EDENFlatCodesDistanceComputer* make_distance_computer<SIMDLevel::AVX512>(
        MetricType metric_type,
        size_t d,
        size_t nb_bits,
        const float* centroid) {
    return make_optimized_distance_computer<SIMDLevel::AVX512>(
            metric_type, d, nb_bits, centroid);
}

} // namespace eden_distance

} // namespace faiss

#endif
