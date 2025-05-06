/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/pq_code_distance/code_distance.h>

#include <cstddef>
#include <cstdint>

namespace faiss {

// explicit template instanciations
template struct PQCodeDistance<PQDecoder8, SIMDLevel::NONE>;
template struct PQCodeDistance<PQDecoder16, SIMDLevel::NONE>;
template struct PQCodeDistance<PQDecoderGeneric, SIMDLevel::NONE>;

} // namespace faiss
