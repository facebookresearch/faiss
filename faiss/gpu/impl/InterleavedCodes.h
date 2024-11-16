/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stdint.h>
#include <vector>

// Utilities for bit packing and unpacking CPU non-interleaved and GPU
// interleaved by 32 encodings
namespace faiss {
namespace gpu {

// Unpacks arbitrary bitwidth codes to a whole number of bytes per code
// The layout of the input is (v0 d0)(v0 d1) ... (v0 dD)(v1 d0) ...
// (bit packed)
// The layout of the output is the same (byte packed to roundUp(bitsPerCode, 8)
// / 8 bytes)
std::vector<uint8_t> unpackNonInterleaved(
        std::vector<uint8_t> data,
        int numVecs,
        int dims,
        int bitsPerCode);

// Unpacks arbitrary bitwidth codes to a whole number of bytes per scalar code
// The layout of the input is (v0 d0)(v1 d0) ... (v31 d0)(v0 d1) ...
// (bit packed)
// The layout of the input is (v0 d0)(v0 d1) ... (v0 dD)(v1 d0) ...
// (byte packed)
std::vector<uint8_t> unpackInterleaved(
        std::vector<uint8_t> data,
        int numVecs,
        int dims,
        int bitsPerCode);

// Packs data in the byte packed non-interleaved form to bit packed
// non-interleaved form
std::vector<uint8_t> packNonInterleaved(
        std::vector<uint8_t> data,
        int numVecs,
        int dims,
        int bitsPerCode);

// Packs data in the byte packed non-interleaved form to bit packed
// interleaved form
std::vector<uint8_t> packInterleaved(
        std::vector<uint8_t> data,
        int numVecs,
        int dims,
        int bitsPerCode);

} // namespace gpu
} // namespace faiss
