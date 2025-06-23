/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>

namespace faiss {
namespace cppcontrib {
namespace detail {

template <int COARSE_BITS>
struct CoarseBitType {};

template <>
struct CoarseBitType<8> {
    using bit_type = uint8_t;
};

template <>
struct CoarseBitType<16> {
    using bit_type = uint16_t;
};

} // namespace detail
} // namespace cppcontrib
} // namespace faiss
