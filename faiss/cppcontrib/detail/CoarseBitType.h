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
