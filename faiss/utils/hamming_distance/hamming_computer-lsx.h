/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef HAMMING_COMPUTER_LSX_H
#define HAMMING_COMPUTER_LSX_H

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include <lsxintrin.h>

#include <faiss/utils/hamming_distance/hamming_computer-generic.h>

namespace faiss {
namespace detail {

inline uint64_t lsx_hamming_popcount(__m128i value) {
    value = __lsx_vpcnt_d(value);
    value = __lsx_vadd_d(value, __lsx_vbsrl_v(value, 8));
    return static_cast<uint64_t>(__lsx_vpickve2gr_d(value, 0));
}

template <bool generalized>
inline int lsx_hamming_distance(__m128i a, __m128i b) {
    if constexpr (generalized) {
        auto equal = __lsx_vseq_b(a, b);
        // Count one bit per unequal byte.
        auto different = __lsx_vandi_b(__lsx_vnor_v(equal, equal), 1);
        return lsx_hamming_popcount(different);
    } else {
        return lsx_hamming_popcount(__lsx_vxor_v(a, b));
    }
}

template <size_t N>
inline __m128i lsx_hamming_load_partial(const uint8_t* p) {
    // Zero-pad without reading past the code.
    alignas(16) uint8_t bytes[16] = {};
    std::memcpy(bytes, p, N);
    return __lsx_vld(bytes, 0);
}

} // namespace detail

template <>
struct HammingComputer16_tpl<SIMDLevel::LOONGARCH_LSX> {
    __m128i data[1];

    HammingComputer16_tpl() = default;

    HammingComputer16_tpl(const uint8_t* a, int size) {
        set(a, size);
    }

    void set(const uint8_t* a, FAISS_MAYBE_UNUSED int size) {
        assert(size == 16);
        data[0] = __lsx_vld(a, 0);
    }

    int hamming(const uint8_t* b) const {
        int result = 0;
        result += detail::lsx_hamming_distance<false>(data[0], __lsx_vld(b, 0));
        return result;
    }

    static constexpr int get_code_size() {
        return 16;
    }
};

template <>
struct HammingComputer20_tpl<SIMDLevel::LOONGARCH_LSX> {
    __m128i data[2];

    HammingComputer20_tpl() = default;

    HammingComputer20_tpl(const uint8_t* a, int size) {
        set(a, size);
    }

    void set(const uint8_t* a, FAISS_MAYBE_UNUSED int size) {
        assert(size == 20);
        data[0] = __lsx_vld(a, 0);
        data[1] = detail::lsx_hamming_load_partial<4>(a + 16);
    }

    int hamming(const uint8_t* b) const {
        int result = 0;
        result += detail::lsx_hamming_distance<false>(data[0], __lsx_vld(b, 0));
        result += detail::lsx_hamming_distance<false>(
                data[1], detail::lsx_hamming_load_partial<4>(b + 16));
        return result;
    }

    static constexpr int get_code_size() {
        return 20;
    }
};

template <>
struct HammingComputer32_tpl<SIMDLevel::LOONGARCH_LSX> {
    __m128i data[2];

    HammingComputer32_tpl() = default;

    HammingComputer32_tpl(const uint8_t* a, int size) {
        set(a, size);
    }

    void set(const uint8_t* a, FAISS_MAYBE_UNUSED int size) {
        assert(size == 32);
        for (int i = 0; i < 2; ++i) {
            data[i] = __lsx_vld(a + i * 16, 0);
        }
    }

    int hamming(const uint8_t* b) const {
        int result = 0;
        for (int i = 0; i < 2; ++i) {
            result += detail::lsx_hamming_distance<false>(
                    data[i], __lsx_vld(b + i * 16, 0));
        }
        return result;
    }

    static constexpr int get_code_size() {
        return 32;
    }
};

template <>
struct HammingComputer64_tpl<SIMDLevel::LOONGARCH_LSX> {
    __m128i data[4];

    HammingComputer64_tpl() = default;

    HammingComputer64_tpl(const uint8_t* a, int size) {
        set(a, size);
    }

    void set(const uint8_t* a, FAISS_MAYBE_UNUSED int size) {
        assert(size == 64);
        for (int i = 0; i < 4; ++i) {
            data[i] = __lsx_vld(a + i * 16, 0);
        }
    }

    int hamming(const uint8_t* b) const {
        int result = 0;
        for (int i = 0; i < 4; ++i) {
            result += detail::lsx_hamming_distance<false>(
                    data[i], __lsx_vld(b + i * 16, 0));
        }
        return result;
    }

    static constexpr int get_code_size() {
        return 64;
    }
};

template <>
struct GenHammingComputer8_tpl<SIMDLevel::LOONGARCH_LSX> {
    __m128i data[1];

    GenHammingComputer8_tpl() = default;

    GenHammingComputer8_tpl(const uint8_t* a, int size) {
        set(a, size);
    }

    void set(const uint8_t* a, FAISS_MAYBE_UNUSED int size) {
        assert(size == 8);
        data[0] = detail::lsx_hamming_load_partial<8>(a + 0);
    }

    int hamming(const uint8_t* b) const {
        int result = 0;
        result += detail::lsx_hamming_distance<true>(
                data[0], detail::lsx_hamming_load_partial<8>(b + 0));
        return result;
    }

    static constexpr int get_code_size() {
        return 8;
    }
};

template <>
struct GenHammingComputer16_tpl<SIMDLevel::LOONGARCH_LSX> {
    __m128i data[1];

    GenHammingComputer16_tpl() = default;

    GenHammingComputer16_tpl(const uint8_t* a, int size) {
        set(a, size);
    }

    void set(const uint8_t* a, FAISS_MAYBE_UNUSED int size) {
        assert(size == 16);
        data[0] = __lsx_vld(a, 0);
    }

    int hamming(const uint8_t* b) const {
        int result = 0;
        result += detail::lsx_hamming_distance<true>(data[0], __lsx_vld(b, 0));
        return result;
    }

    static constexpr int get_code_size() {
        return 16;
    }
};

template <>
struct GenHammingComputer32_tpl<SIMDLevel::LOONGARCH_LSX> {
    __m128i data[2];

    GenHammingComputer32_tpl() = default;

    GenHammingComputer32_tpl(const uint8_t* a, int size) {
        set(a, size);
    }

    void set(const uint8_t* a, FAISS_MAYBE_UNUSED int size) {
        assert(size == 32);
        for (int i = 0; i < 2; ++i) {
            data[i] = __lsx_vld(a + i * 16, 0);
        }
    }

    int hamming(const uint8_t* b) const {
        int result = 0;
        for (int i = 0; i < 2; ++i) {
            result += detail::lsx_hamming_distance<true>(
                    data[i], __lsx_vld(b + i * 16, 0));
        }
        return result;
    }

    static constexpr int get_code_size() {
        return 32;
    }
};

template <>
struct HammingComputerDefault_tpl<SIMDLevel::LOONGARCH_LSX> {
    const uint8_t* a;
    int size;

    HammingComputerDefault_tpl() = default;

    HammingComputerDefault_tpl(const uint8_t* a, int size) {
        set(a, size);
    }

    void set(const uint8_t* a_in, int size_in) {
        a = a_in;
        size = size_in;
    }

    int hamming(const uint8_t* b) const {
        int result = 0;
        size_t i = 0;
        for (; i + 16 <= size; i += 16) {
            result += detail::lsx_hamming_distance<false>(
                    __lsx_vld(a + i, 0), __lsx_vld(b + i, 0));
        }
        for (; i < size; ++i) {
            result += popcount32(a[i] ^ b[i]);
        }
        return result;
    }

    int get_code_size() const {
        return size;
    }
};

template <>
struct GenHammingComputerM8_tpl<SIMDLevel::LOONGARCH_LSX> {
    const uint8_t* a;
    int size;

    GenHammingComputerM8_tpl() = default;

    GenHammingComputerM8_tpl(const uint8_t* a, int size) {
        set(a, size);
    }

    void set(const uint8_t* a_in, int size_in) {
        a = a_in;
        size = size_in;
    }

    int hamming(const uint8_t* b) const {
        int result = 0;
        size_t i = 0;
        for (; i + 16 <= size; i += 16) {
            result += detail::lsx_hamming_distance<true>(
                    __lsx_vld(a + i, 0), __lsx_vld(b + i, 0));
        }
        for (; i < size; ++i) {
            result += a[i] != b[i];
        }
        return result;
    }

    int get_code_size() const {
        return size;
    }
};

} // namespace faiss

#endif
