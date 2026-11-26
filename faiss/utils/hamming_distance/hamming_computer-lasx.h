/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef HAMMING_COMPUTER_LASX_H
#define HAMMING_COMPUTER_LASX_H

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstring>

#include <lasxintrin.h>
#include <lsxintrin.h>

#include <faiss/utils/hamming_distance/hamming_computer-generic.h>

namespace faiss {
namespace detail {

inline uint64_t lasx_hamming_popcount(__m128i value) {
    value = __lsx_vpcnt_d(value);
    value = __lsx_vadd_d(value, __lsx_vbsrl_v(value, 8));
    return static_cast<uint64_t>(__lsx_vpickve2gr_d(value, 0));
}

template <bool generalized>
inline int lasx_hamming_distance(__m128i a, __m128i b) {
    if constexpr (generalized) {
        auto equal = __lsx_vseq_b(a, b);
        // Count one bit per unequal byte.
        auto different = __lsx_vandi_b(__lsx_vnor_v(equal, equal), 1);
        return lasx_hamming_popcount(different);
    } else {
        return lasx_hamming_popcount(__lsx_vxor_v(a, b));
    }
}

inline uint64_t lasx_hamming_popcount(__m256i value) {
    value = __lasx_xvpcnt_d(value);
    value = __lasx_xvadd_d(value, __lasx_xvpermi_q(value, value, 0x01));
    value = __lasx_xvadd_d(value, __lasx_xvbsrl_v(value, 8));
    return static_cast<uint64_t>(__lasx_xvpickve2gr_d(value, 0));
}

template <bool generalized>
inline int lasx_hamming_distance(__m256i a, __m256i b) {
    if constexpr (generalized) {
        auto equal = __lasx_xvseq_b(a, b);
        // Count one bit per unequal byte.
        auto different = __lasx_xvandi_b(__lasx_xvnor_v(equal, equal), 1);
        return lasx_hamming_popcount(different);
    } else {
        return lasx_hamming_popcount(__lasx_xvxor_v(a, b));
    }
}

template <size_t N>
inline __m128i lasx_hamming_load_partial(const uint8_t* p) {
    // Zero-pad without reading past the code.
    alignas(16) uint8_t bytes[16] = {};
    std::memcpy(bytes, p, N);
    return __lsx_vld(bytes, 0);
}

} // namespace detail

template <>
struct HammingComputer16_tpl<SIMDLevel::LOONGARCH_LASX> {
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
        result +=
                detail::lasx_hamming_distance<false>(data[0], __lsx_vld(b, 0));
        return result;
    }

    static constexpr int get_code_size() {
        return 16;
    }
};

template <>
struct HammingComputer20_tpl<SIMDLevel::LOONGARCH_LASX> {
    __m128i data[2];

    HammingComputer20_tpl() = default;

    HammingComputer20_tpl(const uint8_t* a, int size) {
        set(a, size);
    }

    void set(const uint8_t* a, FAISS_MAYBE_UNUSED int size) {
        assert(size == 20);
        data[0] = __lsx_vld(a, 0);
        data[1] = detail::lasx_hamming_load_partial<4>(a + 16);
    }

    int hamming(const uint8_t* b) const {
        int result = 0;
        result +=
                detail::lasx_hamming_distance<false>(data[0], __lsx_vld(b, 0));
        result += detail::lasx_hamming_distance<false>(
                data[1], detail::lasx_hamming_load_partial<4>(b + 16));
        return result;
    }

    static constexpr int get_code_size() {
        return 20;
    }
};

template <>
struct HammingComputer32_tpl<SIMDLevel::LOONGARCH_LASX> {
    __m256i data[1];

    HammingComputer32_tpl() = default;

    HammingComputer32_tpl(const uint8_t* a, int size) {
        set(a, size);
    }

    void set(const uint8_t* a, FAISS_MAYBE_UNUSED int size) {
        assert(size == 32);
        data[0] = __lasx_xvld(a, 0);
    }

    int hamming(const uint8_t* b) const {
        int result = 0;
        result += detail::lasx_hamming_distance<false>(
                data[0], __lasx_xvld(b, 0));
        return result;
    }

    static constexpr int get_code_size() {
        return 32;
    }
};

template <>
struct HammingComputer64_tpl<SIMDLevel::LOONGARCH_LASX> {
    __m256i data[2];

    HammingComputer64_tpl() = default;

    HammingComputer64_tpl(const uint8_t* a, int size) {
        set(a, size);
    }

    void set(const uint8_t* a, FAISS_MAYBE_UNUSED int size) {
        assert(size == 64);
        for (int i = 0; i < 2; ++i) {
            data[i] = __lasx_xvld(a + i * 32, 0);
        }
    }

    int hamming(const uint8_t* b) const {
        int result = 0;
        for (int i = 0; i < 2; ++i) {
            result += detail::lasx_hamming_distance<false>(
                    data[i], __lasx_xvld(b + i * 32, 0));
        }
        return result;
    }

    static constexpr int get_code_size() {
        return 64;
    }
};

template <>
struct GenHammingComputer8_tpl<SIMDLevel::LOONGARCH_LASX> {
    __m128i data[1];

    GenHammingComputer8_tpl() = default;

    GenHammingComputer8_tpl(const uint8_t* a, int size) {
        set(a, size);
    }

    void set(const uint8_t* a, FAISS_MAYBE_UNUSED int size) {
        assert(size == 8);
        data[0] = detail::lasx_hamming_load_partial<8>(a + 0);
    }

    int hamming(const uint8_t* b) const {
        int result = 0;
        result += detail::lasx_hamming_distance<true>(
                data[0], detail::lasx_hamming_load_partial<8>(b + 0));
        return result;
    }

    static constexpr int get_code_size() {
        return 8;
    }
};

template <>
struct GenHammingComputer16_tpl<SIMDLevel::LOONGARCH_LASX> {
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
        result += detail::lasx_hamming_distance<true>(data[0], __lsx_vld(b, 0));
        return result;
    }

    static constexpr int get_code_size() {
        return 16;
    }
};

template <>
struct GenHammingComputer32_tpl<SIMDLevel::LOONGARCH_LASX> {
    __m256i data[1];

    GenHammingComputer32_tpl() = default;

    GenHammingComputer32_tpl(const uint8_t* a, int size) {
        set(a, size);
    }

    void set(const uint8_t* a, FAISS_MAYBE_UNUSED int size) {
        assert(size == 32);
        data[0] = __lasx_xvld(a, 0);
    }

    int hamming(const uint8_t* b) const {
        int result = 0;
        result +=
                detail::lasx_hamming_distance<true>(data[0], __lasx_xvld(b, 0));
        return result;
    }

    static constexpr int get_code_size() {
        return 32;
    }
};

template <>
struct HammingComputerDefault_tpl<SIMDLevel::LOONGARCH_LASX> {
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
        for (; i + 32 <= size; i += 32) {
            result += detail::lasx_hamming_distance<false>(
                    __lasx_xvld(a + i, 0), __lasx_xvld(b + i, 0));
        }
        if (i + 16 <= size) {
            result += detail::lasx_hamming_distance<false>(
                    __lsx_vld(a + i, 0), __lsx_vld(b + i, 0));
            i += 16;
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
struct GenHammingComputerM8_tpl<SIMDLevel::LOONGARCH_LASX> {
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
        for (; i + 32 <= size; i += 32) {
            result += detail::lasx_hamming_distance<true>(
                    __lasx_xvld(a + i, 0), __lasx_xvld(b + i, 0));
        }
        if (i + 16 <= size) {
            result += detail::lasx_hamming_distance<true>(
                    __lsx_vld(a + i, 0), __lsx_vld(b + i, 0));
            i += 16;
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
