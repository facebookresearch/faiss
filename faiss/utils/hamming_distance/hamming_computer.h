/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// This file contains forward declarations, architecture-independent
// HammingComputer structs (sizes 4 and 8), and the with_HammingComputer
// dispatch function. SIMDLevel-specific specializations live in:
//   hamming_computer-generic.h  (NONE — scalar fallback)
//   hamming_computer-avx2.h     (AVX2)
//   hamming_computer-avx512.h   (AVX512)
//   hamming_computer-neon.h     (ARM NEON)

#ifndef FAISS_hamming_computer_h
#define FAISS_hamming_computer_h

#include <faiss/utils/hamming_distance/common.h>

namespace faiss {

/***************************************************************************
 * HammingComputer primary templates.
 *
 * Per-ISA backend files (hamming_computer-avx512.h, hamming_computer-neon.h,
 * etc.) provide explicit specializations that override the scalar (NONE)
 * defaults in hamming_computer-generic.h with ISA-optimized code.
 * Templating on SIMDLevel gives each specialization a distinct mangled
 * name, so DD builds with multiple per-ISA TUs do NOT create ODR-violating
 * struct collisions.
 *
 * Call sites use with_HammingComputer<SL>, which is templatized on
 * SIMDLevel to select the matching specialization.
 ***************************************************************************/

// Forward declarations. The struct bodies live in hamming_computer-generic.h
// (NONE) and per-ISA hamming_computer-*.h files.
template <SIMDLevel SL>
struct HammingComputer16_tpl;
template <SIMDLevel SL>
struct HammingComputer20_tpl;
template <SIMDLevel SL>
struct HammingComputer32_tpl;
template <SIMDLevel SL>
struct HammingComputer64_tpl;
template <SIMDLevel SL>
struct HammingComputerDefault_tpl;
template <SIMDLevel SL>
struct GenHammingComputer8_tpl;
template <SIMDLevel SL>
struct GenHammingComputer16_tpl;
template <SIMDLevel SL>
struct GenHammingComputer32_tpl;
template <SIMDLevel SL>
struct GenHammingComputerM8_tpl;

/******************************************************************
 * The HammingComputer series of classes compares a single code of
 * size 4 to 32 to incoming codes. They are intended for use as a
 * template class where it would be inefficient to switch on the code
 * size in the inner loop. Hopefully the compiler will inline the
 * hamming() functions and put the a0, a1, ... in registers.
 * For code_size = 4 and 8 we don't use SIMD implementations, because
 * register widths are too large.
 ******************************************************************/

struct HammingComputer4 {
    uint32_t a0;

    HammingComputer4() {}

    HammingComputer4(const uint8_t* a, int code_size) {
        set(a, code_size);
    }

    void set(const uint8_t* a, FAISS_MAYBE_UNUSED int code_size) {
        assert(code_size == 4);
        const uint32_t* a32 = reinterpret_cast<const uint32_t*>(a);
        a0 = *a32;
    }

    inline int hamming(const uint8_t* b) const {
        const uint32_t* b32 = reinterpret_cast<const uint32_t*>(b);
        return popcount64(*b32 ^ a0);
    }

    inline static constexpr int get_code_size() {
        return 4;
    }
};

struct HammingComputer8 {
    uint64_t a0;

    HammingComputer8() {}

    HammingComputer8(const uint8_t* a, int code_size) {
        set(a, code_size);
    }

    void set(const uint8_t* a, FAISS_MAYBE_UNUSED int code_size) {
        assert(code_size == 8);
        const uint64_t* a64 = reinterpret_cast<const uint64_t*>(a);
        a0 = *a64;
    }

    inline int hamming(const uint8_t* b) const {
        const uint64_t* b64 = reinterpret_cast<const uint64_t*>(b);
        return popcount64(*b64 ^ a0);
    }

    inline static constexpr int get_code_size() {
        return 8;
    }
};

/***************************************************************************
 * Dispatching function that takes a code size and a C++20 template lambda.
 * The lambda is called with the appropriate HammingComputer type:
 *   with_HammingComputer<SL>(code_size, [&]<class HammingComputer>() { ... });
 **************************************************************************/

template <SIMDLevel SL, class F>
decltype(auto) with_HammingComputer(int code_size, F&& f) {
    switch (code_size) {
        case 4:
            return f.template operator()<HammingComputer4>();
        case 8:
            return f.template operator()<HammingComputer8>();
        case 16:
            return f.template operator()<HammingComputer16_tpl<SL>>();
        case 20:
            return f.template operator()<HammingComputer20_tpl<SL>>();
        case 32:
            return f.template operator()<HammingComputer32_tpl<SL>>();
        case 64:
            return f.template operator()<HammingComputer64_tpl<SL>>();
        default:
            return f.template operator()<HammingComputerDefault_tpl<SL>>();
    }
}

} // namespace faiss

#endif
