/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/impl/platform_macros.h>
#include <cstdint>

namespace faiss {
namespace cppcontrib {
namespace detail {

namespace {

template <intptr_t N_ELEMENTS, intptr_t CPOS>
struct Uint8Reader {
    static_assert(CPOS < N_ELEMENTS, "CPOS should be less than N_ELEMENTS");

    static intptr_t get(const uint8_t* const __restrict codes) {
        // Read using 4-bytes, if possible.
        // Reading using 8-byte takes too many registers somewhy.

        constexpr intptr_t ELEMENT_TO_READ = CPOS / 4;
        constexpr intptr_t SUB_ELEMENT = CPOS % 4;

        switch (SUB_ELEMENT) {
            case 0: {
                if (N_ELEMENTS > CPOS + 3) {
                    const uint32_t code32 = *reinterpret_cast<const uint32_t*>(
                            codes + ELEMENT_TO_READ * 4);
#ifdef FAISS_BIG_ENDIAN
                    return (code32) >> 24;
#else
                    return (code32 & 0x000000FF);
#endif
                } else {
                    return codes[CPOS];
                }
            }
            case 1: {
                if (N_ELEMENTS > CPOS + 2) {
                    const uint32_t code32 = *reinterpret_cast<const uint32_t*>(
                            codes + ELEMENT_TO_READ * 4);
#ifdef FAISS_BIG_ENDIAN
                    return (code32 & 0x00FF0000) >> 16;
#else
                    return (code32 & 0x0000FF00) >> 8;
#endif
                } else {
                    return codes[CPOS];
                }
            }
            case 2: {
                if (N_ELEMENTS > CPOS + 1) {
                    const uint32_t code32 = *reinterpret_cast<const uint32_t*>(
                            codes + ELEMENT_TO_READ * 4);
#ifdef FAISS_BIG_ENDIAN
                    return (code32 & 0x0000FF00) >> 8;
#else
                    return (code32 & 0x00FF0000) >> 16;
#endif
                } else {
                    return codes[CPOS];
                }
            }
            case 3: {
                if (N_ELEMENTS > CPOS) {
                    const uint32_t code32 = *reinterpret_cast<const uint32_t*>(
                            codes + ELEMENT_TO_READ * 4);
#ifdef FAISS_BIG_ENDIAN
                    return (code32 & 0x000000FF);
#else
                    return (code32) >> 24;
#endif
                } else {
                    return codes[CPOS];
                }
            }
        }
    }
};

// reduces the number of read operations from RAM
///////////////////////////////////////////////
// 76543210 76543210 76543210 76543210 76543210
// 00000000 00
//            111111 1111
//                       2222 222222
//                                  33 33333333
template <intptr_t N_ELEMENTS, intptr_t CPOS>
struct Uint10Reader {
    static_assert(CPOS < N_ELEMENTS, "CPOS should be less than N_ELEMENTS");

    static intptr_t get(const uint8_t* const __restrict codes) {
        // Read using 4-bytes or 2-bytes.

        constexpr intptr_t ELEMENT_TO_READ = CPOS / 4;
        constexpr intptr_t SUB_ELEMENT = CPOS % 4;

        switch (SUB_ELEMENT) {
            case 0: {
                if (N_ELEMENTS > CPOS + 2) {
                    uint32_t code32 = *reinterpret_cast<const uint32_t*>(
                            codes + ELEMENT_TO_READ * 5);
#ifdef FAISS_BIG_ENDIAN
                    code32 = Swap4Bytes(code32);
#endif
                    return (code32 & 0b0000001111111111);
                } else {
                    uint16_t code16 = *reinterpret_cast<const uint16_t*>(
                            codes + ELEMENT_TO_READ * 5 + 0);
#ifdef FAISS_BIG_ENDIAN
                    code16 = Swap2Bytes(code16);
#endif
                    return (code16 & 0b0000001111111111);
                }
            }
            case 1: {
                if (N_ELEMENTS > CPOS + 1) {
                    uint32_t code32 = *reinterpret_cast<const uint32_t*>(
                            codes + ELEMENT_TO_READ * 5);
#ifdef FAISS_BIG_ENDIAN
                    code32 = Swap4Bytes(code32);
#endif
                    return (code32 & 0b000011111111110000000000) >> 10;
                } else {
                    uint16_t code16 = *reinterpret_cast<const uint16_t*>(
                            codes + ELEMENT_TO_READ * 5 + 1);
#ifdef FAISS_BIG_ENDIAN
                    code16 = Swap2Bytes(code16);
#endif
                    return (code16 & 0b0000111111111100) >> 2;
                }
            }
            case 2: {
                if (N_ELEMENTS > CPOS) {
                    uint32_t code32 = *reinterpret_cast<const uint32_t*>(
                            codes + ELEMENT_TO_READ * 5);
#ifdef FAISS_BIG_ENDIAN
                    code32 = Swap4Bytes(code32);
#endif
                    return (code32 & 0b00111111111100000000000000000000) >> 20;
                } else {
                    uint16_t code16 = *reinterpret_cast<const uint16_t*>(
                            codes + ELEMENT_TO_READ * 5 + 2);
#ifdef FAISS_BIG_ENDIAN
                    code16 = Swap2Bytes(code16);
#endif
                    return (code16 & 0b0011111111110000) >> 4;
                }
            }
            case 3: {
                uint16_t code16 = *reinterpret_cast<const uint16_t*>(
                        codes + ELEMENT_TO_READ * 5 + 3);
#ifdef FAISS_BIG_ENDIAN
                code16 = Swap2Bytes(code16);
#endif
                return (code16 & 0b1111111111000000) >> 6;
            }
        }
    }
};

// reduces the number of read operations from RAM
///////////////////////////////////////////////
// 76543210 76543210 76543210 76543210 76543210 76543210
// 00000000 0000
//              1111 11111111
//                            22222222 2222
//                                         3333 33333333
template <intptr_t N_ELEMENTS, intptr_t CPOS>
struct Uint12Reader {
    static_assert(CPOS < N_ELEMENTS, "CPOS should be less than N_ELEMENTS");

    static intptr_t get(const uint8_t* const __restrict codes) {
        // Read using 4-bytes or 2-bytes.

        constexpr intptr_t ELEMENT_TO_READ = CPOS / 4;
        constexpr intptr_t SUB_ELEMENT = CPOS % 4;

        switch (SUB_ELEMENT) {
            case 0: {
                if (N_ELEMENTS > CPOS + 2) {
                    uint32_t code32 = *reinterpret_cast<const uint32_t*>(
                            codes + ELEMENT_TO_READ * 6);
#ifdef FAISS_BIG_ENDIAN
                    code32 = Swap4Bytes(code32);
#endif
                    return (code32 & 0b0000111111111111);
                } else {
                    uint16_t code16 = *reinterpret_cast<const uint16_t*>(
                            codes + ELEMENT_TO_READ * 6 + 0);
#ifdef FAISS_BIG_ENDIAN
                    code16 = Swap2Bytes(code16);
#endif
                    return (code16 & 0b0000111111111111);
                }
            }
            case 1: {
                if (N_ELEMENTS > CPOS + 1) {
                    uint32_t code32 = *reinterpret_cast<const uint32_t*>(
                            codes + ELEMENT_TO_READ * 6);
#ifdef FAISS_BIG_ENDIAN
                    code32 = Swap4Bytes(code32);
#endif
                    return (code32 & 0b111111111111000000000000) >> 12;
                } else {
                    uint16_t code16 = *reinterpret_cast<const uint16_t*>(
                            codes + ELEMENT_TO_READ * 6 + 1);
#ifdef FAISS_BIG_ENDIAN
                    code16 = Swap2Bytes(code16);
#endif
                    return (code16 & 0b1111111111110000) >> 4;
                }
            }
            case 2: {
                if (N_ELEMENTS > CPOS + 1) {
                    uint32_t code32 = *reinterpret_cast<const uint32_t*>(
                            codes + ELEMENT_TO_READ * 6 + 2);
#ifdef FAISS_BIG_ENDIAN
                    code32 = Swap4Bytes(code32);
#endif
                    return (code32 & 0b000011111111111100000000) >> 8;
                } else {
                    uint16_t code16 = *reinterpret_cast<const uint16_t*>(
                            codes + ELEMENT_TO_READ * 6 + 3);
#ifdef FAISS_BIG_ENDIAN
                    code16 = Swap2Bytes(code16);
#endif
                    return (code16 & 0b0000111111111111);
                }
            }
            case 3: {
                if (N_ELEMENTS > CPOS) {
                    uint32_t code32 = *reinterpret_cast<const uint32_t*>(
                            codes + ELEMENT_TO_READ * 6 + 2);
#ifdef FAISS_BIG_ENDIAN
                    code32 = Swap4Bytes(code32);
#endif
                    return (code32 & 0b11111111111100000000000000000000) >> 20;
                } else {
                    uint16_t code16 = *reinterpret_cast<const uint16_t*>(
                            codes + ELEMENT_TO_READ * 6 + 4);
#ifdef FAISS_BIG_ENDIAN
                    code16 = Swap2Bytes(code16);
#endif
                    return (code16 & 0b1111111111110000) >> 4;
                }
            }
        }
    }
};

// reduces the number of read operations from RAM
template <intptr_t N_ELEMENTS, intptr_t CPOS>
struct Uint16Reader {
    static_assert(CPOS < N_ELEMENTS, "CPOS should be less than N_ELEMENTS");

    static intptr_t get(const uint8_t* const __restrict codes) {
        // Read using 4-bytes or 2-bytes.
        // Reading using 8-byte takes too many registers somewhy.

        constexpr intptr_t ELEMENT_TO_READ = CPOS / 2;
        constexpr intptr_t SUB_ELEMENT = CPOS % 2;

        switch (SUB_ELEMENT) {
            case 0: {
                if (N_ELEMENTS > CPOS + 1) {
                    uint32_t code32 = *reinterpret_cast<const uint32_t*>(
                            codes + ELEMENT_TO_READ * 4);
#ifdef FAISS_BIG_ENDIAN
                    code32 = Swap4Bytes(code32);
#endif
                    return (code32 & 0x0000FFFF);
                } else {
                    const uint16_t* const __restrict codesFp16 =
                            reinterpret_cast<const uint16_t*>(codes);
#ifdef FAISS_BIG_ENDIAN
                    uint16_t rt = codesFp16[CPOS];
                    rt = Swap2Bytes(rt);
                    return rt;
#endif
                    return codesFp16[CPOS];
                }
            }
            case 1: {
                if (N_ELEMENTS > CPOS) {
                    uint32_t code32 = *reinterpret_cast<const uint32_t*>(
                            codes + ELEMENT_TO_READ * 4);
#ifdef FAISS_BIG_ENDIAN
                    code32 = Swap4Bytes(code32);
#endif
                    return code32 >> 16;
                } else {
                    const uint16_t* const __restrict codesFp16 =
                            reinterpret_cast<const uint16_t*>(codes);
#ifdef FAISS_BIG_ENDIAN
                    uint16_t rt = codesFp16[CPOS];
                    rt = Swap2Bytes(rt);
                    return rt;
#endif
                    return codesFp16[CPOS];
                }
            }
        }
    }
};

//
template <intptr_t N_ELEMENTS, intptr_t CODE_BITS, intptr_t CPOS>
struct UintReaderImplType {};

template <intptr_t N_ELEMENTS, intptr_t CPOS>
struct UintReaderImplType<N_ELEMENTS, 8, CPOS> {
    using reader_type = Uint8Reader<N_ELEMENTS, CPOS>;
};

template <intptr_t N_ELEMENTS, intptr_t CPOS>
struct UintReaderImplType<N_ELEMENTS, 10, CPOS> {
    using reader_type = Uint10Reader<N_ELEMENTS, CPOS>;
};

template <intptr_t N_ELEMENTS, intptr_t CPOS>
struct UintReaderImplType<N_ELEMENTS, 12, CPOS> {
    using reader_type = Uint12Reader<N_ELEMENTS, CPOS>;
};

template <intptr_t N_ELEMENTS, intptr_t CPOS>
struct UintReaderImplType<N_ELEMENTS, 16, CPOS> {
    using reader_type = Uint16Reader<N_ELEMENTS, CPOS>;
};

} // namespace

// reduces the number of read operations from RAM
template <intptr_t DIM, intptr_t CODE_SIZE, intptr_t CODE_BITS, intptr_t CPOS>
using UintReader =
        typename UintReaderImplType<DIM / CODE_SIZE, CODE_BITS, CPOS>::
                reader_type;

template <intptr_t N_ELEMENTS, intptr_t CODE_BITS, intptr_t CPOS>
using UintReaderRaw =
        typename UintReaderImplType<N_ELEMENTS, CODE_BITS, CPOS>::reader_type;

} // namespace detail
} // namespace cppcontrib
} // namespace faiss
