/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// This file contains low level inline facilities for computing
// Hamming distances, such as HammingComputerXX and GenHammingComputerXX.

#ifndef FAISS_hamming_inl_h
#define FAISS_hamming_inl_h

#include <faiss/utils/hamming_distance/common.h>

#ifdef __aarch64__
// ARM compilers may produce inoptimal code for Hamming distance somewhy.
#include <faiss/utils/hamming_distance/neon-inl.h>
#elif __AVX512F__
// offers better performance where __AVX512VPOPCNTDQ__ is supported
#include <faiss/utils/hamming_distance/avx512-inl.h>
#elif __AVX2__
// better versions for GenHammingComputer
#include <faiss/utils/hamming_distance/avx2-inl.h>
#else
#include <faiss/utils/hamming_distance/generic-inl.h>
#endif

namespace faiss {

/***************************************************************************
 * Equivalence with a template class when code size is known at compile time
 **************************************************************************/

// default template
template <int CODE_SIZE>
struct HammingComputer : HammingComputerDefault {
    HammingComputer(const uint8_t* a, int code_size)
            : HammingComputerDefault(a, code_size) {}
};

#define SPECIALIZED_HC(CODE_SIZE)                                    \
    template <>                                                      \
    struct HammingComputer<CODE_SIZE> : HammingComputer##CODE_SIZE { \
        HammingComputer(const uint8_t* a)                            \
                : HammingComputer##CODE_SIZE(a, CODE_SIZE) {}        \
    }

SPECIALIZED_HC(4);
SPECIALIZED_HC(8);
SPECIALIZED_HC(16);
SPECIALIZED_HC(20);
SPECIALIZED_HC(32);
SPECIALIZED_HC(64);

#undef SPECIALIZED_HC

/***************************************************************************
 * Dispatching function that takes a code size and a consumer object
 * the consumer object should contain a retun type t and a operation template
 * function f() that must be called to perform the operation.
 **************************************************************************/

template <class Consumer, class... Types>
typename Consumer::T dispatch_HammingComputer(
        int code_size,
        Consumer& consumer,
        Types... args) {
    switch (code_size) {
#define DISPATCH_HC(CODE_SIZE) \
    case CODE_SIZE:            \
        return consumer.template f<HammingComputer##CODE_SIZE>(args...);
        DISPATCH_HC(4);
        DISPATCH_HC(8);
        DISPATCH_HC(16);
        DISPATCH_HC(20);
        DISPATCH_HC(32);
        DISPATCH_HC(64);
        default:
            return consumer.template f<HammingComputerDefault>(args...);
    }
#undef DISPATCH_HC
}

} // namespace faiss

#endif
