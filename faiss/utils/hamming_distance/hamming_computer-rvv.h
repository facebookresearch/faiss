/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef HAMMING_COMPUTER_RVV_H
#define HAMMING_COMPUTER_RVV_H

// RVV HammingComputer fallbacks. There is no RVV-optimized HammingComputer
// implementation yet, so provide concrete RISCV_RVV specializations backed by
// the scalar NONE implementations.

#include <faiss/utils/hamming_distance/hamming_computer-generic.h>

namespace faiss {

#define FAISS_INHERIT_HAMMING_RVV(Class)                                      \
    template <>                                                               \
    struct Class##_tpl<SIMDLevel::RISCV_RVV> : Class##_tpl<SIMDLevel::NONE> { \
        using Class##_tpl<SIMDLevel::NONE>::Class##_tpl;                      \
    }

FAISS_INHERIT_HAMMING_RVV(HammingComputer16);
FAISS_INHERIT_HAMMING_RVV(HammingComputer20);
FAISS_INHERIT_HAMMING_RVV(HammingComputer32);
FAISS_INHERIT_HAMMING_RVV(HammingComputer64);
FAISS_INHERIT_HAMMING_RVV(HammingComputerDefault);
FAISS_INHERIT_HAMMING_RVV(GenHammingComputer8);
FAISS_INHERIT_HAMMING_RVV(GenHammingComputer16);
FAISS_INHERIT_HAMMING_RVV(GenHammingComputer32);
FAISS_INHERIT_HAMMING_RVV(GenHammingComputerM8);

#undef FAISS_INHERIT_HAMMING_RVV

} // namespace faiss

#endif
