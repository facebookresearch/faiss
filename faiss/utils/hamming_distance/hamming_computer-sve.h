/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef HAMMING_COMPUTER_SVE_H
#define HAMMING_COMPUTER_SVE_H

#ifdef __aarch64__

#include <faiss/utils/hamming_distance/hamming_computer-generic.h>
#include <faiss/utils/hamming_distance/hamming_computer-neon.h>

namespace faiss {

/***************************************************************************
 * ARM_SVE specializations inheriting the NEON kernels.
 ***************************************************************************/

#define FAISS_INHERIT_HAMMING_SVE(Class)                                  \
    template <>                                                           \
    struct Class##                                                        \
            _tpl<SIMDLevel::ARM_SVE> : Class##_tpl<SIMDLevel::ARM_NEON> { \
        using Class##_tpl<SIMDLevel::ARM_NEON>::Class##_tpl;              \
    }

FAISS_INHERIT_HAMMING_SVE(HammingComputer16);
FAISS_INHERIT_HAMMING_SVE(HammingComputer20);
FAISS_INHERIT_HAMMING_SVE(HammingComputer32);
FAISS_INHERIT_HAMMING_SVE(HammingComputer64);
FAISS_INHERIT_HAMMING_SVE(HammingComputerDefault);
FAISS_INHERIT_HAMMING_SVE(GenHammingComputer8);
FAISS_INHERIT_HAMMING_SVE(GenHammingComputer16);
FAISS_INHERIT_HAMMING_SVE(GenHammingComputer32);
FAISS_INHERIT_HAMMING_SVE(GenHammingComputerM8);

#undef FAISS_INHERIT_HAMMING_SVE

} // namespace faiss

#endif

#endif
