/**
 * Copyright (c) Facebook, Inc. and its affiliates.
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
#elif __AVX2__
// better versions for GenHammingComputer
#include <faiss/utils/hamming_distance/avx2-inl.h>
#else
#include <faiss/utils/hamming_distance/generic-inl.h>
#endif

#endif
