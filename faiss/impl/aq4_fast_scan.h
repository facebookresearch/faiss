/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <cstdlib>

/** AQ4 SIMD accumulation functions
 *
 * The basic kernel accumulates nq query vectors with bbs = nb * 2 * 16 vectors
 * and produces an output matrix for that. It is interesting for nq * nb <= 4,
 * otherwise register spilling becomes too large.
 *
 */

namespace faiss {

/** Loop over database elements and accumulate results into result handler
 *
 * @param nq      number of queries
 * @param nb      number of database elements
 * @param bbs     size of database blocks (multiple of 32)
 * @param nsq     number of sub-quantizers (muliple of 2)
 * @param nscale  number of look-up tables to be scaled
 * @param scale   scale factor
 * @param codes   packed codes array
 * @param LUT     packed look-up table
 */
template <class ResultHandler>
void aq4_accumulate_loop(
        int nq,
        size_t nb,
        int bbs,
        int nsq,
        int nscale,
        int scale,
        const uint8_t* codes,
        const uint8_t* LUT,
        ResultHandler& res);

/** Run accumulation loop.
 *
 * @param qbs     4-bit encded number of queries
 * @param nb      number of database codes (mutliple of bbs)
 * @param nsq     number of sub-quantizers
 * @param nscale  number of look-up tables to be scaled
 * @param scale   scale factor
 * @param codes   encoded database vectors (packed)
 * @param LUT     look-up table (packed)
 * @param res     call-back for the resutls
 */
template <class ResultHandler>
void aq4_accumulate_loop_qbs(
        int qbs,
        size_t nb,
        int nsq,
        int nscale,
        int scale,
        const uint8_t* codes,
        const uint8_t* LUT,
        ResultHandler& res);

} // namespace faiss
