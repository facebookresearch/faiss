/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <cstdlib>

/** PQ4 SIMD packing and accumulation functions
 *
 * The basic kernel accumulates nq query vectors with bbs = nb * 2 * 16 vectors
 * and produces an output matrix for that. It is interesting for nq * nb <= 4,
 * otherwise register spilling becomes too large.
 *
 * The implementation of these functions is spread over 3 cpp files to reduce
 * parallel compile times. Templates are instanciated explicitly.
 */

namespace faiss {

/** Pack codes for consumption by the SIMD kernels.
 *  The unused bytes are set to 0.
 *
 * @param codes   input codes, size (ntotal, ceil(M / 2))
 * @param ntotal  number of input codes
 * @param nb      output number of codes (ntotal rounded up to a multiple of
 *                bbs)
 * @param M2      number of sub-quantizers (=M rounded up to a muliple of 2)
 * @param bbs     size of database blocks (multiple of 32)
 * @param blocks  output array, size nb * nsq / 2.
 */
void pq4_pack_codes(
        const uint8_t* codes,
        size_t ntotal,
        size_t M,
        size_t nb,
        size_t bbs,
        size_t M2,
        uint8_t* blocks);

/** Same as pack_codes but write in a given range of the output,
 * leaving the rest untouched. Assumes allocated entries are 0 on input.
 *
 * @param codes   input codes, size (i1 - i0, ceil(M / 2))
 * @param i0      first output code to write
 * @param i1      last output code to write
 * @param blocks  output array, size at least ceil(i1 / bbs) * bbs * nsq / 2
 */
void pq4_pack_codes_range(
        const uint8_t* codes,
        size_t M,
        size_t i0,
        size_t i1,
        size_t bbs,
        size_t M2,
        uint8_t* blocks);

/** get a single element from a packed codes table
 *
 * @param i        vector id
 * @param sq       subquantizer (< nsq)
 */
uint8_t pq4_get_packed_element(
        const uint8_t* data,
        size_t bbs,
        size_t nsq,
        size_t i,
        size_t sq);

/** Pack Look-up table for consumption by the kernel.
 *
 * @param nq      number of queries
 * @param nsq     number of sub-quantizers (muliple of 2)
 * @param src     input array, size (nq, 16)
 * @param dest    output array, size (nq, 16)
 */
void pq4_pack_LUT(int nq, int nsq, const uint8_t* src, uint8_t* dest);

/** Loop over database elements and accumulate results into result handler
 *
 * @param nq      number of queries
 * @param nb      number of database elements
 * @param bbs     size of database blocks (multiple of 32)
 * @param nsq     number of sub-quantizers (muliple of 2)
 * @param codes   packed codes array
 * @param LUT     packed look-up table
 * @param scaler  scaler to scale the encoded norm
 */
template <class ResultHandler, class Scaler>
void pq4_accumulate_loop(
        int nq,
        size_t nb,
        int bbs,
        int nsq,
        const uint8_t* codes,
        const uint8_t* LUT,
        ResultHandler& res,
        const Scaler& scaler);

/* qbs versions, supported only for bbs=32.
 *
 * The kernel function runs the kernel for *several* query blocks
 * and bbs database vectors. The sizes of the blocks are encoded in qbs as
 * base-16 digits.
 *
 * For example, qbs = 0x1223 means that the kernel will be run 4 times, the
 * first time with 3 query vectors, second time with 2 query vectors, then 2
 * vectors again and finally with 1 query vector. The output block will thus be
 * nq = 3 + 2 + 2 + 1 = 6 queries. For a given total block size, the optimal
 * decomposition into sub-blocks (measured empirically) is given by
 * preferred_qbs().
 */

/* compute the number of queries from a base-16 decomposition */
int pq4_qbs_to_nq(int qbs);

/** return the preferred decomposition in blocks for a nb of queries. */
int pq4_preferred_qbs(int nq);

/** Pack Look-up table for consumption by the kernel.
 *
 * @param qbs     4-bit encoded number of query blocks, the total number of
 *                queries handled (nq) is deduced from it
 * @param nsq     number of sub-quantizers (muliple of 2)
 * @param src     input array, size (nq, 16)
 * @param dest    output array, size (nq, 16)
 * @return nq
 */
int pq4_pack_LUT_qbs(int fqbs, int nsq, const uint8_t* src, uint8_t* dest);

/** Same as pq4_pack_LUT_qbs, except the source vectors are remapped with q_map
 */
int pq4_pack_LUT_qbs_q_map(
        int qbs,
        int nsq,
        const uint8_t* src,
        const int* q_map,
        uint8_t* dest);

/** Run accumulation loop.
 *
 * @param qbs     4-bit encoded number of queries
 * @param nb      number of database codes (mutliple of bbs)
 * @param nsq     number of sub-quantizers
 * @param codes   encoded database vectors (packed)
 * @param LUT     look-up table (packed)
 * @param res     call-back for the resutls
 * @param scaler  scaler to scale the encoded norm
 */
template <class ResultHandler, class Scaler>
void pq4_accumulate_loop_qbs(
        int qbs,
        size_t nb,
        int nsq,
        const uint8_t* codes,
        const uint8_t* LUT,
        ResultHandler& res,
        const Scaler& scaler);

} // namespace faiss
