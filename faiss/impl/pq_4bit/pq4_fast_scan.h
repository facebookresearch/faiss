/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <cstdlib>
#include <vector>

#include <faiss/impl/CodePacker.h>
#include <faiss/utils/simd_levels.h>

/** PQ4 SIMD packing and accumulation functions
 *
 * The basic kernel accumulates nq query vectors with bbs = nb * 2 * 16 vectors
 * and produces an output matrix for that. It is interesting for nq * nb <= 4,
 * otherwise register spilling becomes too large.
 *
 * The implementation of these functions is spread over 3 cpp files to reduce
 * parallel compile times. Templates are instantiated explicitly.
 */

namespace faiss {

/* Result handler that will return float resutls eventually */
struct PQ4CodeScanner {
    size_t nq;     // number of queries
    size_t ntotal; // ignore excess elements after ntotal

    bool disable = false; // for benchmarking
    int norm_scale = -1;  // do the codes include 2x4 bits of scale?

    /// these fields are used for the IVF variants (with_id_map=true)
    const idx_t* id_map = nullptr; // map offset in invlist to vector id
    const int* q_map = nullptr;    // map q to global query
    const uint16_t* dbias =
            nullptr; // table of biases to add to each query (for IVF L2 search)
    const float* normalizers = nullptr; // size 2 * nq, to convert to float

    PQ4CodeScanner(size_t nq, size_t ntotal) : nq(nq), ntotal(ntotal) {}

    virtual void accumulate_loop(
            int nq,
            size_t nb,
            int bbs,
            int nsq,
            const uint8_t* codes,
            const uint8_t* LUT) = 0;

    virtual void accumulate_loop_qbs(
            int qbs,
            size_t nb,
            int nsq,
            const uint8_t* codes,
            const uint8_t* LUT) = 0;

    virtual void begin(const float* norms) {
        normalizers = norms;
    }

    // called at end of search to convert int16 distances to float, before
    // normalizers are deallocated
    virtual void end() {
        normalizers = nullptr;
    }

    /// For IVF handlers: set the current list context for factor lookup.
    /// Default implementation does nothing.
    virtual void set_list_context(
            size_t /* list_no */,
            const std::vector<int>& /* probe_map */) {}

    /// Return the number of heap updates performed.
    /// Default implementation returns 0.
    virtual size_t num_updates() const {
        return 0;
    }

    virtual ~PQ4CodeScanner() {}
};

/** Pack codes for consumption by the SIMD kernels.
 *  The unused bytes are set to 0.
 *
 * @param codes   input codes, size (ntotal, ceil(M / 2))
 * @param ntotal  number of input codes
 * @param nb      output number of codes (ntotal rounded up to a multiple of
 *                bbs)
 * @param nsq      number of sub-quantizers (=M rounded up to a muliple of 2)
 * @param bbs     size of database blocks (multiple of 32)
 * @param blocks  output array, size nb * nsq / 2.
 * @param code_stride  stride between consecutive input codes in bytes
 *                     (0 = use default stride of (M+1)/2)
 */
void pq4_pack_codes(
        const uint8_t* codes,
        size_t ntotal,
        size_t M,
        size_t nb,
        size_t bbs,
        size_t nsq,
        uint8_t* blocks,
        size_t code_stride = 0);

/** Same as pack_codes but write in a given range of the output,
 * leaving the rest untouched. Assumes allocated entries are 0 on input.
 *
 * @param codes   input codes, size (i1 - i0, ceil(M / 2))
 * @param i0      first output code to write
 * @param i1      last output code to write
 * @param blocks  output array, size at least ceil(i1 / bbs) * bbs * nsq / 2
 * @param code_stride  stride between consecutive input codes in bytes
 *                     (0 = use default stride of (M+1)/2)
 */
void pq4_pack_codes_range(
        const uint8_t* codes,
        size_t M,
        size_t i0,
        size_t i1,
        size_t bbs,
        size_t nsq,
        uint8_t* blocks,
        size_t code_stride = 0);

/** get a single element from a packed codes table
 *
 * @param vector_id        vector id
 * @param sq       subquantizer (< nsq)
 */
uint8_t pq4_get_packed_element(
        const uint8_t* data,
        size_t bbs,
        size_t nsq,
        size_t vector_id,
        size_t sq);

/** set a single element "code" into a packed codes table
 *
 * @param vector_id       vector id
 * @param sq       subquantizer (< nsq)
 */
void pq4_set_packed_element(
        uint8_t* data,
        uint8_t code,
        size_t bbs,
        size_t nsq,
        size_t vector_id,
        size_t sq);

/** CodePacker API for the PQ4 fast-scan */
struct CodePackerPQ4 : CodePacker {
    size_t nsq;

    CodePackerPQ4(size_t nsq, size_t bbs);

    void pack_1(const uint8_t* flat_code, size_t offset, uint8_t* block)
            const final;
    void unpack_1(const uint8_t* block, size_t offset, uint8_t* flat_code)
            const final;
};

/** Pack Look-up table for consumption by the kernel.
 *
 * @param nq      number of queries
 * @param nsq     number of sub-quantizers (muliple of 2)
 * @param src     input array, size (nq, 16)
 * @param dest    output array, size (nq, 16)
 */
void pq4_pack_LUT(int nq, int nsq, const uint8_t* src, uint8_t* dest);

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

/** Wrapper of pq4_accumulate_loop_qbs using simple StoreResultHandler
 *  and DummyScaler
 *
 * @param nq      number of queries
 * @param ntotal2 number of database elements (multiple of 32)
 * @param nsq     number of sub-quantizers (muliple of 2)
 * @param codes   packed codes array
 * @param LUT     packed look-up table
 * @param accu    array to store the results
 */
void accumulate_to_mem(
        int nq,
        size_t ntotal2,
        int nsq,
        const uint8_t* codes,
        const uint8_t* LUT,
        uint16_t* accu);

PQ4CodeScanner* pq4_make_flat_knn_handler(
        bool is_max,
        bool use_reservoir,
        idx_t nq,
        idx_t k,
        idx_t ntotal,
        float* distances,
        idx_t* labels,
        int norm_scale,
        const float* normalizers = nullptr,
        bool disable = false);

struct IDSelector;

PQ4CodeScanner* pq4_make_ivf_knn_handler(
        bool is_max,
        bool use_reservoir,
        idx_t nq,
        idx_t k,
        float* distances,
        idx_t* labels,
        int norm_scale,
        const IDSelector* sel);

struct RangeSearchResult;

PQ4CodeScanner* pq4_make_ivf_range_handler(
        bool is_max,
        RangeSearchResult& rres,
        float radius,
        int norm_scale,
        const IDSelector* sel);

struct RangeSearchPartialResult;

PQ4CodeScanner* pq4_make_ivf_partial_range_handler(
        bool is_max,
        RangeSearchPartialResult& pres,
        float radius,
        idx_t i0,
        idx_t i1,
        int norm_scale,
        const IDSelector* sel);

/// Type alias for SIMD result handlers that output float distances.
/// PQ4CodeScanner is the polymorphic base class for all SIMD result handlers.
/// This alias is used in IndexFastScan for the make_knn_handler virtual method.
using SIMDResultHandlerToFloat = PQ4CodeScanner;

} // namespace faiss
