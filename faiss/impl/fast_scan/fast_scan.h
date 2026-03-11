/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <cstdlib>
#include <memory>

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

struct IDSelector;
struct RangeSearchResult;
struct RangeSearchPartialResult;
struct SIMDResultHandler;
struct SIMDResultHandlerToFloat;

/** Pack codes for consumption by the SIMD kernels.
 *  The unused bytes are set to 0.
 *
 * @param codes   input codes, size (ntotal, ceil(M / 2))
 * @param ntotal  number of input codes
 * @param nb      output number of codes (ntotal rounded up to a multiple of
 *                bbs)
 * @param nsq      number of sub-quantizers (=M rounded up to a multiple of 2)
 * @param bbs     size of database blocks (multiple of 32)
 * @param blocks  output array, size nb * nsq / 2.
 * @param code_stride  optional stride between consecutive codes (0 = use
default (M + 1) / 2)
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
 * @param code_stride  optional stride between consecutive codes (0 = use
 * default (M + 1) / 2)
 * @param block_stride  stride in bytes between consecutive blocks.
 */
void pq4_pack_codes_range(
        const uint8_t* codes,
        size_t M,
        size_t i0,
        size_t i1,
        size_t bbs,
        size_t nsq,
        uint8_t* blocks,
        size_t code_stride,
        size_t block_stride);

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

    CodePacker* clone() const final;

    void pack_1(const uint8_t* flat_code, size_t offset, uint8_t* block)
            const final;
    void unpack_1(const uint8_t* block, size_t offset, uint8_t* flat_code)
            const final;
};

/** Pack Look-up table for consumption by the kernel.
 *
 * @param nq      number of queries
 * @param nsq     number of sub-quantizers (multiple of 2)
 * @param src     input array, size (nq, 16)
 * @param dest    output array, size (nq, 16)
 */
void pq4_pack_LUT(int nq, int nsq, const uint8_t* src, uint8_t* dest);

/** Loop over database elements and accumulate results into result handler
 *
 * @param nq      number of queries
 * @param nb      number of database elements
 * @param bbs     size of database blocks (multiple of 32)
 * @param nsq     number of sub-quantizers (multiple of 2)
 * @param codes   packed codes array
 * @param LUT     packed look-up table
 * @param scaler  scaler to scale the encoded norm
 * @param block_stride  stride in bytes between consecutive blocks.
 */
void pq4_accumulate_loop(
        int nq,
        size_t nb,
        int bbs,
        int nsq,
        const uint8_t* codes,
        const uint8_t* LUT,
        SIMDResultHandler& res,
        int pq2x4_scale,
        size_t block_stride);

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
 * @param nsq     number of sub-quantizers (multiple of 2)
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
 * @param nb      number of database codes (multiple of bbs)
 * @param nsq     number of sub-quantizers
 * @param codes   encoded database vectors (packed)
 * @param LUT     look-up table (packed)
 * @param res     call-back for the results
 * @param pq2x4_scale  scaler to scale the encoded norm
 * @param block_stride  stride in bytes between consecutive blocks.
 */
void pq4_accumulate_loop_qbs(
        int qbs,
        size_t nb,
        int nsq,
        const uint8_t* codes,
        const uint8_t* LUT,
        SIMDResultHandler& res,
        int pq2x4_scale,
        size_t block_stride);

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

/***************************************************************
 * FastScanCodeScanner: virtual base that bundles handler + kernel
 * behind the SIMD dispatch boundary. Per-SIMD TUs instantiate this
 * with the correct SIMDLevel so that handler and kernel share the
 * same SIMD types.
 ***************************************************************/

struct FastScanCodeScanner {
    virtual ~FastScanCodeScanner() = default;

    /// Access the underlying result handler (for begin/end/normalizer calls)
    virtual SIMDResultHandlerToFloat* handler() = 0;

    /// Run the search_1 accumulation loop (bbs > 32, multi-BB kernel)
    virtual void accumulate_loop(
            int nq,
            size_t nb,
            int bbs,
            int nsq,
            const uint8_t* codes,
            const uint8_t* LUT,
            int pq2x4_scale,
            size_t block_stride) = 0;

    /// Run the QBS accumulation loop (bbs == 32)
    virtual void accumulate_loop_qbs(
            int qbs,
            size_t nb,
            int nsq,
            const uint8_t* codes,
            const uint8_t* LUT,
            int pq2x4_scale,
            size_t block_stride) = 0;
};

/// Per-SIMD factory: explicitly specialized in each per-SIMD TU
/// (impl-avx2.cpp, impl-avx512.cpp, impl-neon.cpp, fast_scan.cpp for NONE).
/// Not called directly — use make_fast_scan_knn_scanner() instead.
template <SIMDLevel SL>
std::unique_ptr<FastScanCodeScanner> make_fast_scan_scanner_impl(
        bool is_max,
        int impl,
        size_t nq,
        size_t ntotal,
        int64_t k,
        float* distances,
        int64_t* ids,
        const IDSelector* sel,
        bool with_id_map);

/// Runtime dispatch wrapper: selects the best available SIMD level
/// (via DISPATCH_SIMDLevel) and delegates to the corresponding
/// make_fast_scan_scanner_impl<SL> specialization.
std::unique_ptr<FastScanCodeScanner> make_fast_scan_knn_scanner(
        bool is_max,
        int impl,
        size_t nq,
        size_t ntotal,
        int64_t k,
        float* distances,
        int64_t* ids,
        const IDSelector* sel,
        bool with_id_map = false);

/// Per-SIMD range scanner factories (defined in per-SIMD TUs via dispatching.h)
template <SIMDLevel SL>
std::unique_ptr<FastScanCodeScanner> make_range_scanner_impl(
        bool is_max,
        RangeSearchResult& rres,
        float radius,
        size_t ntotal,
        const IDSelector* sel);

template <SIMDLevel SL>
std::unique_ptr<FastScanCodeScanner> make_partial_range_scanner_impl(
        bool is_max,
        RangeSearchPartialResult& pres,
        float radius,
        size_t ntotal,
        size_t q0,
        size_t q1,
        const IDSelector* sel);

/// Runtime dispatch: range search scanner.
std::unique_ptr<FastScanCodeScanner> make_range_scanner(
        bool is_max,
        RangeSearchResult& rres,
        float radius,
        size_t ntotal,
        const IDSelector* sel);

/// Runtime dispatch: partial range search scanner (per-thread).
std::unique_ptr<FastScanCodeScanner> make_partial_range_scanner(
        bool is_max,
        RangeSearchPartialResult& pres,
        float radius,
        size_t ntotal,
        size_t q0,
        size_t q1,
        const IDSelector* sel);

/***************************************************************
 * RaBitQ scanner factory: per-SIMD specializations live in
 * rabitq_dispatching.h, included by each per-SIMD TU.
 ***************************************************************/

struct IndexRaBitQFastScan;
struct IndexIVFRaBitQFastScan;
struct FastScanDistancePostProcessing;

/// Per-SIMD factory (primary template; specializations in rabitq_dispatching.h)
template <SIMDLevel SL>
std::unique_ptr<FastScanCodeScanner> rabitq_make_knn_scanner_impl(
        const IndexRaBitQFastScan* index,
        bool is_max,
        size_t nq,
        int64_t k,
        float* distances,
        int64_t* ids,
        const IDSelector* sel,
        const FastScanDistancePostProcessing& context,
        bool is_multi_bit);

/// Runtime dispatch wrapper for rabitq_make_knn_scanner_impl
std::unique_ptr<FastScanCodeScanner> rabitq_make_knn_scanner(
        const IndexRaBitQFastScan* index,
        bool is_max,
        size_t nq,
        int64_t k,
        float* distances,
        int64_t* ids,
        const IDSelector* sel,
        const FastScanDistancePostProcessing& context,
        bool is_multi_bit);

/// Per-SIMD IVF RaBitQ scanner factory.
template <SIMDLevel SL>
std::unique_ptr<FastScanCodeScanner> rabitq_ivf_make_knn_scanner_impl(
        bool is_max,
        const IndexIVFRaBitQFastScan* index,
        size_t nq,
        size_t k,
        float* distances,
        int64_t* ids,
        const FastScanDistancePostProcessing* context,
        bool multi_bit);

/// Runtime dispatch wrapper for IVF RaBitQ scanner.
std::unique_ptr<FastScanCodeScanner> rabitq_ivf_make_knn_scanner(
        bool is_max,
        const IndexIVFRaBitQFastScan* index,
        size_t nq,
        size_t k,
        float* distances,
        int64_t* ids,
        const FastScanDistancePostProcessing* context,
        bool multi_bit);

} // namespace faiss
