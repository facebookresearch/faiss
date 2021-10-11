/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <vector>

#include <faiss/Index.h>
#include <faiss/IndexFlat.h>

namespace faiss {

/** Abstract structure for additive quantizers
 *
 * Different from the product quantizer in which the decoded vector is the
 * concatenation of M sub-vectors, additive quantizers sum M sub-vectors
 * to get the decoded vector.
 */
struct AdditiveQuantizer {
    size_t d;                     ///< size of the input vectors
    size_t M;                     ///< number of codebooks
    std::vector<size_t> nbits;    ///< bits for each step
    std::vector<float> codebooks; ///< codebooks

    // derived values
    std::vector<uint64_t> codebook_offsets;
    size_t code_size;           ///< code size in bytes
    size_t tot_bits;            ///< total number of bits
    size_t total_codebook_size; ///< size of the codebook in vectors
    bool only_8bit;             ///< are all nbits = 8 (use faster decoder)

    bool verbose;    ///< verbose during training?
    bool is_trained; ///< is trained or not

    IndexFlat1D qnorm; ///< store and search norms

    uint32_t encode_qcint(
            float x) const; ///< encode norm by non-uniform scalar quantization

    float decode_qcint(uint32_t c)
            const; ///< decode norm by non-uniform scalar quantization

    /// Encodes how search is performed and how vectors are encoded
    enum Search_type_t {
        ST_decompress,    ///< decompress database vector
        ST_LUT_nonorm,    ///< use a LUT, don't include norms (OK for IP or
                          ///< normalized vectors)
        ST_norm_from_LUT, ///< compute the norms from the look-up tables (cost
                          ///< is in O(M^2))
        ST_norm_float, ///< use a LUT, and store float32 norm with the vectors
        ST_norm_qint8, ///< use a LUT, and store 8bit-quantized norm
        ST_norm_qint4,
        ST_norm_cqint8, ///< use a LUT, and store non-uniform quantized norm
        ST_norm_cqint4,
    };

    AdditiveQuantizer(
            size_t d,
            const std::vector<size_t>& nbits,
            Search_type_t search_type = ST_decompress);

    AdditiveQuantizer();

    ///< compute derived values when d, M and nbits have been set
    void set_derived_values();

    ///< Train the additive quantizer
    virtual void train(size_t n, const float* x) = 0;

    /** Encode a set of vectors
     *
     * @param x      vectors to encode, size n * d
     * @param codes  output codes, size n * code_size
     */
    virtual void compute_codes(const float* x, uint8_t* codes, size_t n)
            const = 0;

    /** pack a series of code to bit-compact format
     *
     * @param codes        codes to be packed, size n * code_size
     * @param packed_codes output bit-compact codes
     * @param ld_codes     leading dimension of codes
     * @param norms        norms of the vectors (size n). Will be computed if
     *                     needed but not provided
     */
    void pack_codes(
            size_t n,
            const int32_t* codes,
            uint8_t* packed_codes,
            int64_t ld_codes = -1,
            const float* norms = nullptr) const;

    /** Decode a set of vectors
     *
     * @param codes  codes to decode, size n * code_size
     * @param x      output vectors, size n * d
     */
    void decode(const uint8_t* codes, float* x, size_t n) const;

    /** Decode a set of vectors in non-packed format
     *
     * @param codes  codes to decode, size n * ld_codes
     * @param x      output vectors, size n * d
     */
    void decode_unpacked(
            const int32_t* codes,
            float* x,
            size_t n,
            int64_t ld_codes = -1) const;

    /****************************************************************************
     * Search functions in an external set of codes.
     ****************************************************************************/

    /// Also determines what's in the codes
    Search_type_t search_type;

    /// min/max for quantization of norms
    float norm_min, norm_max;

    template <bool is_IP, Search_type_t effective_search_type>
    float compute_1_distance_LUT(const uint8_t* codes, const float* LUT) const;

    /*
        float compute_1_L2sqr(const uint8_t* codes, const float* LUT);
    */
    /****************************************************************************
     * Support for exhaustive distance computations with all the centroids.
     * Hence, the number of these centroids should not be too large.
     ****************************************************************************/
    using idx_t = Index::idx_t;

    /// decoding function for a code in a 64-bit word
    void decode_64bit(idx_t n, float* x) const;

    /** Compute inner-product look-up tables. Used in the centroid search
     * functions.
     *
     * @param xq     query vector, size (n, d)
     * @param LUT    look-up table, size (n, total_codebook_size)
     */
    void compute_LUT(size_t n, const float* xq, float* LUT) const;

    /// exact IP search
    void knn_centroids_inner_product(
            idx_t n,
            const float* xq,
            idx_t k,
            float* distances,
            idx_t* labels) const;

    /** For L2 search we need the L2 norms of the centroids
     *
     * @param norms    output norms table, size total_codebook_size
     */
    void compute_centroid_norms(float* norms) const;

    /** Exact L2 search, with precomputed norms */
    void knn_centroids_L2(
            idx_t n,
            const float* xq,
            idx_t k,
            float* distances,
            idx_t* labels,
            const float* centroid_norms) const;

    virtual ~AdditiveQuantizer();
};

}; // namespace faiss
