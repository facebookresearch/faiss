/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stdint.h>

#include <vector>

#include <faiss/Clustering.h>

namespace faiss {

/** Residual quantizer with variable number of bits per sub-quantizer
 *
 * The residual centroids are stored in a big cumulative centroid table.
 * The codes are represented either as a non-compact table of size (n, M) or
 * as the compact output (n, code_size).
 */

struct ResidualQuantizer {
    size_t d;                  ///< size of the input vectors
    size_t M;                  ///< number of steps
    std::vector<size_t> nbits; ///< bits for each step

    bool verbose; ///< verbose during training?

    // derived values
    std::vector<size_t> centroid_offsets;

    size_t tot_bits;  ///< total number of bits
    size_t code_size; ///< code size in bytes
    bool is_byte_aligned;

    /// initialization
    enum train_type_t {
        Train_default,         ///< regular k-means
        Train_progressive_dim, ///< progressive dim clustering
    };

    // set this bit on train_type if beam is to be trained only on the
    // first element of the beam (faster but less accurate)
    static const int Train_top_beam = 1024;
    train_type_t train_type;

    /// beam size used for training and for encoding
    int max_beam_size;

    /// distance matrixes with beam search can get large, so use this
    /// to batch computations at encoding time.
    size_t max_mem_distances;

    /// clustering parameters
    ProgressiveDimClusteringParameters cp;

    /// if non-NULL, use this index for assignment
    ProgressiveDimIndexFactory* assign_index_factory;

    /// size d * centroid_offsets.end()
    std::vector<float> centroids;

    ResidualQuantizer(size_t d, const std::vector<size_t>& nbits);

    ResidualQuantizer(
            size_t d,      /* dimensionality of the input vectors */
            size_t M,      /* number of subquantizers */
            size_t nbits); /* number of bit per subvector index */

    ResidualQuantizer();

    /// compute derived values when d, M and nbits have been set
    void set_derived_values();

    // Train the residual quantizer
    void train(size_t n, const float* x);

    /** pack a series of code to bit-compact format
     *
     * @param ld_codes  leading dimension of codes
     */
    void pack_codes(
            size_t n,
            const int32_t* codes,
            uint8_t* packed_codes,
            int64_t ld_codes = -1) const;

    /** Encode a set of vectors
     *
     * @param x      vectors to encode, size n * d
     * @param codes  output codes, size n * code_size
     */
    void compute_codes(const float* x, uint8_t* codes, size_t n) const;

    /** Decode a set of vectors
     *
     * @param codes  codes to decode, size n * code_size
     * @param x      output vectors, size n * d
     */
    void decode(const uint8_t* code, float* x, size_t n) const;

    /** lower-level encode function
     *
     * @param n              number of vectors to hanlde
     * @param residuals      vectors to encode, size (n, beam_size, d)
     * @param beam_size      input beam size
     * @param new_beam_size  output beam size (should be <= K * beam_size)
     * @param new_codes      output codes, size (n, new_beam_size, m + 1)
     * @param new_residuals  output residuals, size (n, new_beam_size, d)
     * @param new_distances  output distances, size (n, new_beam_size)
     */
    void refine_beam(
            size_t n,
            size_t beam_size,
            const float* residuals,
            int new_beam_size,
            int32_t* new_codes,
            float* new_residuals = nullptr,
            float* new_distances = nullptr) const;

    /** Beam search can consume a lot of memory. This function estimates the
     * amount of mem used by refine_beam to adjust the batch size
     *
     * @param beam_size  if != -1, override the beam size
     */
    size_t memory_per_point(int beam_size = -1) const;

};

/** Encode a residual by sampling from a centroid table.
 *
 * This is a single encoding step the residual quantizer.
 * It allows low-level access to the encoding function, exposed mainly for unit
 * tests.
 *
 * @param n              number of vectors to hanlde
 * @param residuals      vectors to encode, size (n, beam_size, d)
 * @param cent           centroids, size (K, d)
 * @param beam_size      input beam size
 * @param m              size of the codes for the previous encoding steps
 * @param codes          code array for the previous steps of the beam (n,
 * beam_size, m)
 * @param new_beam_size  output beam size (should be <= K * beam_size)
 * @param new_codes      output codes, size (n, new_beam_size, m + 1)
 * @param new_residuals  output residuals, size (n, new_beam_size, d)
 * @param new_distances  output distances, size (n, new_beam_size)
 * @param assign_index   if non-NULL, will be used to perform assignment
 */
void beam_search_encode_step(
        size_t d,
        size_t K,
        const float* cent,
        size_t n,
        size_t beam_size,
        const float* residuals,
        size_t m,
        const int32_t* codes,
        size_t new_beam_size,
        int32_t* new_codes,
        float* new_residuals,
        float* new_distances,
        Index* assign_index = nullptr);

}; // namespace faiss
