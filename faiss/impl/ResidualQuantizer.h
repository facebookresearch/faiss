/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <vector>

#include <faiss/Clustering.h>
#include <faiss/impl/AdditiveQuantizer.h>

namespace faiss {

/** Residual quantizer with variable number of bits per sub-quantizer
 *
 * The residual centroids are stored in a big cumulative centroid table.
 * The codes are represented either as a non-compact table of size (n, M) or
 * as the compact output (n, code_size).
 */

struct ResidualQuantizer : AdditiveQuantizer {
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

    ResidualQuantizer(size_t d, const std::vector<size_t>& nbits);

    ResidualQuantizer(
            size_t d,      /* dimensionality of the input vectors */
            size_t M,      /* number of subquantizers */
            size_t nbits); /* number of bit per subvector index */

    ResidualQuantizer();

    // Train the residual quantizer
    void train(size_t n, const float* x) override;

    /** Encode a set of vectors
     *
     * @param x      vectors to encode, size n * d
     * @param codes  output codes, size n * code_size
     */
    void compute_codes(const float* x, uint8_t* codes, size_t n) const override;

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
