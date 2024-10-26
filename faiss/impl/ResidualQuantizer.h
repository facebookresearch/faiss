/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <vector>

#include <faiss/Clustering.h>
#include <faiss/impl/AdditiveQuantizer.h>

#include <faiss/utils/approx_topk/mode.h>

namespace faiss {

/** Residual quantizer with variable number of bits per sub-quantizer
 *
 * The residual centroids are stored in a big cumulative centroid table.
 * The codes are represented either as a non-compact table of size (n, M) or
 * as the compact output (n, code_size).
 */

struct ResidualQuantizer : AdditiveQuantizer {
    /// initialization

    //  Was enum but that does not work so well with bitmasks
    using train_type_t = int;

    /// Binary or of the Train_* flags below
    train_type_t train_type = Train_progressive_dim;

    /// regular k-means (minimal amount of computation)
    static const int Train_default = 0;

    /// progressive dim clustering (set by default)
    static const int Train_progressive_dim = 1;

    /// do a few iterations of codebook refinement after first level estimation
    static const int Train_refine_codebook = 2;

    /// number of iterations for codebook refinement.
    int niter_codebook_refine = 5;

    /** set this bit on train_type if beam is to be trained only on the
     *  first element of the beam (faster but less accurate) */
    static const int Train_top_beam = 1024;

    /** set this bit to *not* autmatically compute the codebook tables
     * after training */
    static const int Skip_codebook_tables = 2048;

    /// beam size used for training and for encoding
    int max_beam_size = 5;

    /// use LUT for beam search
    int use_beam_LUT = 0;

    /// Currently used mode of approximate min-k computations.
    /// Default value is EXACT_TOPK.
    ApproxTopK_mode_t approx_topk_mode = ApproxTopK_mode_t::EXACT_TOPK;

    /// clustering parameters
    ProgressiveDimClusteringParameters cp;

    /// if non-NULL, use this index for assignment
    ProgressiveDimIndexFactory* assign_index_factory = nullptr;

    ResidualQuantizer(
            size_t d,
            const std::vector<size_t>& nbits,
            Search_type_t search_type = ST_decompress);

    ResidualQuantizer(
            size_t d,     /* dimensionality of the input vectors */
            size_t M,     /* number of subquantizers */
            size_t nbits, /* number of bit per subvector index */
            Search_type_t search_type = ST_decompress);

    ResidualQuantizer();

    /// Train the residual quantizer
    void train(size_t n, const float* x) override;

    /// Copy the M codebook levels from other, starting from skip_M
    void initialize_from(const ResidualQuantizer& other, int skip_M = 0);

    /** Encode the vectors and compute codebook that minimizes the quantization
     * error on these codes
     *
     * @param x      training vectors, size n * d
     * @param n      nb of training vectors, n >= total_codebook_size
     * @return       returns quantization error for the new codebook with old
     * codes
     */
    float retrain_AQ_codebook(size_t n, const float* x);

    /** Encode a set of vectors
     *
     * @param x      vectors to encode, size n * d
     * @param codes  output codes, size n * code_size
     * @param centroids  centroids to be added to x, size n * d
     */
    void compute_codes_add_centroids(
            const float* x,
            uint8_t* codes,
            size_t n,
            const float* centroids = nullptr) const override;

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

    void refine_beam_LUT(
            size_t n,
            const float* query_norms,
            const float* query_cp,
            int new_beam_size,
            int32_t* new_codes,
            float* new_distances = nullptr) const;

    /** Beam search can consume a lot of memory. This function estimates the
     * amount of mem used by refine_beam to adjust the batch size
     *
     * @param beam_size  if != -1, override the beam size
     */
    size_t memory_per_point(int beam_size = -1) const;
};

} // namespace faiss
