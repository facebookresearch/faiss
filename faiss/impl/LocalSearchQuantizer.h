/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stdint.h>

#include <vector>

// #include <faiss/Clustering.h>

namespace faiss {

/** LSQ/LSQ++
 *
 * TODO: add algo description
 */

struct LocalSearchQuantizer {
    size_t d;     ///< size of the input vectors
    size_t M;     ///< number of codebooks
    size_t nbits; ///< bits per subcode

    bool verbose; ///< verbose during training?

    size_t code_size; ///< code size in bytes

    size_t train_iters;  ///< number of iterations in training
    size_t encode_iters; ///< number of iterations in encoding

    size_t ils_iters; ///< number of iterations in local search
    size_t icm_iters; ///< number of iterations in icm

    size_t nperts; ///< number of perturbation in icm

    std::vector<float> codebooks;

    LocalSearchQuantizer(
            size_t d,      /* dimensionality of the input vectors */
            size_t M,      /* number of subquantizers */
            size_t nbits); /* number of bit per subvector index */

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

    void update_codebooks(const float* x, const int32_t* codes, size_t n);

    void icm_encode(const float* x, int32_t* codes, size_t n);

    void perturb_codes(int32_t* codes, size_t n);

    void compute_binary_terms(float* binaries);

    void compute_unary_terms(const float* x, float* unaries, size_t n);

    float evaluate(
            const int32_t* codes,
            const float* x,
            size_t n,
            float* objs = nullptr) const;
};

}; // namespace faiss
