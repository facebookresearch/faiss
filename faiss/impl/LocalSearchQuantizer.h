/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stdint.h>

#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include <faiss/utils/utils.h>

namespace faiss {

/** Implementation of LSQ/LSQ++ described in the following two papers:
 *
 * Revisiting additive quantization
 * Julieta Martinez, et al. ECCV 2016
 *
 * LSQ++: Lower running time and higher recall in multi-codebook quantization
 * Julieta Martinez, et al. ECCV 2018
 *
 * This implementation is mostly translated from the Julia implementations
 * by Julieta Martinez:
 * (https://github.com/una-dinosauria/local-search-quantization,
 *  https://github.com/una-dinosauria/Rayuela.jl)
 *
 * The trained codes are stored in `codebooks` which is called
 * `centroids` in PQ and RQ.
 */

struct LocalSearchQuantizer {
    size_t d;     ///< size of the input vectors
    size_t M;     ///< number of codebooks
    size_t nbits; ///< bits per subcode
    size_t K;     ///< number of codes per codebook

    bool is_trained; ///< is trained or not

    ///< logging level
    ///< 0 for no logging, 1 for basic logging, 2 for detailed logging
    int log_level;

    size_t code_size; ///< code size in bytes

    size_t train_iters; ///< number of iterations in training

    size_t encode_ils_iters; ///< iterations of local search while encoding
    size_t train_ils_iters;  ///< iterations of local search while training
    size_t icm_iters;        ///< number of iterations in icm

    float p;      ///< temperature factor
    float lambd; ///< regularization factor

    size_t chunk_size; ///< nb of vectors to encode at a time

    int random_seed;  ///< seed for random generator
    size_t nperts; ///< number of perturbation in each code

    std::vector<float> codebooks; ///< codebooks

    LocalSearchQuantizer(
            size_t d,      /* dimensionality of the input vectors */
            size_t M,      /* number of subquantizers */
            size_t nbits); /* number of bit per subvector index */

    // Train the local search quantizer
    void train(size_t n, const float* x);

    /** Encode a set of vectors
     *
     * @param x      vectors to encode, size n * d
     * @param codes  output codes, size n * code_size
     */
    void compute_codes(const float* x, uint8_t* codes, size_t n) const;

    /** pack a series of code to bit-compact format
     *
     * @param codes  codes to be packed, size n * code_size
     * @param packed_codes output bit-compact codes
     */
    void pack_codes(size_t n, const int32_t* codes, uint8_t* packed_codes)
            const;

    /** Decode a set of vectors
     *
     * @param codes  codes to decode, size n * code_size
     * @param x      output vectors, size n * d
     */
    void decode(const uint8_t* codes, float* x, size_t n) const;

    /** Update codebooks given encodings
     *
     * @param x      training vectors, size n * d
     * @param codes  encoded training vectors, size n * M
     */
    void update_codebooks(const float* x, const int32_t* codes, size_t n);

    /** Encode vectors given codebooks using iterative conditional mode (icm).
     *
     * @param x      vectors to encode, size n * d
     * @param codes  output codes, size n * M
     * @param ils_iters number of iterations of iterative local search
     */
    void icm_encode(const float* x, int32_t* codes, size_t n, size_t ils_iters, std::mt19937 &gen)
            const;

    void icm_encode_partial(
            size_t index,
            const float* x,
            int32_t* codes,
            size_t n,
            const float* binaries,
            size_t ils_iters,
            std::mt19937 &gen) const;

    /** Add some perturbation to codebooks
     *
     * @param T         temperature of simulated annealing
     * @param stddev    standard derivations of each dimension in training data
     */
    void perturb_codebooks(
            float T,
            const std::vector<float>& stddev,
            std::mt19937& gen);

    /** Add some perturbation to codes
     *
     * @param codes codes to be perturbed, size n * M
     */
    void perturb_codes(int32_t* codes, size_t n, std::mt19937 &gen) const;

    /** Compute binary terms
     *
     * @param binaries binary terms, size M * M * K * K
     */
    void compute_binary_terms(float* binaries) const;

    /** Compute unary terms
     *
     * @param x       vectors to encode, size n * d
     * @param unaries unary terms, size n * M * K
     */
    void compute_unary_terms(const float* x, float* unaries, size_t n) const;

    /** Helper function to compute reconstruction error
     *
     * @param x     vectors to encode, size n * d
     * @param codes encoded codes, size n * M
     * @param objs  if it is not null, store reconstruction
                    error of each vector into it, size n
     */
    float evaluate(
            const int32_t* codes,
            const float* x,
            size_t n,
            float* objs = nullptr) const;
};

/** A helper struct to count consuming time during training.
 *  It is NOT thread-safe.
 */
struct LSQTimer {
    std::unordered_map<std::string, double> duration;
    std::unordered_map<std::string, double> t0;
    std::unordered_map<std::string, bool> started;

    LSQTimer() {
        reset();
    }

    double get(const std::string& name);

    void start(const std::string& name);

    void end(const std::string& name);

    void reset();
};

FAISS_API extern LSQTimer lsq_timer; ///< timer to count consuming time

}; // namespace faiss
