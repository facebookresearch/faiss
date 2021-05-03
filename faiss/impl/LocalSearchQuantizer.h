/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <stdint.h>

#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

#include <faiss/utils/utils.h>
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
    size_t K;     ///< number of codes per codebook

    ///< logging level
    ///< 0 for no logging, 1 for basic logging, 2 for detailed logging
    int log_level;

    size_t code_size; ///< code size in bytes

    size_t train_iters; ///< number of iterations in training

    size_t encode_ils_iters; ///< iterations of local search while encoding
    size_t train_ils_iters;  ///< iterations of local search while training
    size_t icm_iters;        ///< number of iterations in icm

    float p;      ///< temperature factor
    float lambda; ///< regularization factor

    size_t chunk_size; ///< batch size to process icm encoding

    size_t nperts; ///< number of perturbation in icm

    std::vector<float> codebooks;

    LocalSearchQuantizer(
            size_t d,      /* dimensionality of the input vectors */
            size_t M,      /* number of subquantizers */
            size_t nbits); /* number of bit per subvector index */

    // Train the residual quantizer
    void train(size_t n, const float* x);

    /** Encode a set of vectors
     *
     * @param x      vectors to encode, size n * d
     * @param codes  output codes, size n * code_size
     */
    void compute_codes(const float* x, uint8_t* codes, size_t n) const;

    void pack_codes(size_t n, const int32_t* codes, uint8_t* packed_codes)
            const;

    /** Decode a set of vectors
     *
     * @param codes  codes to decode, size n * code_size
     * @param x      output vectors, size n * d
     */
    void decode(const uint8_t* codes, float* x, size_t n) const;

    void update_codebooks(const float* x, const int32_t* codes, size_t n);

    void icm_encode(const float* x, int32_t* codes, size_t n, size_t ils_iters)
            const;

    void icm_encode_partial(
            size_t index,
            const float* x,
            int32_t* codes,
            size_t n,
            const float* binaries,
            size_t ils_iters) const;

    void perturb_codebooks(
            float T,
            const std::vector<float>& stddev,
            std::mt19937& gen);

    void perturb_codes(int32_t* codes, size_t n) const;

    void compute_binary_terms(float* binaries) const;

    void compute_unary_terms(const float* x, float* unaries, size_t n) const;

    float evaluate(
            const int32_t* codes,
            const float* x,
            size_t n,
            float* objs = nullptr) const;
};

/**
 *  A helper struct to count consuming time during training
 *  It is NOT thread-safe
 */
struct LSQTimer {
    std::unordered_map<std::string, double> duration;
    std::unordered_map<std::string, double> t0;
    std::unordered_map<std::string, bool> started;
    // double t0;

    LSQTimer() {
        reset();
    }

    double get(const std::string& name) {
        return duration[name];
    }

    void start(const std::string& name) {
        FAISS_THROW_IF_NOT_MSG(!started[name], " timer is already running");
        started[name] = true;
        t0[name] = getmillisecs();
    }

    void end(const std::string& name) {
        FAISS_THROW_IF_NOT_MSG(started[name], " timer is not running");
        double t1 = getmillisecs();
        double sec = (t1 - t0[name]) / 1000;
        duration[name] += sec;
        started[name] = false;
    }

    void reset() {
        duration.clear();
        t0.clear();
        started.clear();
    }
};

FAISS_API extern LSQTimer lsq_timer; ///< timer to count consuming time

}; // namespace faiss
