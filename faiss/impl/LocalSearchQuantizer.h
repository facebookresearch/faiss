/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
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

#include <faiss/impl/AdditiveQuantizer.h>
#include <faiss/impl/platform_macros.h>
#include <faiss/utils/utils.h>

namespace faiss {

namespace lsq {

struct IcmEncoderFactory;

} // namespace lsq

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
struct LocalSearchQuantizer : AdditiveQuantizer {
    size_t K; ///< number of codes per codebook

    size_t train_iters = 25;      ///< number of iterations in training
    size_t encode_ils_iters = 16; ///< iterations of local search in encoding
    size_t train_ils_iters = 8;   ///< iterations of local search in training
    size_t icm_iters = 4;         ///< number of iterations in icm

    float p = 0.5f;      ///< temperature factor
    float lambd = 1e-2f; ///< regularization factor

    size_t chunk_size = 10000; ///< nb of vectors to encode at a time

    int random_seed = 0x12345; ///< seed for random generator
    size_t nperts = 4;         ///< number of perturbation in each code

    ///< if non-NULL, use this encoder to encode (owned by the object)
    lsq::IcmEncoderFactory* icm_encoder_factory = nullptr;

    bool update_codebooks_with_double = true;

    LocalSearchQuantizer(
            size_t d,     /* dimensionality of the input vectors */
            size_t M,     /* number of subquantizers */
            size_t nbits, /* number of bit per subvector index */
            Search_type_t search_type =
                    ST_decompress); /* determines the storage type */

    LocalSearchQuantizer();

    ~LocalSearchQuantizer() override;

    // Train the local search quantizer
    void train(size_t n, const float* x) override;

    /** Encode a set of vectors
     *
     * @param x      vectors to encode, size n * d
     * @param codes  output codes, size n * code_size
     * @param n      number of vectors
     * @param centroids  centroids to be added to x, size n * d
     */
    void compute_codes_add_centroids(
            const float* x,
            uint8_t* codes,
            size_t n,
            const float* centroids = nullptr) const override;

    /** Update codebooks given encodings
     *
     * @param x      training vectors, size n * d
     * @param codes  encoded training vectors, size n * M
     * @param n      number of vectors
     */
    void update_codebooks(const float* x, const int32_t* codes, size_t n);

    /** Encode vectors given codebooks using iterative conditional mode (icm).
     *
     * @param codes     output codes, size n * M
     * @param x         vectors to encode, size n * d
     * @param n         number of vectors
     * @param ils_iters number of iterations of iterative local search
     */
    void icm_encode(
            int32_t* codes,
            const float* x,
            size_t n,
            size_t ils_iters,
            std::mt19937& gen) const;

    void icm_encode_impl(
            int32_t* codes,
            const float* x,
            const float* unaries,
            std::mt19937& gen,
            size_t n,
            size_t ils_iters,
            bool verbose) const;

    void icm_encode_step(
            int32_t* codes,
            const float* unaries,
            const float* binaries,
            size_t n,
            size_t n_iters) const;

    /** Add some perturbation to codes
     *
     * @param codes codes to be perturbed, size n * M
     * @param n     number of vectors
     */
    void perturb_codes(int32_t* codes, size_t n, std::mt19937& gen) const;

    /** Add some perturbation to codebooks
     *
     * @param T         temperature of simulated annealing
     * @param stddev    standard derivations of each dimension in training data
     */
    void perturb_codebooks(
            float T,
            const std::vector<float>& stddev,
            std::mt19937& gen);

    /** Compute binary terms
     *
     * @param binaries binary terms, size M * M * K * K
     */
    void compute_binary_terms(float* binaries) const;

    /** Compute unary terms
     *
     * @param n       number of vectors
     * @param x       vectors to encode, size n * d
     * @param unaries unary terms, size n * M * K
     */
    void compute_unary_terms(const float* x, float* unaries, size_t n) const;

    /** Helper function to compute reconstruction error
     *
     * @param codes encoded codes, size n * M
     * @param x     vectors to encode, size n * d
     * @param n     number of vectors
     * @param objs  if it is not null, store reconstruction
                    error of each vector into it, size n
     */
    float evaluate(
            const int32_t* codes,
            const float* x,
            size_t n,
            float* objs = nullptr) const;
};

namespace lsq {

struct IcmEncoder {
    std::vector<float> binaries;

    bool verbose;

    const LocalSearchQuantizer* lsq;

    explicit IcmEncoder(const LocalSearchQuantizer* lsq);

    virtual ~IcmEncoder() {}

    ///< compute binary terms
    virtual void set_binary_term();

    /** Encode vectors given codebooks
     *
     * @param codes     output codes, size n * M
     * @param x         vectors to encode, size n * d
     * @param gen       random generator
     * @param n         number of vectors
     * @param ils_iters number of iterations of iterative local search
     */
    virtual void encode(
            int32_t* codes,
            const float* x,
            std::mt19937& gen,
            size_t n,
            size_t ils_iters) const;
};

struct IcmEncoderFactory {
    virtual IcmEncoder* get(const LocalSearchQuantizer* lsq) {
        return new IcmEncoder(lsq);
    }
    virtual ~IcmEncoderFactory() {}
};

/** A helper struct to count consuming time during training.
 *  It is NOT thread-safe.
 */
struct LSQTimer {
    std::unordered_map<std::string, double> t;

    LSQTimer() {
        reset();
    }

    double get(const std::string& name);

    void add(const std::string& name, double delta);

    void reset();
};

struct LSQTimerScope {
    double t0;
    LSQTimer* timer;
    std::string name;
    bool finished;

    LSQTimerScope(LSQTimer* timer, std::string name);

    void finish();

    ~LSQTimerScope();
};

} // namespace lsq

} // namespace faiss
