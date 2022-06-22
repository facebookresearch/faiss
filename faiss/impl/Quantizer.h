// (c) Meta Platforms, Inc. and affiliates. Confidential and proprietary.

#pragma once

#include <stdint.h>

namespace faiss {

/** Product Quantizer. Implemented only for METRIC_L2 */
struct Quantizer {
    using idx_t = Index::idx_t;

    size_t d;         ///< size of the input vectors
    size_t code_size; ///< bytes per indexed vector

    explicit Quantizer(size_t d = 0, size_t code_size = 0)
            : d(d), code_size(code_size) {}

    /** Train the quantizer
     *
     * @param x       training vectors, size n * d
     */
    virtual void train(size_t n, const float* x) = 0;

    /** Quantize a set of vectors
     *
     * @param x        input vectors, size n * d
     * @param codes    output codes, size n * code_size
     */
    virtual void compute_codes(const float* x, uint8_t* codes, size_t n)
            const = 0;

    /** Decode a set of vectors
     *
     * @param codes    input codes, size n * code_size
     * @param x        output vectors, size n * d
     */
    virtual void decode(const uint8_t* code, float* x, size_t n) const = 0;

    virtual ~Quantizer() {}
};

} // namespace faiss
