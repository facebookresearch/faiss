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
#include <faiss/impl/AdditiveQuantizer.h>
#include <faiss/impl/LocalSearchQuantizer.h>
#include <faiss/impl/ResidualQuantizer.h>

namespace faiss {

/** Product Additive Quantizers
 * TODO: add some docs
 */
struct ProductAdditiveQuantizer : AdditiveQuantizer {
    size_t nsplits; ///< number of sub-vectors we split a vector into
    size_t dsub;    ///< dimensionality of a sub-vector
    size_t Msub;    ///< number of codebooks per sub-vectror

    std::vector<AdditiveQuantizer*> quantizers;

    ProductAdditiveQuantizer();

    ProductAdditiveQuantizer(
            size_t d,
            const std::vector<AdditiveQuantizer*>& aqs,
            Search_type_t search_type);

    void init(
            size_t d,
            const std::vector<AdditiveQuantizer*>& aqs,
            Search_type_t search_type);

    ///< Train the additive quantizer
    void train(size_t n, const float* x) override;

    /** Encode a set of vectors
     *
     * @param x      vectors to encode, size n * d
     * @param codes  output codes, size n * code_size
     */
    void compute_codes(const float* x, uint8_t* codes, size_t n) const override;

    /** Decode a set of vectors
     *
     * @param codes  codes to decode, size n * code_size
     * @param x      output vectors, size n * d
     */
    void decode(const uint8_t* codes, float* x, size_t n) const override;

    void set_verbose();

    void copy_codebooks();
};

// /** Product Local Search Quantizers
//  */
// struct ProductLSQ : ProductAdditiveQuantizer {
//     ProductLSQ(size_t d, size_t M, size_t M_sub, size_t nbits);
//     ProductLSQ();

//     virtual LocalSearchQuantizer* subquantizer(size_t m) override;
// };

// /** Product Local Search Quantizers
//  */
// struct ProductRQ : ProductAdditiveQuantizer {
//     ProductRQ(size_t d, size_t M, size_t M_sub, size_t nbits);
//     ProductRQ();

//     virtual ResidualQuantizer* subquantizer(size_t m) override;
// };

}; // namespace faiss