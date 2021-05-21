/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <vector>

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
    std::vector<size_t> codebook_offsets;
    size_t code_size; ///< code size in bytes
    size_t tot_bits;  ///< total number of bits
    bool is_byte_aligned;

    bool verbose;    ///< verbose during training?
    bool is_trained; ///< is trained or not

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
     * @param codes  codes to be packed, size n * code_size
     * @param packed_codes output bit-compact codes
     * @param ld_codes  leading dimension of codes
     */
    void pack_codes(
            size_t n,
            const int32_t* codes,
            uint8_t* packed_codes,
            int64_t ld_codes = -1) const;

    /** Decode a set of vectors
     *
     * @param codes  codes to decode, size n * code_size
     * @param x      output vectors, size n * d
     */
    void decode(const uint8_t* codes, float* x, size_t n) const;

    virtual ~AdditiveQuantizer();
};

}; // namespace faiss
