/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#pragma once

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/Quantizer.h>

namespace faiss {

struct InvertedListScanner;

/**
 * The uniform quantizer has a range [vmin, vmax]. The range can be
 * the same for all dimensions (uniform) or specific per dimension
 * (default).
 */

struct ScalarQuantizer : Quantizer {
    enum QuantizerType {
        QT_8bit,         ///< 8 bits per component
        QT_4bit,         ///< 4 bits per component
        QT_8bit_uniform, ///< same, shared range for all dimensions
        QT_4bit_uniform,
        QT_fp16,
        QT_8bit_direct, ///< fast indexing of uint8s
        QT_6bit,        ///< 6 bits per component
    };

    QuantizerType qtype;

    /** The uniform encoder can estimate the range of representable
     * values of the unform encoder using different statistics. Here
     * rs = rangestat_arg */

    // rangestat_arg.
    enum RangeStat {
        RS_minmax,    ///< [min - rs*(max-min), max + rs*(max-min)]
        RS_meanstd,   ///< [mean - std * rs, mean + std * rs]
        RS_quantiles, ///< [Q(rs), Q(1-rs)]
        RS_optim,     ///< alternate optimization of reconstruction error
    };

    RangeStat rangestat;
    float rangestat_arg;

    /// bits per scalar code
    size_t bits;

    /// trained values (including the range)
    std::vector<float> trained;

    ScalarQuantizer(size_t d, QuantizerType qtype);
    ScalarQuantizer();

    /// updates internal values based on qtype and d
    void set_derived_sizes();

    void train(size_t n, const float* x) override;

    /// Used by an IVF index to train based on the residuals
    void train_residual(
            size_t n,
            const float* x,
            Index* quantizer,
            bool by_residual,
            bool verbose);

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
    void decode(const uint8_t* code, float* x, size_t n) const override;

    /*****************************************************
     * Objects that provide methods for encoding/decoding, distance
     * computation and inverted list scanning
     *****************************************************/

    struct SQuantizer {
        // encodes one vector. Assumes code is filled with 0s on input!
        virtual void encode_vector(const float* x, uint8_t* code) const = 0;
        virtual void decode_vector(const uint8_t* code, float* x) const = 0;

        virtual ~SQuantizer() {}
    };

    SQuantizer* select_quantizer() const;

    struct SQDistanceComputer : FlatCodesDistanceComputer {
        const float* q;

        SQDistanceComputer() : q(nullptr) {}

        virtual float query_to_code(const uint8_t* code) const = 0;

        float distance_to_code(const uint8_t* code) final {
            return query_to_code(code);
        }
    };

    SQDistanceComputer* get_distance_computer(
            MetricType metric = METRIC_L2) const;

    InvertedListScanner* select_InvertedListScanner(
            MetricType mt,
            const Index* quantizer,
            bool store_pairs,
            bool by_residual = false) const;
};

} // namespace faiss
