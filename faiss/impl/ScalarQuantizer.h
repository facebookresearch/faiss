/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstring>

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
        QT_bf16,
        QT_8bit_direct_signed, ///< fast indexing of signed int8s ranging from
                               ///< [-128 to 127]
        QT_0bit, ///< 0 bits per component, centroid-only distance (for IVF)
        QT_1bit_tqmse, ///< TurboQuant MSE-optimized, 1 bit per component
        QT_2bit_tqmse, ///< TurboQuant MSE-optimized, 2 bits per component
        QT_3bit_tqmse, ///< TurboQuant MSE-optimized, 3 bits per component
        QT_4bit_tqmse, ///< TurboQuant MSE-optimized, 4 bits per component
        QT_8bit_tqmse, ///< TurboQuant MSE-optimized, 8 bits per component
        QT_2bit_tq,    ///< Full TurboQuant (1-bit MSE + 1-bit QJL + factors)
        QT_3bit_tq,    ///< Full TurboQuant (2-bit MSE + 1-bit QJL + factors)
        QT_4bit_tq,    ///< Full TurboQuant (3-bit MSE + 1-bit QJL + factors)
        QT_5bit_tq,    ///< Full TurboQuant (4-bit MSE + 1-bit QJL + factors)
        QT_count
    };

    QuantizerType qtype = QT_8bit;

    /** The uniform encoder can estimate the range of representable
     * values of the uniform encoder using different statistics. Here
     * rs = rangestat_arg */

    // rangestat_arg.
    enum RangeStat {
        RS_minmax,    ///< [min - rs*(max-min), max + rs*(max-min)]
        RS_meanstd,   ///< [mean - std * rs, mean + std * rs]
        RS_quantiles, ///< [Q(rs), Q(1-rs)]
        RS_optim,     ///< alternate optimization of reconstruction error
    };

    RangeStat rangestat = RS_minmax;
    float rangestat_arg = 0;

    /// bits per scalar code
    size_t bits = 0;

    /// trained values (including the range)
    std::vector<float> trained;

    ScalarQuantizer(size_t d_in, QuantizerType qtype_in);
    ScalarQuantizer();

    /// updates internal values based on qtype and d
    void set_derived_sizes();

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
        SQDistanceComputer() : FlatCodesDistanceComputer(nullptr) {}

        virtual float query_to_code(const uint8_t* code) const = 0;

        /// Compute four query-to-code distances in one call. Default loops
        /// query_to_code four times; per-SIMD specializations may batch the
        /// inner dim loop across the four codes to amortize query state and
        /// expose ILP across independent accumulators.
        virtual void query_to_codes_batch_4(
                const uint8_t* code_0,
                const uint8_t* code_1,
                const uint8_t* code_2,
                const uint8_t* code_3,
                float& dis0,
                float& dis1,
                float& dis2,
                float& dis3) const {
            dis0 = query_to_code(code_0);
            dis1 = query_to_code(code_1);
            dis2 = query_to_code(code_2);
            dis3 = query_to_code(code_3);
        }

        float distance_to_code(const uint8_t* code) final {
            return query_to_code(code);
        }
    };

    /// TurboQuant full (QT_*_tq) refinement state, isolated from the
    /// main ScalarQuantizer to avoid polluting it with TQ-specific data.
    struct TurboQuantRefine {
        static bool is_turboq_full(QuantizerType qt) {
            return qt >= QT_2bit_tq && qt <= QT_5bit_tq;
        }

        static void pack_seed(uint64_t seed, float out[2]) {
            static_assert(sizeof(uint64_t) == 2 * sizeof(float));
            std::memcpy(out, &seed, sizeof(uint64_t));
        }

        static uint64_t unpack_seed(float lo, float hi) {
            float tmp[2] = {lo, hi};
            uint64_t s;
            static_assert(sizeof(uint64_t) == 2 * sizeof(float));
            std::memcpy(&s, tmp, sizeof(uint64_t));
            return s;
        }

        uint8_t qjl_type = 0;
        uint64_t seed = 42;
        size_t padded_d = 0;
        std::vector<float> fwht_signs;
        std::vector<float> rr_matrix;
        size_t nb_bits_lo = 0;
        size_t n_hi_dims = 0;

        void init_projection(size_t d);
        bool use_fwht() const {
            return qjl_type == 0;
        }

        struct DistanceComputer : SQDistanceComputer {
            virtual void configure(uint8_t qb, bool int_qjl) = 0;
            virtual void set_prescreen_threshold(
                    const float* t,
                    bool minimize) = 0;
            virtual void clear_prescreen_threshold() = 0;
        };
    };

    TurboQuantRefine turboq_refine;

    SQDistanceComputer* get_distance_computer(
            MetricType metric = METRIC_L2) const;

    InvertedListScanner* select_InvertedListScanner(
            MetricType mt,
            const Index* quantizer,
            bool store_pairs,
            const IDSelector* sel,
            bool by_residual = false) const;
};

} // namespace faiss
