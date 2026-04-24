/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// TurboQuant: Two-stage vector quantizer from ICLR 2026.
//
// Reference:
//   Zandieh, Daliri, Hadian, Mirrokni. "TurboQuant: Online Vector
//   Quantization with Near-optimal Distortion Rate." ICLR 2026.
//
// Algorithm overview (paper Algorithm 2, "TurboQuant_prod"):
//
//   Stage 1 (MSE, b-1 bits): Lloyd-Max scalar quantization of the rotated,
//   normalized vector. Each coordinate is independently quantized using
//   a codebook optimal for the Beta distribution of random points on the
//   unit hypersphere (Lemma 1 of paper). In high dimensions this converges
//   to N(0, 1/d). At b=2 (1-bit MSE), the boundary is at 0 (= sign bit),
//   identical to RaBitQ's first bit.
//
//   Stage 2 (QJL, 1 bit): Quantized Johnson-Lindenstrauss transform of the
//   Stage 1 residual. A random projection (FWHT or Gaussian) followed by
//   sign-bit quantization. The QJL paper proves this yields an unbiased
//   inner product estimator (Lemma 3.2 of Zandieh et al., 2024).
//
// The random rotation for variance equalization is handled externally by
// IndexPreTransform (RR prefix), same as RaBitQ.
//
// Code layout per vector:
//   [MSE hi bit planes][MSE lo bit planes][QJL sign bits][TurboQFactors]
//
// Factory strings:
//   "TurboQ2"    = 2-bit (1-bit MSE + 1-bit QJL), FWHT projection (default)
//   "TurboQ4"    = 4-bit (3-bit MSE + 1-bit QJL), FWHT projection
//   "TurboQ2r"   = 2-bit, random rotation projection (QR-orthogonalized)
//   "TurboQ2.5"  = adaptive: 3-bit hi on d/4 dims, 2-bit lo on rest
//   "TurboQ3.5"  = adaptive: 4-bit hi on d/2 dims, 3-bit lo on rest

#pragma once

#include <cstddef>
#include <cstdint>
#include <vector>

#include <faiss/MetricType.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/Quantizer.h>
#include <faiss/impl/platform_macros.h>

namespace faiss {

/// QJL projection matrix type for TurboQuant Stage 2.
enum class QJLProjectionType {
    FWHT = 0,            ///< SRHT: Hadamard x random signs, O(d log d)
    RANDOM_ROTATION = 2, ///< Dense random orthogonal matrix (QR of Gaussian)
};

/// Per-vector correction factors for TurboQ distance computation.
///
/// Paper Algorithm 2, lines 7-8:
///   norm  = ||x||  (for scaling back from unit sphere)
///   gamma = ||r||  (residual norm, for QJL scaling: sqrt(pi/2)/d * gamma)
FAISS_PACK_STRUCTS_BEGIN
struct FAISS_PACKED TurboQFactors {
    float norm = 0;  ///< ||x|| before normalization
    float gamma = 0; ///< ||residual|| after MSE quantization
};
FAISS_PACK_STRUCTS_END

struct TurboQuantizer : Quantizer {
    MetricType metric_type = MetricType::METRIC_L2;

    /// Total bits per dimension (1-5).
    size_t nb_bits = 2;

    /// QJL projection type: FWHT (default), Gaussian, or Random Rotation.
    QJLProjectionType qjl_type = QJLProjectionType::FWHT;

    /// Seed for random projection matrix generation.
    uint64_t seed = 42;

    /// Padded dimension for FWHT (next power of 2 >= d).
    size_t padded_d = 0;

    /// Random signs for FWHT mode (length padded_d).
    std::vector<float> fwht_signs;

    /// Dense matrix for Random Rotation mode (d x d floats).
    std::vector<float> gaussian_matrix;

    // --- Lloyd-Max codebook for the MSE stage ---
    size_t mse_levels = 0;
    std::vector<float> mse_centroids;
    std::vector<float> mse_boundaries;

    // --- Adaptive bit width (optional) ---
    /// Low-precision bits for non-outlier dimensions. 0 = uniform mode.
    size_t nb_bits_lo = 0;
    /// Number of outlier (high-precision) dimensions.
    size_t n_hi_dims = 0;
    size_t mse_levels_lo = 0;
    std::vector<float> mse_centroids_lo;
    std::vector<float> mse_boundaries_lo;

    TurboQuantizer(
            size_t d = 0,
            MetricType metric = MetricType::METRIC_L2,
            size_t nb_bits = 2,
            QJLProjectionType qjl_type = QJLProjectionType::FWHT,
            size_t nb_bits_lo = 0,
            size_t n_hi_dims = 0);

    void train(size_t n, const float* x) override;
    void compute_codes(const float* x, uint8_t* codes, size_t n) const override;
    void compute_codes_core(
            const float* x,
            uint8_t* codes,
            size_t n,
            const float* centroid_in) const;
    void decode(const uint8_t* codes, float* x, size_t n) const override;
    void decode_core(
            const uint8_t* codes,
            float* x,
            size_t n,
            const float* centroid_in) const;

    FlatCodesDistanceComputer* get_distance_computer() const;

    void init_fwht();
    void init_random_rotation();
    void init_codebook();
    void apply_qjl_projection(const float* in, float* out) const;

    bool use_fwht() const {
        return qjl_type == QJLProjectionType::FWHT;
    }
    bool use_dense_matrix() const {
        return qjl_type != QJLProjectionType::FWHT;
    }
    bool is_adaptive() const {
        return nb_bits_lo > 0;
    }

    size_t mse_code_size_hi() const {
        size_t mse_bits = (nb_bits > 1) ? (nb_bits - 1) : 0;
        size_t hi_dims = is_adaptive() ? n_hi_dims : d;
        return mse_bits * ((hi_dims + 7) / 8);
    }
    size_t mse_code_size_lo() const {
        if (!is_adaptive())
            return 0;
        size_t mse_bits_lo_val = (nb_bits_lo > 1) ? (nb_bits_lo - 1) : 0;
        size_t lo_dims = d - n_hi_dims;
        return mse_bits_lo_val * ((lo_dims + 7) / 8);
    }
    size_t mse_code_size() const {
        return mse_code_size_hi() + mse_code_size_lo();
    }
    size_t qjl_code_size() const {
        return (d + 7) / 8;
    }
};

/// Distance computer for TurboQ with two-stage pre-screening.
///
/// When prescreen is active (threshold_ptr != nullptr), distance_to_code()
/// computes the cheap MSE component first and checks whether the full
/// MSE+QJL distance could possibly beat the current threshold. If not,
/// it returns the MSE-only estimate, skipping the expensive QJL dot product.
struct TurboQDistanceComputer : FlatCodesDistanceComputer {
    const TurboQuantizer* tq = nullptr;

    std::vector<float> q;
    std::vector<float> q_proj;
    float q_norm_sq = 0;
    float inv_sqrt_d = 0;
    float qjl_coeff = 0;

    /// Pre-computed scaled centroids for SIMD accumulation.
    std::vector<float> scaled_centroids;
    std::vector<float> scaled_centroids_lo;

    /// Per-query QJL error coefficient for pre-screening.
    float qjl_error_coeff = 0;

    /// Pointer to current heap threshold for pre-screening.
    const float* threshold_ptr = nullptr;
    bool prescreen_l2 = false;
    mutable size_t n_total = 0;
    mutable size_t n_skipped = 0;

    /// Pre-computed sums for SIMD masked accumulation.
    float total_qproj_sum = 0;
    float total_q_sum = 0;
    float delta_centroid = 0;

    /// Integer popcount path for 1-bit MSE (qb > 0, nb_bits == 2, uniform).
    uint8_t qb = 0;
    std::vector<uint8_t> rearranged_q;
    float mse_base = 0;
    float mse_int_scale = 0;
    float mse_popcnt_scale = 0;

    /// Integer QJL path.
    bool int_qjl = false;
    std::vector<uint8_t> rearranged_qproj;
    float qjl_and_scale = 0;
    float qjl_pop_scale = 0;
    float qjl_base = 0;

    float symmetric_dis(idx_t, idx_t) override;
    void set_query(const float* x) override;
    float distance_to_code(const uint8_t* code) override;
};

} // namespace faiss
