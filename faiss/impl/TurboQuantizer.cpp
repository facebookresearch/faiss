/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/TurboQuantizer.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/RaBitQUtils.h>
#include <faiss/impl/scalar_quantizer/training.h>
#include <faiss/impl/simd_dispatch.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/rabitq_simd.h>
#include <faiss/utils/random.h>
#include <faiss/utils/turboq_simd.h>
#include <faiss/utils/utils.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <vector>

namespace faiss {

namespace {

/// Return the smallest power of 2 >= n.
size_t next_power_of_2(size_t n) {
    if (n == 0) {
        return 1;
    }
    size_t p = 1;
    while (p < n) {
        p <<= 1;
    }
    return p;
}

/// In-place unnormalized Walsh-Hadamard transform of length-d vector x.
/// d must be a power of 2.
void fwht_inplace(float* x, size_t d) {
    for (size_t h = 1; h < d; h <<= 1) {
        for (size_t i = 0; i < d; i += h << 1) {
            for (size_t j = i; j < i + h; j++) {
                float u = x[j];
                float v = x[j + h];
                x[j] = u + v;
                x[j + h] = u - v;
            }
        }
    }
}

/// Compute dimension-aware Lloyd-Max codebook using the exact Beta distribution
/// of unit-sphere coordinates. Centroids and boundaries are returned in
/// "N(0,1)-equivalent" space (scaled by sqrt(d)) so that the encoding logic
/// can work uniformly: val = v_normalized * sqrt(d), then search boundaries.
void compute_codebook_for_dim(
        size_t d,
        size_t mse_bits,
        size_t& levels_out,
        std::vector<float>& centroids_out,
        std::vector<float>& boundaries_out) {
    levels_out = size_t(1) << mse_bits;

    // Use D102195204's numerical Beta-distribution codebook computation
    std::vector<float> trained;
    scalar_quantizer::train_TurboQuantMSE(d, mse_bits, trained);

    size_t k = levels_out;
    centroids_out.assign(trained.begin(), trained.begin() + k);
    boundaries_out.assign(trained.begin() + k, trained.end());

    // train_TurboQuantMSE returns centroids on [-1, 1] (coordinate space).
    // Scale by sqrt(d) to convert to "N(0,1)-equivalent" space, matching
    // the encoding convention: val = v_normalized * sqrt(d).
    float sqrt_d = std::sqrt(static_cast<float>(d));
    for (auto& c : centroids_out) {
        c *= sqrt_d;
    }
    for (auto& b : boundaries_out) {
        b *= sqrt_d;
    }
}

} // anonymous namespace

// =========================================================================
// Constructor
// =========================================================================

TurboQuantizer::TurboQuantizer(
        size_t d_in,
        MetricType metric,
        size_t nb_bits_in,
        QJLProjectionType qjl_type_in,
        size_t nb_bits_lo_in,
        size_t n_hi_dims_in)
        : Quantizer(d_in, 0),
          metric_type{metric},
          nb_bits{nb_bits_in},
          qjl_type{qjl_type_in},
          nb_bits_lo{nb_bits_lo_in},
          n_hi_dims{n_hi_dims_in} {
    if (d == 0) {
        return;
    }

    // Validate nb_bits range (1-5 total = 0-4 MSE bits + 1 QJL bit)
    FAISS_THROW_IF_NOT_MSG(
            nb_bits >= 1 && nb_bits <= 5,
            "TurboQuantizer: nb_bits must be in [1, 5]");

    // Validate adaptive parameters
    if (nb_bits_lo > 0) {
        FAISS_THROW_IF_NOT_MSG(
                nb_bits_lo < nb_bits,
                "TurboQuantizer: nb_bits_lo must be < nb_bits");
        FAISS_THROW_IF_NOT_MSG(
                n_hi_dims > 0 && n_hi_dims < d,
                "TurboQuantizer: n_hi_dims must be in (0, d)");
    }

    // Code layout: [MSE hi planes][MSE lo planes][QJL signs][TurboQFactors]
    code_size = mse_code_size() + qjl_code_size() + sizeof(TurboQFactors);

    // Initialize codebooks
    init_codebook();

    // Initialize projection matrix
    switch (qjl_type) {
        case QJLProjectionType::FWHT:
            init_fwht();
            break;
        case QJLProjectionType::RANDOM_ROTATION:
            init_random_rotation();
            break;
    }
}

// =========================================================================
// Initialization methods
// =========================================================================

void TurboQuantizer::init_fwht() {
    padded_d = next_power_of_2(d);
    fwht_signs.resize(padded_d);

    // Generate random +/-1 signs from seed
    RandomGenerator rng(seed);
    for (size_t i = 0; i < padded_d; i++) {
        fwht_signs[i] = (rng.rand_int(2) == 0) ? 1.0f : -1.0f;
    }
}

void TurboQuantizer::init_random_rotation() {
    // Generate d x d Gaussian matrix, then QR orthogonalize
    gaussian_matrix.resize(d * d);

    // Fill with N(0,1) random values
    float_randn(gaussian_matrix.data(), d * d, seed);

    // QR decomposition to get orthogonal matrix
    // matrix_qr expects column-major m x n matrix with m >= n
    // We have a d x d matrix
    matrix_qr(static_cast<int>(d), static_cast<int>(d), gaussian_matrix.data());
}

void TurboQuantizer::init_codebook() {
    if (nb_bits <= 1) {
        mse_levels = 0;
        return;
    }

    size_t mse_bits = nb_bits - 1;
    compute_codebook_for_dim(
            d, mse_bits, mse_levels, mse_centroids, mse_boundaries);

    if (is_adaptive()) {
        size_t mse_bits_lo = nb_bits_lo > 1 ? nb_bits_lo - 1 : 0;
        if (mse_bits_lo > 0) {
            compute_codebook_for_dim(
                    d,
                    mse_bits_lo,
                    mse_levels_lo,
                    mse_centroids_lo,
                    mse_boundaries_lo);
        } else {
            mse_levels_lo = 0;
        }
    }
}

void TurboQuantizer::apply_qjl_projection(const float* in, float* out) const {
    if (use_fwht()) {
        // FWHT mode: pad, sign-multiply, FWHT, truncate to d
        std::vector<float> padded(padded_d, 0.0f);

        // Copy input and multiply by random signs
        for (size_t i = 0; i < d; i++) {
            padded[i] = in[i] * fwht_signs[i];
        }
        // Zero-pad remaining (already done by initialization)
        for (size_t i = d; i < padded_d; i++) {
            padded[i] = 0.0f * fwht_signs[i]; // = 0
        }

        // Apply Walsh-Hadamard transform
        fwht_inplace(padded.data(), padded_d);

        // Truncate to d dimensions and normalize
        float scale = 1.0f / std::sqrt(static_cast<float>(padded_d));
        for (size_t i = 0; i < d; i++) {
            out[i] = padded[i] * scale;
        }
    } else {
        // Dense matrix multiply: out = gaussian_matrix * in
        // gaussian_matrix is d x d, in is d x 1, out is d x 1
        for (size_t i = 0; i < d; i++) {
            float sum = 0.0f;
            for (size_t j = 0; j < d; j++) {
                sum += gaussian_matrix[i * d + j] * in[j];
            }
            out[i] = sum;
        }
    }
}

// =========================================================================
// Training (no-op: codebooks are analytic)
// =========================================================================

void TurboQuantizer::train(size_t /*n*/, const float* /*x*/) {
    // No-op: Lloyd-Max codebooks are analytic for N(0,1).
}

// =========================================================================
// Encoding
// =========================================================================

void TurboQuantizer::compute_codes(const float* x, uint8_t* codes, size_t n)
        const {
    compute_codes_core(x, codes, n, nullptr);
}

void TurboQuantizer::compute_codes_core(
        const float* x,
        uint8_t* codes,
        size_t n,
        const float* centroid_in) const {
    FAISS_ASSERT(codes != nullptr);
    FAISS_ASSERT(x != nullptr);

    if (n == 0) {
        return;
    }

    const size_t mse_bits = (nb_bits > 1) ? (nb_bits - 1) : 0;
    const float sqrt_d = std::sqrt(static_cast<float>(d));
    const float inv_sqrt_d = 1.0f / sqrt_d;
    const size_t mse_hi_size = mse_code_size_hi();
    const size_t mse_total = mse_code_size();
    const size_t qjl_size = qjl_code_size();
    const size_t hi_dims = is_adaptive() ? n_hi_dims : d;

    // Adaptive: lo bits
    const size_t mse_bits_lo =
            is_adaptive() ? ((nb_bits_lo > 1) ? (nb_bits_lo - 1) : 0) : 0;

#pragma omp parallel for if (n > 1000)
    for (int64_t i = 0; i < static_cast<int64_t>(n); i++) {
        const float* x_row = x + i * d;
        uint8_t* code = codes + i * code_size;

        // Clear code memory
        memset(code, 0, code_size);

        // Step 1: Compute norm and normalize
        float norm_sq = fvec_norm_L2sqr(x_row, d);
        float norm = std::sqrt(norm_sq);
        if (norm < 1e-30f) {
            norm = 1e-30f;
        }
        float inv_norm = 1.0f / norm;

        // v = x / ||x|| (unit vector)
        std::vector<float> v(d);
        for (size_t j = 0; j < d; j++) {
            v[j] = x_row[j] * inv_norm;
        }

        // Step 2: MSE quantize in N(0,1) space
        // val = v[j] * sqrt(d) maps unit sphere coords to approx N(0,1)
        std::vector<float> dequant(d, 0.0f);

        if (mse_bits > 0) {
            // Pointers into MSE code region
            uint8_t* mse_code = code;

            // --- Hi dimensions ---
            size_t hi_byte_plane = (hi_dims + 7) / 8;
            for (size_t j = 0; j < hi_dims; j++) {
                float val = v[j] * sqrt_d;

                size_t idx = std::upper_bound(
                                     mse_boundaries.begin(),
                                     mse_boundaries.end(),
                                     val) -
                        mse_boundaries.begin();

                // Store as BIT-PLANE layout
                for (size_t b = 0; b < mse_bits; b++) {
                    if (idx & (size_t(1) << b)) {
                        mse_code[b * hi_byte_plane + j / 8] |= (1 << (j % 8));
                    }
                }

                // Dequantize for residual computation
                dequant[j] = mse_centroids[idx] * inv_sqrt_d;
            }

            // --- Lo dimensions (adaptive mode) ---
            if (is_adaptive() && mse_bits_lo > 0) {
                size_t lo_dims = d - n_hi_dims;
                size_t lo_byte_plane = (lo_dims + 7) / 8;
                uint8_t* mse_lo_code = mse_code + mse_hi_size;

                for (size_t jj = 0; jj < lo_dims; jj++) {
                    size_t j = n_hi_dims + jj;
                    float val = v[j] * sqrt_d;

                    size_t idx = std::upper_bound(
                                         mse_boundaries_lo.begin(),
                                         mse_boundaries_lo.end(),
                                         val) -
                            mse_boundaries_lo.begin();

                    // Store as BIT-PLANE layout
                    for (size_t b = 0; b < mse_bits_lo; b++) {
                        if (idx & (size_t(1) << b)) {
                            mse_lo_code[b * lo_byte_plane + jj / 8] |=
                                    (1 << (jj % 8));
                        }
                    }

                    // Dequantize
                    dequant[j] = mse_centroids_lo[idx] * inv_sqrt_d;
                }
            }
        }

        // Step 3: Compute residual: r = v - dequant
        std::vector<float> residual(d);
        for (size_t j = 0; j < d; j++) {
            residual[j] = v[j] - dequant[j];
        }

        // Step 4: QJL - project residual and take signs
        std::vector<float> proj(d);
        apply_qjl_projection(residual.data(), proj.data());

        uint8_t* qjl_code = code + mse_total;
        for (size_t j = 0; j < d; j++) {
            if (proj[j] >= 0.0f) {
                qjl_code[j / 8] |= (1 << (j % 8));
            }
        }

        // Step 5: Store TurboQFactors
        float gamma_sq = fvec_norm_L2sqr(residual.data(), d);
        float gamma = std::sqrt(gamma_sq);

        TurboQFactors* factors =
                reinterpret_cast<TurboQFactors*>(code + mse_total + qjl_size);
        factors->norm = norm;
        factors->gamma = gamma;
    }
}

// =========================================================================
// Decoding
// =========================================================================

void TurboQuantizer::decode(const uint8_t* codes, float* x, size_t n) const {
    decode_core(codes, x, n, nullptr);
}

void TurboQuantizer::decode_core(
        const uint8_t* codes,
        float* x,
        size_t n,
        const float* /*centroid_in*/) const {
    FAISS_ASSERT(codes != nullptr);
    FAISS_ASSERT(x != nullptr);

    if (n == 0) {
        return;
    }

    const size_t mse_bits = (nb_bits > 1) ? (nb_bits - 1) : 0;
    const float inv_sqrt_d = 1.0f / std::sqrt(static_cast<float>(d));
    const float sqrt_pi_over_2 = std::sqrt(static_cast<float>(M_PI) / 2.0f);
    const float qjl_recon_coeff = sqrt_pi_over_2 / static_cast<float>(d);
    const size_t mse_hi_size = mse_code_size_hi();
    const size_t mse_total = mse_code_size();
    const size_t qjl_size = qjl_code_size();
    const size_t hi_dims = is_adaptive() ? n_hi_dims : d;

    const size_t mse_bits_lo =
            is_adaptive() ? ((nb_bits_lo > 1) ? (nb_bits_lo - 1) : 0) : 0;

#pragma omp parallel for if (n > 1000)
    for (int64_t i = 0; i < static_cast<int64_t>(n); i++) {
        const uint8_t* code = codes + i * code_size;
        float* x_out = x + i * d;

        // Extract TurboQFactors
        const TurboQFactors* factors = reinterpret_cast<const TurboQFactors*>(
                code + mse_total + qjl_size);
        float norm = factors->norm;
        float gamma = factors->gamma;

        // Step 1: Reconstruct MSE component
        std::vector<float> x_mse(d, 0.0f);
        if (mse_bits > 0) {
            const uint8_t* mse_code = code;
            size_t hi_byte_plane = (hi_dims + 7) / 8;

            // Hi dimensions
            for (size_t j = 0; j < hi_dims; j++) {
                size_t idx = 0;
                for (size_t b = 0; b < mse_bits; b++) {
                    bool bit = (mse_code[b * hi_byte_plane + j / 8] &
                                (1 << (j % 8))) != 0;
                    if (bit) {
                        idx |= (size_t(1) << b);
                    }
                }
                x_mse[j] = mse_centroids[idx] * inv_sqrt_d;
            }

            // Lo dimensions (adaptive mode)
            if (is_adaptive() && mse_bits_lo > 0) {
                size_t lo_dims = d - n_hi_dims;
                size_t lo_byte_plane = (lo_dims + 7) / 8;
                const uint8_t* mse_lo_code = mse_code + mse_hi_size;

                for (size_t jj = 0; jj < lo_dims; jj++) {
                    size_t j = n_hi_dims + jj;
                    size_t idx = 0;
                    for (size_t b = 0; b < mse_bits_lo; b++) {
                        bool bit = (mse_lo_code[b * lo_byte_plane + jj / 8] &
                                    (1 << (jj % 8))) != 0;
                        if (bit) {
                            idx |= (size_t(1) << b);
                        }
                    }
                    x_mse[j] = mse_centroids_lo[idx] * inv_sqrt_d;
                }
            }
        }

        // Step 2: Reconstruct QJL component
        // x_qjl = sqrt(pi/2)/d * gamma * S^T * signs
        std::vector<float> x_qjl(d, 0.0f);
        const uint8_t* qjl_code = code + mse_total;

        // Build sign vector (+1/-1 from bits)
        std::vector<float> signs(d);
        for (size_t j = 0; j < d; j++) {
            bool bit = (qjl_code[j / 8] & (1 << (j % 8))) != 0;
            signs[j] = bit ? 1.0f : -1.0f;
        }

        if (use_fwht()) {
            // FWHT adjoint: embed signs in padded_d, FWHT, multiply by
            // fwht_signs
            std::vector<float> padded(padded_d, 0.0f);
            float scale = 1.0f / std::sqrt(static_cast<float>(padded_d));
            for (size_t j = 0; j < d; j++) {
                padded[j] = signs[j] * scale;
            }
            fwht_inplace(padded.data(), padded_d);
            for (size_t j = 0; j < d; j++) {
                x_qjl[j] = qjl_recon_coeff * gamma * padded[j] * fwht_signs[j];
            }
        } else {
            // Dense matrix transpose: S^T * signs
            for (size_t j = 0; j < d; j++) {
                float sum = 0.0f;
                for (size_t k = 0; k < d; k++) {
                    sum += gaussian_matrix[k * d + j] * signs[k];
                }
                x_qjl[j] = qjl_recon_coeff * gamma * sum;
            }
        }

        // Step 3: Combine: x = norm * (x_mse + x_qjl)
        for (size_t j = 0; j < d; j++) {
            x_out[j] = norm * (x_mse[j] + x_qjl[j]);
        }
    }
}

// =========================================================================
// Distance Computer
// =========================================================================

namespace {

/// Template distance computer for TurboQ, parameterized on SIMD level.
template <SIMDLevel SL>
struct TurboQDistanceComputerImpl : TurboQDistanceComputer {
    TurboQDistanceComputerImpl() {
        codes = nullptr;
        code_size = 0;
    }

    float symmetric_dis(idx_t /*i*/, idx_t /*j*/) override {
        FAISS_THROW_MSG("TurboQ symmetric_dis not implemented");
    }

    void set_query(const float* x) override {
        FAISS_ASSERT(x != nullptr);
        FAISS_ASSERT(tq != nullptr);

        const size_t dim = tq->d;
        const size_t mse_bits = (tq->nb_bits > 1) ? (tq->nb_bits - 1) : 0;
        const float id = 1.0f / std::sqrt(static_cast<float>(dim));
        inv_sqrt_d = id;
        qjl_coeff = std::sqrt(static_cast<float>(M_PI) / 2.0f) /
                static_cast<float>(dim);

        // Store query
        q.resize(dim);
        std::copy(x, x + dim, q.begin());
        q_norm_sq = fvec_norm_L2sqr(x, dim);

        // Project query: q_proj = S * q
        q_proj.resize(dim);
        tq->apply_qjl_projection(x, q_proj.data());

        // Pre-compute total sums
        total_q_sum = 0.0f;
        for (size_t j = 0; j < dim; j++) {
            total_q_sum += q[j];
        }

        total_qproj_sum = 0.0f;
        for (size_t j = 0; j < dim; j++) {
            total_qproj_sum += q_proj[j];
        }

        // Pre-compute scaled centroids for each quantization level
        if (mse_bits > 0 && tq->mse_levels > 0) {
            scaled_centroids.resize(tq->mse_levels);
            for (size_t k = 0; k < tq->mse_levels; k++) {
                scaled_centroids[k] = tq->mse_centroids[k] * inv_sqrt_d;
            }

            // For 1-bit MSE (nb_bits==2, uniform): compute delta_centroid
            if (tq->nb_bits == 2 && !tq->is_adaptive()) {
                delta_centroid = scaled_centroids[1] - scaled_centroids[0];
            }

            // Lo codebook scaled centroids (adaptive)
            if (tq->is_adaptive() && tq->mse_levels_lo > 0) {
                scaled_centroids_lo.resize(tq->mse_levels_lo);
                for (size_t k = 0; k < tq->mse_levels_lo; k++) {
                    scaled_centroids_lo[k] =
                            tq->mse_centroids_lo[k] * inv_sqrt_d;
                }
            }
        }

        // Pre-screening: worst-case L1 bound on QJL error
        {
            float qproj_l1 = 0.0f;
            for (size_t j = 0; j < dim; j++) {
                qproj_l1 += std::abs(q_proj[j]);
            }
            qjl_error_coeff = qjl_coeff * qproj_l1;
        }

        // Integer popcount state (if qb > 0, nb_bits == 2, uniform)
        if (qb > 0 && tq->nb_bits == 2 && !tq->is_adaptive()) {
            // Quantize query to qb bits for integer popcount path
            size_t byte_size = (dim + 7) / 8;

            // Scale q by scaled_centroids for MSE accumulation
            // For 1-bit MSE: the MSE dot product is
            //   sum_j q[j] * centroid[idx_j] / sqrt(d)
            // = scaled_centroids[0] * sum(q) + delta_centroid * sum(q[j] where
            // bit=1)
            //
            // For integer path: quantize q into qb-bit values,
            // rearrange into bit-planes for popcount
            float q_min = *std::min_element(q.begin(), q.end());
            float q_max = *std::max_element(q.begin(), q.end());
            float q_range = q_max - q_min;
            if (q_range < 1e-30f) {
                q_range = 1e-30f;
            }
            float max_val = static_cast<float>((1 << qb) - 1);

            rearranged_q.resize(byte_size * qb);
            std::fill(rearranged_q.begin(), rearranged_q.end(), 0);

            float scale = max_val / q_range;
            // Quantize and rearrange into bit-planes
            for (size_t j = 0; j < dim; j++) {
                int qval = static_cast<int>(std::round((q[j] - q_min) * scale));
                qval = std::max(0, std::min(static_cast<int>(max_val), qval));

                for (int b = 0; b < qb; b++) {
                    if (qval & (1 << b)) {
                        rearranged_q[b * byte_size + j / 8] |= (1 << (j % 8));
                    }
                }
            }

            // Pre-compute base/scale for converting popcount to MSE dot
            // mse_dot ≈ base + int_scale * and_dot + popcnt_scale * popcount
            float delta_q = q_range / max_val;
            mse_base = scaled_centroids[0] * total_q_sum;
            mse_int_scale = delta_centroid * delta_q;
            mse_popcnt_scale = delta_centroid * q_min;
        }

        // Integer QJL state (if qb > 0, int_qjl)
        if (qb > 0 && int_qjl) {
            size_t byte_size = (dim + 7) / 8;

            float qp_min = *std::min_element(q_proj.begin(), q_proj.end());
            float qp_max = *std::max_element(q_proj.begin(), q_proj.end());
            float qp_range = qp_max - qp_min;
            if (qp_range < 1e-30f) {
                qp_range = 1e-30f;
            }
            float max_val = static_cast<float>((1 << qb) - 1);
            float scale = max_val / qp_range;

            rearranged_qproj.resize(byte_size * qb);
            std::fill(rearranged_qproj.begin(), rearranged_qproj.end(), 0);

            for (size_t j = 0; j < dim; j++) {
                int qval = static_cast<int>(
                        std::round((q_proj[j] - qp_min) * scale));
                qval = std::max(0, std::min(static_cast<int>(max_val), qval));

                for (int b = 0; b < qb; b++) {
                    if (qval & (1 << b)) {
                        rearranged_qproj[b * byte_size + j / 8] |=
                                (1 << (j % 8));
                    }
                }
            }

            // Pre-compute coefficients for integer QJL dot product
            // raw_and = sum over j: q_quantized[j] * sign_bit[j]
            // real_dot approx= (raw_and / scale + qp_min * popcount(signs))
            // but we fold this into: qjl_dot = qjl_base + qjl_and_scale *
            // and_result + qjl_pop_scale * popcount
            qjl_and_scale = 2.0f * qjl_coeff / scale;
            qjl_pop_scale = 2.0f * qjl_coeff * qp_min;
            qjl_base = -qjl_coeff * total_qproj_sum;
        }
    }

    float distance_to_code(const uint8_t* code) override {
        return turboq_distance_to_code(code);
    }

    float turboq_distance_to_code(const uint8_t* code) {
        FAISS_ASSERT(tq != nullptr);
        FAISS_ASSERT(code != nullptr);

        const size_t dim = tq->d;
        const size_t mse_bits = (tq->nb_bits > 1) ? (tq->nb_bits - 1) : 0;
        const size_t mse_total = tq->mse_code_size();
        const size_t qjl_size = tq->qjl_code_size();
        const size_t hi_dims = tq->is_adaptive() ? tq->n_hi_dims : dim;

        // Extract TurboQFactors
        const TurboQFactors* factors = reinterpret_cast<const TurboQFactors*>(
                code + mse_total + qjl_size);
        float norm = factors->norm;
        float gamma = factors->gamma;

        n_total++;

        // ============================================================
        // Stage 1: MSE dot product
        // ============================================================
        float mse_dot = 0.0f;

        if (mse_bits > 0) {
            const uint8_t* mse_code = code;

            if (tq->nb_bits == 2 && !tq->is_adaptive()) {
                // 1-bit MSE (nb_bits == 2, uniform mode)
                size_t byte_size = (dim + 7) / 8;

                if (qb > 0) {
                    // Integer popcount path
                    uint64_t and_result = rabitq::bitwise_and_dot_product<SL>(
                            rearranged_q.data(), mse_code, byte_size, qb);
                    uint64_t pop = rabitq::popcount<SL>(mse_code, byte_size);
                    mse_dot = mse_base +
                            mse_int_scale * static_cast<float>(and_result) +
                            mse_popcnt_scale * static_cast<float>(pop);
                } else {
                    // Float masked_sum path
                    float masked =
                            turboq::masked_sum<SL>(q.data(), mse_code, dim);
                    mse_dot = scaled_centroids[0] * total_q_sum +
                            delta_centroid * masked;
                }
            } else {
                // Multi-bit or adaptive: scalar bit-plane decoding
                size_t hi_byte_plane = (hi_dims + 7) / 8;

                for (size_t j = 0; j < hi_dims; j++) {
                    size_t idx = 0;
                    for (size_t b = 0; b < mse_bits; b++) {
                        bool bit = (mse_code[b * hi_byte_plane + j / 8] &
                                    (1 << (j % 8))) != 0;
                        if (bit) {
                            idx |= (size_t(1) << b);
                        }
                    }
                    mse_dot += scaled_centroids[idx] * q[j];
                }

                // Adaptive lo dimensions
                if (tq->is_adaptive()) {
                    const size_t mse_bits_lo_val =
                            (tq->nb_bits_lo > 1) ? (tq->nb_bits_lo - 1) : 0;
                    if (mse_bits_lo_val > 0) {
                        size_t lo_dims = dim - tq->n_hi_dims;
                        size_t lo_byte_plane = (lo_dims + 7) / 8;
                        const uint8_t* mse_lo_code =
                                mse_code + tq->mse_code_size_hi();

                        for (size_t jj = 0; jj < lo_dims; jj++) {
                            size_t j = tq->n_hi_dims + jj;
                            size_t idx = 0;
                            for (size_t b = 0; b < mse_bits_lo_val; b++) {
                                bool bit =
                                        (mse_lo_code
                                                 [b * lo_byte_plane + jj / 8] &
                                         (1 << (jj % 8))) != 0;
                                if (bit) {
                                    idx |= (size_t(1) << b);
                                }
                            }
                            mse_dot += scaled_centroids_lo[idx] * q[j];
                        }
                    }
                }
            }
        }

        // Pre-screening: check if MSE-only distance +/- bound can beat
        // threshold
        if (threshold_ptr != nullptr) {
            float bound = qjl_error_coeff * gamma * norm;
            float mse_ip = norm * mse_dot;

            if (tq->metric_type == MetricType::METRIC_INNER_PRODUCT) {
                // IP max-heap: threshold is the worst kept result.
                // Best possible IP = mse_ip + bound.
                // If best possible <= threshold, can't beat it.
                if (mse_ip + bound <= *threshold_ptr) {
                    n_skipped++;
                    return mse_ip;
                }
            } else {
                // L2 (min-heap): want distance < threshold
                // best possible = ||q||^2 + norm^2 - 2*(mse_ip + bound)
                float best_possible =
                        q_norm_sq + norm * norm - 2.0f * (mse_ip + bound);
                if (best_possible >= *threshold_ptr) {
                    n_skipped++;
                    return q_norm_sq + norm * norm - 2.0f * mse_ip;
                }
            }
        }

        // ============================================================
        // Stage 2: QJL dot product
        // ============================================================
        float qjl_dot = 0.0f;
        const uint8_t* qjl_code = code + mse_total;

        if (qb > 0 && int_qjl) {
            // Integer popcount path for QJL
            size_t byte_size = (dim + 7) / 8;
            uint64_t and_result = rabitq::bitwise_and_dot_product<SL>(
                    rearranged_qproj.data(), qjl_code, byte_size, qb);
            uint64_t pop = rabitq::popcount<SL>(qjl_code, byte_size);
            qjl_dot = gamma *
                    (qjl_base + qjl_and_scale * static_cast<float>(and_result) +
                     qjl_pop_scale * static_cast<float>(pop));
        } else {
            // Float masked_sum path
            float masked = turboq::masked_sum<SL>(q_proj.data(), qjl_code, dim);
            float raw_dot = 2.0f * masked - total_qproj_sum;
            qjl_dot = qjl_coeff * gamma * raw_dot;
        }

        // ============================================================
        // Combined distance
        // ============================================================
        float estimated_ip = norm * (mse_dot + qjl_dot);

        if (tq->metric_type == MetricType::METRIC_INNER_PRODUCT) {
            return estimated_ip;
        } else {
            return q_norm_sq + norm * norm - 2.0f * estimated_ip;
        }
    }
};

} // anonymous namespace

// =========================================================================
// TurboQDistanceComputer non-template methods
// =========================================================================

float TurboQDistanceComputer::symmetric_dis(idx_t /*i*/, idx_t /*j*/) {
    FAISS_THROW_MSG("TurboQ symmetric_dis not implemented");
}

void TurboQDistanceComputer::set_query(const float* /*x*/) {
    FAISS_THROW_MSG(
            "TurboQDistanceComputer::set_query should not be called directly; "
            "use the templatized implementation");
}

float TurboQDistanceComputer::distance_to_code(const uint8_t* /*code*/) {
    FAISS_THROW_MSG(
            "TurboQDistanceComputer::distance_to_code should not be called "
            "directly; use the templatized implementation");
}

FlatCodesDistanceComputer* TurboQuantizer::get_distance_computer() const {
    return with_selected_simd_levels<AVAILABLE_SIMD_LEVELS_A0>(
            [&]<SIMDLevel SL>() -> FlatCodesDistanceComputer* {
                auto dc = std::make_unique<TurboQDistanceComputerImpl<SL>>();
                dc->tq = this;
                dc->codes = nullptr;
                dc->code_size = code_size;
                return dc.release();
            });
}

} // namespace faiss
