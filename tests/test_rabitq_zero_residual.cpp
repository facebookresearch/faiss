/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/RaBitQUtils.h>
#include <faiss/impl/RaBitQuantizer.h>
#include <faiss/utils/rabitq_simd.h>
#include <faiss/utils/simd_levels.h>

using faiss::MetricType;
using faiss::RaBitQuantizer;
using faiss::SIMDLevel;
using faiss::rabitq_utils::SignBitFactorsWithError;

namespace {

float centered_code_at(
        const uint8_t* code,
        size_t d,
        size_t ex_bits,
        size_t j) {
    const float cb = -(static_cast<float>(1 << ex_bits) - 0.5f);
    const uint8_t* ex_code =
            code + (d + 7) / 8 + sizeof(SignBitFactorsWithError);

    std::vector<float> onehot(d, 0.0f);
    onehot[j] = 1.0f;

    return faiss::rabitq::multibit::compute_inner_product<SIMDLevel::NONE>(
            code, ex_code, onehot.data(), d, ex_bits, cb);
}

// Deterministic vector with no zero coordinates
std::vector<float> base_vector(size_t d) {
    std::vector<float> x(d);
    for (size_t i = 0; i < d; i++) {
        x[i] = std::sin(0.7f * static_cast<float>(i) + 1.0f);
        if (x[i] == 0.0f) {
            x[i] = 0.5f;
        }
    }
    return x;
}

} // namespace

TEST(RaBitQZeroResidual, ZeroCoordinateEncodesToSmallestMagnitude) {
    constexpr size_t d = 64;
    const std::vector<size_t> zero_dims = {0, 17, 33, 63};

    for (size_t nb_bits = 2; nb_bits <= 9; nb_bits++) {
        const size_t ex_bits = nb_bits - 1;

        std::vector<float> x = base_vector(d);
        for (size_t j : zero_dims) {
            x[j] = 0.0f;
        }

        RaBitQuantizer rq(d, MetricType::METRIC_L2, nb_bits);
        std::vector<uint8_t> code(rq.code_size);
        rq.compute_codes(x.data(), code.data(), 1);

        for (size_t j : zero_dims) {
            const float centered = centered_code_at(code.data(), d, ex_bits, j);
            EXPECT_FLOAT_EQ(std::fabs(centered), 0.5f)
                    << "nb_bits=" << nb_bits << " dim=" << j
                    << ": zero residual encoded as centered code " << centered
                    << ", expected +/-0.5";
        }
    }
}

TEST(RaBitQZeroResidual, ZeroCoordinateDoesNotInflateReconstructionNorm) {
    constexpr size_t d = 64;
    constexpr size_t nb_bits = 8;
    constexpr size_t ex_bits = nb_bits - 1;

    std::vector<float> x = base_vector(d);
    x[17] = 0.0f;

    RaBitQuantizer rq(d, MetricType::METRIC_L2, nb_bits);
    std::vector<uint8_t> code(rq.code_size);
    rq.compute_codes(x.data(), code.data(), 1);

    const auto* ex_fac = reinterpret_cast<const faiss::rabitq_utils::ExtraBitsFactors*>(
            code.data() + (d + 7) / 8 + sizeof(SignBitFactorsWithError) +
            (d * ex_bits + 7) / 8);
    const double c = ex_fac->f_rescale_ex / -2.0;

    double norm_sqr = 0;
    double recon_sqr = 0;
    for (size_t j = 0; j < d; j++) {
        const double q = centered_code_at(code.data(), d, ex_bits, j);
        norm_sqr += static_cast<double>(x[j]) * x[j];
        recon_sqr += c * c * q * q;
    }

    const double tan_sq = (recon_sqr / norm_sqr) - 1.0;
    EXPECT_LT(tan_sq, 1e-2)
            << "reconstruction norm inflated by a zero-residual coordinate, tan^2 = "
            << std::to_string(tan_sq);
}

TEST(RaBitQZeroResidual, ExactZeroMatchesInfinitesimalNeighbours) {
    constexpr size_t d = 64;
    constexpr size_t nb_bits = 8;

    std::vector<float> q(d);
    for (size_t i = 0; i < d; i++) {
        q[i] = std::cos(0.3f * static_cast<float>(i) + 0.2f);
    }

    double rel_err[3] = {0, 0, 0};
    const float eps[3] = {0.0f, 1e-20f, -1e-20f};

    for (int t = 0; t < 3; t++) {
        std::vector<float> x = base_vector(d);
        x[17] = eps[t];

        RaBitQuantizer rq(d, MetricType::METRIC_L2, nb_bits);
        std::vector<uint8_t> code(rq.code_size);
        rq.compute_codes(x.data(), code.data(), 1);

        std::unique_ptr<faiss::FlatCodesDistanceComputer> dc(
                rq.get_distance_computer(0));
        dc->codes = code.data();
        dc->code_size = rq.code_size;
        dc->set_query(q.data());
        const double est = (*dc)(0);

        double truth = 0;
        for (size_t j = 0; j < d; j++) {
            truth += static_cast<double>(q[j] - x[j]) * (q[j] - x[j]);
        }
        rel_err[t] = std::fabs(est - truth) / truth;
    }

    // The two infinitesimal cases bracket what accuracy is achievable here; the exact
    // zero must not be dramatically worse.
    const double reference = std::max(rel_err[1], rel_err[2]);
    EXPECT_LT(rel_err[0], 10 * reference + 1e-6)
            << "exact zero relative error " << std::to_string(rel_err[0])
            << " vs " << std::to_string(rel_err[1]) << " / "
            << std::to_string(rel_err[2]) << " for +/-1e-20";
}
