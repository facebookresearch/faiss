/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/AdSampling.h>

#include <cmath>

#include <faiss/impl/FaissAssert.h>

namespace faiss {
namespace detail {

double normal_quantile(double p) {
    // Three-branch rational polynomial; branch breakpoint p_low = 0.02425.
    static constexpr double a[] = {
            -3.969683028665376e+01,
            2.209460984245205e+02,
            -2.759285104469687e+02,
            1.383577518672690e+02,
            -3.066479806614716e+01,
            2.506628277459239e+00,
    };
    static constexpr double b[] = {
            -5.447609879822406e+01,
            1.615858368580409e+02,
            -1.556989798598866e+02,
            6.680131188771972e+01,
            -1.328068155288572e+01,
    };
    static constexpr double c[] = {
            -7.784894002430293e-03,
            -3.223964580411365e-01,
            -2.400758277161838e+00,
            -2.549732539343734e+00,
            4.374664141464968e+00,
            2.938163982698783e+00,
    };
    static constexpr double d[] = {
            7.784695709041462e-03,
            3.224671290700398e-01,
            2.445134137142996e+00,
            3.754408661907416e+00,
    };
    constexpr double p_low = 0.02425;
    constexpr double p_high = 1.0 - p_low;
    if (p < p_low) {
        const double q = std::sqrt(-2.0 * std::log(p));
        return (((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q +
                c[5]) /
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    } else if (p <= p_high) {
        const double q = p - 0.5;
        const double r = q * q;
        return (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r +
                a[5]) *
                q /
                (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + b[4]) * r +
                 1.0);
    } else {
        const double q = std::sqrt(-2.0 * std::log(1.0 - p));
        return -(((((c[0] * q + c[1]) * q + c[2]) * q + c[3]) * q + c[4]) * q +
                 c[5]) /
                ((((d[0] * q + d[1]) * q + d[2]) * q + d[3]) * q + 1.0);
    }
}

double chi2_quantile_wh(int p, double alpha) {
    FAISS_THROW_IF_NOT(p > 0);
    // Wilson-Hilferty cube-root approximation:
    //   ((X/p)^(1/3) - (1 - 2/(9p))) / sqrt(2/(9p)) ~ N(0,1)
    // inverted into a quantile formula.
    //
    // Domain constraint: for very small alpha (< ~0.001) and small p
    // (< 4), t can go negative, producing a negative chi-squared quantile
    // (physically impossible). In practice this cannot happen here:
    // precompute_ad_thresholds calls with alpha = 1 - epsilon where
    // epsilon = ad_epsilon_factor / d, and d_prime_min >= 16, so
    // p >= 16 and alpha >= 1 - 1/16 = 0.9375 — well inside the accurate
    // region of the approximation.
    const double z = normal_quantile(alpha);
    const double t = 1.0 - 2.0 / (9.0 * p) + z * std::sqrt(2.0 / (9.0 * p));
    return p * t * t * t;
}

std::vector<float> precompute_ad_thresholds(int d, double epsilon) {
    FAISS_THROW_IF_NOT_MSG(
            epsilon > 0.0 && epsilon < 1.0,
            "precompute_ad_thresholds: epsilon must be in (0, 1)");
    FAISS_THROW_IF_NOT_MSG(
            d > 0, "precompute_ad_thresholds: d must be positive");
    std::vector<float> coeff(d + 1);
    for (int p = 1; p <= d; p++) {
        coeff[p] = static_cast<float>(chi2_quantile_wh(p, 1.0 - epsilon) / d);
    }
    return coeff;
}

} // namespace detail
} // namespace faiss
