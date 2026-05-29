/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>

namespace faiss {
namespace detail {

/** Inverse standard normal CDF. Three-branch rational polynomial,
 * absolute error < 1.15e-9 over `p in (0, 1)`. Behavior at the boundaries
 * (p <= 0 or p >= 1) is unspecified — returns NaN or +/-inf. */
double normal_quantile(double p);

/** Chi-squared quantile via cube-root approximation. Validated to within
 * 2% of scipy for `p in [16, d]` and `alpha <= 1 - 1e-6`. Accuracy
 * degrades for smaller `p` or for `alpha` near 1. */
double chi2_quantile_wh(int p, double alpha);

/** Build ADSampling threshold table of size `d + 1`:
 *   coeff[p] = chi2_quantile_wh(p, 1 - epsilon) / d.
 *
 * Indexing: coeff[0] is reserved (left at 0.0f). coeff[1..15] are
 * computed but NOT accuracy-bounded — callers requiring the 2% scipy
 * tolerance must consume only coeff[16..d]. SuperKMeans enforces
 * this via its `d_prime_min = 16` parameter. */
std::vector<float> precompute_ad_thresholds(int d, double epsilon);

} // namespace detail
} // namespace faiss
