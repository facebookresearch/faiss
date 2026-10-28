/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

/* QT_8bit reconstruction coefficients.
 *
 * Scalar and portable. A caller on any architecture may include this.
 *
 * For a QT_8bit non-uniform quantizer with trained values vmin and vdiff:
 *
 *     a[i]    = vdiff[i] / 255
 *     rmin[i] = vmin[i] + 0.5 * a[i]
 *     e[i]    = query[i] - rmin[i]
 *
 * so reconstruct(code)[i] = rmin[i] + a[i] * code[i].
 */

#include <cstddef>

namespace faiss {
namespace sq8 {

/// Derive the per-dimension scale and offset from a trained quantizer.
/// @param trained  2*d floats: vmin[0..d) then vdiff[0..d)
/// @param levels   the largest code value: 255 for QT_8bit, 15 for QT_4bit.
///                 Both quantizers reconstruct as rmin[i] + a[i] * code[i],
///                 so only the divisor differs.
inline void make_coefficients(
        const float* trained,
        size_t d,
        float* a,
        float* rmin,
        float levels = 255.0f) {
    const float* vmin = trained;
    const float* vdiff = trained + d;
    for (size_t i = 0; i < d; i++) {
        a[i] = vdiff[i] / levels;
        rmin[i] = vmin[i] + 0.5f * a[i];
    }
}

} // namespace sq8
} // namespace faiss
