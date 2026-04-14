/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/RaBitQStats.h>

namespace faiss {

// NOLINTNEXTLINE(facebook-avoid-non-const-global-variables)
RaBitQStats rabitq_stats;

void RaBitQStats::reset() {
    n_1bit_evaluations = 0;
    n_multibit_evaluations = 0;
}

double RaBitQStats::skip_percentage() const {
    const size_t copy_n_1bit_evaluations = n_1bit_evaluations;
    const size_t copy_n_multibit_evaluations = n_multibit_evaluations;
    return copy_n_1bit_evaluations > 0
            ? 100.0 * (copy_n_1bit_evaluations - copy_n_multibit_evaluations) /
                    copy_n_1bit_evaluations
            : 0.0;
}

} // namespace faiss
