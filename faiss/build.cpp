/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "faiss/build.h"

namespace faiss {

bool has_omp() {
    int omp = 1;
    // Detect whether OpenMP is enabled by using the 'max' reduction to render
    // the below assignment a no-op. This works:
    //  1) without starting any threads
    //  2) irrespective of the current thread limit
#pragma omp parallel reduction(max : omp) num_threads(1)
    omp = 0;
    return omp != 0;
}

} // namespace faiss
