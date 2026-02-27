/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

namespace faiss {

// Returns true iff `faiss` was compiled with non-mocked OpenMP support.
bool has_omp();

} // namespace faiss
