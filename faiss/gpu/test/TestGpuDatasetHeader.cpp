/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Keep this translation unit ordinary C++ without CUDA, RAFT, or cuVS headers.
#include <faiss/gpu/Dataset.h>

#include <type_traits>

using faiss::gpu::Dataset;

static_assert(!std::is_copy_constructible_v<Dataset>);
static_assert(!std::is_copy_assignable_v<Dataset>);
static_assert(std::is_nothrow_move_constructible_v<Dataset>);
static_assert(std::is_nothrow_move_assignable_v<Dataset>);
