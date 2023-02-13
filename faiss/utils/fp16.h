/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>

#include <faiss/impl/platform_macros.h>

#if defined(__F16C__)
#include <faiss/utils/fp16-fp16c.h>
#else
#include <faiss/utils/fp16-inl.h>
#endif
