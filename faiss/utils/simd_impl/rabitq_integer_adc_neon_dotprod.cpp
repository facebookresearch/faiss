/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/utils/rabitq_integer_adc.h>

#include <faiss/impl/FaissAssert.h>

#include <arm_neon.h>
#include <algorithm>
#include <cstring>
#include <type_traits>
#include <utility>

// Add only dot-product support; preserve the user's architecture settings.
#if defined(__clang__)
#pragma clang attribute push( \
        __attribute__((target("dotprod"))), apply_to = function)
#elif defined(__GNUC__)
#pragma GCC push_options
#pragma GCC target("+dotprod")
#endif

#define FAISS_RABITQ_NEON_NAMESPACE neon_dotprod
#define FAISS_RABITQ_NEON_DOTPROD
#include "rabitq_integer_adc_neon-inl.h"
#undef FAISS_RABITQ_NEON_DOTPROD
#undef FAISS_RABITQ_NEON_NAMESPACE

#if defined(__clang__)
#pragma clang attribute pop
#elif defined(__GNUC__)
#pragma GCC pop_options
#endif
