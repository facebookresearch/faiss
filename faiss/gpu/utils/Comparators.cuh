/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cuda.h>
#include <faiss/gpu/utils/Float16.cuh>

namespace faiss {
namespace gpu {

template <typename T>
struct Comparator {
    __device__ static inline bool lt(T a, T b) {
        return a < b;
    }

    __device__ static inline bool gt(T a, T b) {
        return a > b;
    }

    template <typename V>
    __device__ static inline bool lt(T a, V, T b, V) {
        return a < b;
    }

    template <typename V>
    __device__ static inline bool gt(T a, V, T b, V) {
        return a > b;
    }
};

/// Orders on the key, then on the value, as `CMin::cmp2` does in
/// `utils/ordered_key_value.h`.
///
/// Do not select over padded entries. A pad ties with the init key, so the
/// value decides the order and the pad can reach the output.
template <typename T>
struct TieBreakComparator {
    __device__ static inline bool lt(T a, T b) {
        return a < b;
    }

    __device__ static inline bool gt(T a, T b) {
        return a > b;
    }

    template <typename V>
    __device__ static inline bool lt(T a, V av, T b, V bv) {
        return (a < b) || ((a == b) && (av < bv));
    }

    template <typename V>
    __device__ static inline bool gt(T a, V av, T b, V bv) {
        return (a > b) || ((a == b) && (av > bv));
    }
};

template <>
struct Comparator<half> {
    __device__ static inline bool lt(half a, half b) {
#if FAISS_USE_FULL_FLOAT16
        return __hlt(a, b);
#else
        return __half2float(a) < __half2float(b);
#endif // FAISS_USE_FULL_FLOAT16
    }

    __device__ static inline bool gt(half a, half b) {
#if FAISS_USE_FULL_FLOAT16
        return __hgt(a, b);
#else
        return __half2float(a) > __half2float(b);
#endif // FAISS_USE_FULL_FLOAT16
    }

    template <typename V>
    __device__ static inline bool lt(half a, V, half b, V) {
        return lt(a, b);
    }

    template <typename V>
    __device__ static inline bool gt(half a, V, half b, V) {
        return gt(a, b);
    }
};

template <>
struct TieBreakComparator<half> {
    __device__ static inline bool lt(half a, half b) {
        return Comparator<half>::lt(a, b);
    }

    __device__ static inline bool gt(half a, half b) {
        return Comparator<half>::gt(a, b);
    }

    template <typename V>
    __device__ static inline bool lt(half a, V av, half b, V bv) {
        return Comparator<half>::lt(a, b) ||
                (!Comparator<half>::gt(a, b) && av < bv);
    }

    template <typename V>
    __device__ static inline bool gt(half a, V av, half b, V bv) {
        return Comparator<half>::gt(a, b) ||
                (!Comparator<half>::lt(a, b) && av > bv);
    }
};

} // namespace gpu
} // namespace faiss
