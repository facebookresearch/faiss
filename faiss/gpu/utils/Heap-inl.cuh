/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * C++ support for heaps in GPU. The set of functions is tailored for efficient
 * similarity search.
 *
 * There is no specific object for a heap, and the functions that operate on a
 * single heap are inlined, because heaps are often small.
 */

#pragma once

namespace faiss {
namespace gpu {

template <typename T, typename VecT>
__device__ inline void heap_pop(T& size, float* values, VecT* ids) {
    values--;
    ids--;
    float val = values[size];
    int i = 1, i1, i2;
    i1 = i << 1;
    while (i1 <= size) {
        i2 = i1 + 1;
        if (i2 == size + 1 || values[i1] < values[i2]) {
            if (val < values[i1])
                break;
            values[i] = values[i1];
            ids[i] = ids[i1];
            i = i1;
        } else {
            if (val < values[i2])
                break;
            values[i] = values[i2];
            ids[i] = ids[i2];
            i = i2;
        }
        i1 = i << 1;
    }
    values[i] = values[size];
    ids[i] = ids[size];
    size--;
}

template <typename T, typename VecT>
__device__ inline void heap_push(
        T& size,
        float* values,
        VecT* ids,
        float val,
        T id1,
        T id2) {
    size++;
    values--;
    ids--;
    unsigned i = size, i_father;
    while (i > 1) {
        i_father = i >> 1;
        if (val >= values[i_father])
            break;
        values[i] = values[i_father];
        ids[i] = ids[i_father];
        i = i_father;
    }
    values[i] = val;
    ids[i] = {id1, id2};
}

} // namespace gpu
} // namespace faiss
