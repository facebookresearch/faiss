/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/hnsw/LockVector.h>

#include <cstdlib>

#include <utility>

namespace faiss {

LockVector::LockVector(LockVector&& other) noexcept
        : data_(other.data_), size_(other.size_), capacity_(other.capacity_) {
    other.data_ = nullptr;
    other.size_ = 0;
    other.capacity_ = 0;
}

void LockVector::prepare(size_t new_size) {
    if (new_size <= size_) {
        return;
    }
    if (new_size > capacity_) {
        // Ensure geometric capacity growth.
        size_t new_cap = std::max(new_size, capacity_ * 2);
        // Just destroy old and init fresh; omp_lock_t is not copyable.
        clear();
        data_ = static_cast<omp_lock_t*>(malloc(new_cap * sizeof(omp_lock_t)));
        FAISS_THROW_IF_NOT(data_ != nullptr);
        capacity_ = new_cap;
    }
    for (size_t i = size_; i < new_size; i++) {
        omp_init_lock(&data_[i]);
    }
    size_ = new_size;
}

void LockVector::clear() {
    if (data_) {
        for (size_t i = 0; i < size_; i++) {
            omp_destroy_lock(&data_[i]);
        }
        free(data_);
        data_ = nullptr;
    }
    size_ = 0;
    capacity_ = 0;
}

} // namespace faiss
