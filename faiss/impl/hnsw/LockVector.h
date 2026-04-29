/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <omp.h>

#include <faiss/impl/FaissAssert.h>

namespace faiss {

/// Contiguous, growable array of locks with geometric growth.
class LockVector {
   public:
    LockVector() = default;
    explicit LockVector(size_t n) {
        prepare(n);
    }
    // Copy ctor for clone(), initialized as empty.
    LockVector(const LockVector&) : LockVector() {}
    LockVector(LockVector&& other) noexcept;

    ~LockVector() {
        clear();
    }

    LockVector& operator=(const LockVector&) = delete;
    LockVector& operator=(LockVector&& other) = delete;

    size_t size() const {
        return size_;
    }

    // Ensure size is at least 'new_size'. No locks may be held.
    void prepare(size_t new_size);
    // Release all locks and free memory. No locks may be held.
    void clear();

    void lock(size_t i) {
        FAISS_CHECK_RANGE_DEBUG(i, 0, size_);
        omp_set_lock(&data_[i]);
    }

    void unlock(size_t i) {
        FAISS_CHECK_RANGE_DEBUG(i, 0, size_);
        omp_unset_lock(&data_[i]);
    }

    bool try_lock(size_t i) {
        FAISS_CHECK_RANGE_DEBUG(i, 0, size_);
        return omp_test_lock(&data_[i]);
    }

   private:
    omp_lock_t* data_ = nullptr;
    size_t size_ = 0;
    size_t capacity_ = 0;
};

} // namespace faiss
