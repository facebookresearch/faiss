/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>

#include <faiss/impl/maybe_owned_vector.h>

namespace faiss {

// Owns uninitialized flat codes for a conversion that overwrites every byte.
// The owner also distinguishes this writable allocation from mmap-backed views.
struct OverwriteCodeBuffer final : MaybeOwnedVectorOwner {
    explicit OverwriteCodeBuffer(size_t n)
            : data(std::make_unique_for_overwrite<uint8_t[]>(n)) {}

    static MaybeOwnedVector<uint8_t> allocate(size_t n) {
        auto owner = std::make_shared<OverwriteCodeBuffer>(n);
        return MaybeOwnedVector<uint8_t>::create_view(
                owner->data.get(), n, owner);
    }

    static bool backs(const MaybeOwnedVector<uint8_t>& codes) {
        return !codes.is_owned &&
                dynamic_cast<const OverwriteCodeBuffer*>(codes.owner.get());
    }

    std::unique_ptr<uint8_t[]> data;
};

} // namespace faiss
