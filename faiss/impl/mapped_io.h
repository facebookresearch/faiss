/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

#include <faiss/impl/io.h>
#include <faiss/impl/maybe_owned_vector.h>

namespace faiss {

// holds a memory-mapped region over a file
struct MmappedFileMappingOwner : public MaybeOwnedVectorOwner {
    MmappedFileMappingOwner(const std::string& filename);
    MmappedFileMappingOwner(FILE* f);
    ~MmappedFileMappingOwner();

    void* data() const;
    size_t size() const;

    struct PImpl;
    std::unique_ptr<PImpl> p_impl;
};

// A deserializer that supports memory-mapped files.
// All de-allocations should happen as soon as the index gets destroyed,
//   after all underlying the MaybeOwnerVector objects are destroyed.
struct MappedFileIOReader : IOReader {
    std::shared_ptr<MmappedFileMappingOwner> mmap_owner;

    size_t pos = 0;

    MappedFileIOReader(const std::shared_ptr<MmappedFileMappingOwner>& owner);

    // perform a copy
    size_t operator()(void* ptr, size_t size, size_t nitems) override;
    // perform a quasi-read that returns a mmapped address, owned by mmap_owner,
    //   and updates the position
    size_t mmap(void** ptr, size_t size, size_t nitems);

    int filedescriptor() override;
};

} // namespace faiss
