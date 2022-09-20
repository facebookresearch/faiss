/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <unordered_set>
#include <vector>

#include <faiss/Index.h>

namespace faiss {

/** Encapsulates a set of ids to handle. */
struct IDSelector {
    typedef Index::idx_t idx_t;
    virtual bool is_member(idx_t id) const = 0;
    virtual ~IDSelector() {}
};

/** ids between [imni, imax) */
struct IDSelectorRange : IDSelector {
    idx_t imin, imax;

    IDSelectorRange(idx_t imin, idx_t imax);
    bool is_member(idx_t id) const override;
    ~IDSelectorRange() override {}
};

/** simple list of elements to remove
 *
 * this is inefficient in most cases, except for IndexIVF with
 * maintain_direct_map
 */
struct IDSelectorArray : IDSelector {
    size_t n;
    const idx_t* ids;

    IDSelectorArray(size_t n, const idx_t* ids);
    bool is_member(idx_t id) const override;
    ~IDSelectorArray() override {}
};

/** Ids from a set. Repetitions of ids in the indices set
 * passed to the constructor does not hurt performance. The hash
 * function used for the bloom filter and GCC's implementation of
 * unordered_set are just the least significant bits of the id. This
 * works fine for random ids or ids in sequences but will produce many
 * hash collisions if lsb's are always the same */
struct IDSelectorBatch : IDSelector {
    std::unordered_set<idx_t> set;

    typedef unsigned char uint8_t;
    std::vector<uint8_t> bloom; // assumes low bits of id are a good hash value
    int nbits;
    idx_t mask;

    IDSelectorBatch(size_t n, const idx_t* indices);
    bool is_member(idx_t id) const override;
    ~IDSelectorBatch() override {}
};

/// selects all entries (mainly useful for benchmarking)
struct IDSelectorAll : IDSelector {
    virtual bool is_member(idx_t id) const {
        return true;
    }
    virtual ~IDSelectorAll() {}
};

} // namespace faiss
