/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <optional>
#include <unordered_set>
#include <vector>

#include <faiss/MetricType.h>

/** IDSelector is intended to define a subset of vectors to handle (for removal
 * or as subset to search) */

namespace faiss {

/** Encapsulates a set of ids to handle. */
struct IDSelector {
    virtual bool is_member(idx_t id) const {
        (void)id;
        return true;
    }
    virtual bool is_member(idx_t id, std::optional<float> d) const {
        (void)d;
        return is_member(id);
    }
    virtual ~IDSelector() {}
};

/** ids between [imin, imax) */
struct IDSelectorRange : IDSelector {
    idx_t imin, imax;

    /// Assume that the ids to handle are sorted. In some cases this can speed
    /// up processing
    bool assume_sorted;

    IDSelectorRange(idx_t imin, idx_t imax, bool assume_sorted = false);

    bool is_member(idx_t id) const final {
        return is_member(id, std::nullopt);
    }
    bool is_member(idx_t id, std::optional<float> d) const final;

    /// for sorted ids, find the range of list indices where the valid ids are
    /// stored
    void find_sorted_ids_bounds(
            size_t list_size,
            const idx_t* ids,
            size_t* jmin,
            size_t* jmax) const;

    ~IDSelectorRange() override {}
};

/** Simple array of elements
 *
 * is_member calls are very inefficient, but some operations can use the ids
 * directly.
 */
struct IDSelectorArray : IDSelector {
    size_t n;
    const idx_t* ids;

    /** Construct with an array of ids to process
     *
     * @param n number of ids to store
     * @param ids elements to store. The pointer should remain valid during
     *            IDSelectorArray's lifetime
     */
    IDSelectorArray(size_t n, const idx_t* ids);
    bool is_member(idx_t id) const final {
        return is_member(id, std::nullopt);
    }
    bool is_member(idx_t id, std::optional<float> d) const final;
    ~IDSelectorArray() override {}
};

/** Ids from a set.
 *
 * Repetitions of ids in the indices set passed to the constructor does not hurt
 * performance.
 *
 * The hash function used for the bloom filter and GCC's implementation of
 * unordered_set are just the least significant bits of the id. This works fine
 * for random ids or ids in sequences but will produce many hash collisions if
 * lsb's are always the same
 */
struct IDSelectorBatch : IDSelector {
    std::unordered_set<idx_t> set;

    // Bloom filter to avoid accessing the unordered set if it is unlikely
    // to be true
    std::vector<uint8_t> bloom;
    int nbits;
    idx_t mask;

    /** Construct with an array of ids to process
     *
     * @param n number of ids to store
     * @param ids elements to store. The pointer can be released after
     *            construction
     */
    IDSelectorBatch(size_t n, const idx_t* indices);
    bool is_member(idx_t id) const final {
        return is_member(id, std::nullopt);
    }
    bool is_member(idx_t id, std::optional<float> d) const final;
    ~IDSelectorBatch() override {}
};

/** One bit per element. Constructed with a bitmap, size ceil(n / 8).
 */
struct IDSelectorBitmap : IDSelector {
    size_t n;
    const uint8_t* bitmap;

    /** Construct with a binary mask
     *
     * @param n size of the bitmap array
     * @param bitmap id will be selected iff id / 8 < n and bit number
     *               (i%8) of bitmap[floor(i / 8)] is 1.
     */
    IDSelectorBitmap(size_t n, const uint8_t* bitmap);
    bool is_member(idx_t id) const final {
        return is_member(id, std::nullopt);
    }
    bool is_member(idx_t id, std::optional<float> d) const final;
    ~IDSelectorBitmap() override {}
};

/** reverts the membership test of another selector */
struct IDSelectorNot : IDSelector {
    const IDSelector* sel;
    IDSelectorNot(const IDSelector* sel) : sel(sel) {}
    bool is_member(idx_t id) const final {
        return !sel->is_member(id);
    }
    bool is_member(idx_t id, std::optional<float> d) const final {
        return !sel->is_member(id, d);
    }
    virtual ~IDSelectorNot() {}
};

/// selects all entries (useful for benchmarking)
struct IDSelectorAll : IDSelector {
    bool is_member(idx_t id) const final {
        (void)id;
        return true;
    }
    bool is_member(idx_t id, std::optional<float> d) const final {
        (void)id;
        (void)d;
        return true;
    }
    virtual ~IDSelectorAll() {}
};

/// does an AND operation on the the two given IDSelector's is_membership
/// results.
struct IDSelectorAnd : IDSelector {
    const IDSelector* lhs;
    const IDSelector* rhs;
    IDSelectorAnd(const IDSelector* lhs, const IDSelector* rhs)
            : lhs(lhs), rhs(rhs) {}
    bool is_member(idx_t id) const final {
        return lhs->is_member(id) && rhs->is_member(id);
    }
    bool is_member(idx_t id, std::optional<float> d) const final {
        return lhs->is_member(id, d) && rhs->is_member(id, d);
    }
    virtual ~IDSelectorAnd() {}
};

/// does an OR operation on the the two given IDSelector's is_membership
/// results.
struct IDSelectorOr : IDSelector {
    const IDSelector* lhs;
    const IDSelector* rhs;
    IDSelectorOr(const IDSelector* lhs, const IDSelector* rhs)
            : lhs(lhs), rhs(rhs) {}
    bool is_member(idx_t id) const final {
        return lhs->is_member(id) || rhs->is_member(id);
    }
    bool is_member(idx_t id, std::optional<float> d) const final {
        return lhs->is_member(id, d) || rhs->is_member(id, d);
    }
    virtual ~IDSelectorOr() {}
};

/// does an XOR operation on the the two given IDSelector's is_membership
/// results.
struct IDSelectorXOr : IDSelector {
    const IDSelector* lhs;
    const IDSelector* rhs;
    IDSelectorXOr(const IDSelector* lhs, const IDSelector* rhs)
            : lhs(lhs), rhs(rhs) {}
    bool is_member(idx_t id) const final {
        return lhs->is_member(id) ^ rhs->is_member(id);
    }
    bool is_member(idx_t id, std::optional<float> d) const final {
        return lhs->is_member(id, d) ^ rhs->is_member(id, d);
    }
    virtual ~IDSelectorXOr() {}
};

} // namespace faiss
