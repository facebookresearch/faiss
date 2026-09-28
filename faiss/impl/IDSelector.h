/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <unordered_set>
#include <vector>

#include <faiss/MetricType.h>

/** IDSelector is intended to define a subset of vectors to handle (for removal
 * or as subset to search) */

namespace faiss {

/** Encapsulates a set of ids to handle. */
struct IDSelector {
    virtual bool is_member(idx_t id) const = 0;
    virtual ~IDSelector() {}
};

/** Scan context handed to IDSelectorWithContext::is_member_with_context: the
 * contiguous id block being scanned and the position of the id under test. */
struct IDScanContext {
    /// the contiguous block of ids being scanned
    const idx_t* ids;
    /// number of entries in `ids`
    size_t list_size;
    /// index of the tested id within `ids` (i.e. ids[j] == id)
    size_t j;
};

/** IDSelector that also receives the surrounding scan context on each
 * membership test, letting an implementation exploit locality across a scan
 * (for example, prefetching data for an entry it will be asked about soon).
 * The policy is entirely up to the implementation. */
struct IDSelectorWithContext : IDSelector {
    virtual bool is_member_with_context(idx_t id, const IDScanContext& ctx)
            const = 0;
};

/** Routes each per-candidate membership test to is_member_with_context() when
 * the selector implements IDSelectorWithContext, else to plain is_member().
 * Construct one per inverted-list scan: the dynamic_cast is the only RTTI cost
 * and the per-candidate cost is a single predicted branch (cf. the
 * IDSelectorRange dynamic_cast in IndexIVF.cpp). The scan context is only
 * meaningful when the scan exposes a real id array (i.e. !store_pairs); when
 * store_pairs the context path is disabled and every test falls back to
 * is_member. */
struct IDSelectorContextDispatch {
    const IDSelector* sel;
    const IDSelectorWithContext* ctx_sel;

    IDSelectorContextDispatch(const IDSelector* sel, bool store_pairs)
            : sel(sel),
              ctx_sel((sel != nullptr && !store_pairs)
                              ? dynamic_cast<const IDSelectorWithContext*>(sel)
                              : nullptr) {}

    bool is_member(idx_t id, const IDScanContext& ctx) const {
        return ctx_sel ? ctx_sel->is_member_with_context(id, ctx)
                       : sel->is_member(id);
    }
};

/** ids between [imin, imax) */
struct IDSelectorRange : IDSelector {
    idx_t imin, imax;

    /// Assume that the ids to handle are sorted. In some cases this can speed
    /// up processing
    bool assume_sorted;

    IDSelectorRange(idx_t imin, idx_t imax, bool assume_sorted = false);

    bool is_member(idx_t id) const final;

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
    bool is_member(idx_t id) const final;
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
    bool is_member(idx_t id) const final;
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
    bool is_member(idx_t id) const final;
    ~IDSelectorBitmap() override {}
};

/** reverts the membership test of another selector */
struct IDSelectorNot : IDSelector {
    const IDSelector* sel;
    explicit IDSelectorNot(const IDSelector* sel_) : sel(sel_) {}
    bool is_member(idx_t id) const final {
        return !sel->is_member(id);
    }
    virtual ~IDSelectorNot() {}
};

/// selects all entries (useful for benchmarking)
struct IDSelectorAll : IDSelector {
    bool is_member(idx_t /* id */) const final {
        return true;
    }
    virtual ~IDSelectorAll() {}
};

/// does an AND operation on the two given IDSelector's is_membership
/// results.
struct IDSelectorAnd : IDSelector {
    const IDSelector* lhs;
    const IDSelector* rhs;
    IDSelectorAnd(const IDSelector* lhs_, const IDSelector* rhs_)
            : lhs(lhs_), rhs(rhs_) {}
    bool is_member(idx_t id) const final {
        return lhs->is_member(id) && rhs->is_member(id);
    }
    virtual ~IDSelectorAnd() {}
};

/// does an OR operation on the two given IDSelector's is_membership
/// results.
struct IDSelectorOr : IDSelector {
    const IDSelector* lhs;
    const IDSelector* rhs;
    IDSelectorOr(const IDSelector* lhs_, const IDSelector* rhs_)
            : lhs(lhs_), rhs(rhs_) {}
    bool is_member(idx_t id) const final {
        return lhs->is_member(id) || rhs->is_member(id);
    }
    virtual ~IDSelectorOr() {}
};

/// does an XOR operation on the two given IDSelector's is_membership
/// results.
struct IDSelectorXOr : IDSelector {
    const IDSelector* lhs;
    const IDSelector* rhs;
    IDSelectorXOr(const IDSelector* lhs_, const IDSelector* rhs_)
            : lhs(lhs_), rhs(rhs_) {}
    bool is_member(idx_t id) const final {
        return lhs->is_member(id) ^ rhs->is_member(id);
    }
    virtual ~IDSelectorXOr() {}
};

} // namespace faiss
