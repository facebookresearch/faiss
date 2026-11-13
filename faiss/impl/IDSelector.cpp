/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/platform_macros.h>

#include <algorithm>
#include <cstring>
#include <iterator>

namespace faiss {

namespace {

// The ids of [begin, end) that fall in [0, ntotal), sorted and deduplicated.
template <class It>
void sorted_members(It begin, It end, idx_t ntotal, std::vector<idx_t>& ids) {
    ids.clear();
    std::copy_if(begin, end, std::back_inserter(ids), [ntotal](idx_t id) {
        return id >= 0 && id < ntotal;
    });
    std::sort(ids.begin(), ids.end());
    ids.erase(std::unique(ids.begin(), ids.end()), ids.end());
}

// The members of `with` (sorted) that also satisfy `sel`.
void filter_members(
        const std::vector<idx_t>& with,
        const IDSelector& sel,
        std::vector<idx_t>& ids) {
    ids.clear();
    std::copy_if(
            with.begin(), with.end(), std::back_inserter(ids), [&](idx_t id) {
                return sel.is_member(id);
            });
}

} // namespace

/***********************************************************************
 * IDSelectorRange
 ***********************************************************************/

IDSelectorRange::IDSelectorRange(
        idx_t imin_in,
        idx_t imax_in,
        bool assume_sorted_in)
        : imin(imin_in), imax(imax_in), assume_sorted(assume_sorted_in) {}

bool IDSelectorRange::is_member(idx_t id) const {
    return id >= imin && id < imax;
}

bool IDSelectorRange::get_members(idx_t ntotal, std::vector<idx_t>& ids) const {
    const idx_t lo = std::max(imin, idx_t(0));
    const idx_t hi = std::min(imax, ntotal);
    ids.clear();
    for (idx_t id = lo; id < hi; id++) {
        ids.push_back(id);
    }
    return true;
}

void IDSelectorRange::find_sorted_ids_bounds(
        size_t list_size,
        const idx_t* ids,
        size_t* jmin_out,
        size_t* jmax_out) const {
    FAISS_ASSERT(assume_sorted);
    if (list_size == 0 || imax <= ids[0] || imin > ids[list_size - 1]) {
        *jmin_out = *jmax_out = 0;
        return;
    }
    // bisection to find imin
    if (ids[0] >= imin) {
        *jmin_out = 0;
    } else {
        size_t j0 = 0, j1 = list_size;
        while (j1 > j0 + 1) {
            size_t jmed = (j0 + j1) / 2;
            if (ids[jmed] >= imin) {
                j1 = jmed;
            } else {
                j0 = jmed;
            }
        }
        *jmin_out = j1;
    }
    // bisection to find imax
    if (*jmin_out == list_size || ids[*jmin_out] >= imax) {
        *jmax_out = *jmin_out;
    } else {
        size_t j0 = *jmin_out, j1 = list_size;
        while (j1 > j0 + 1) {
            size_t jmed = (j0 + j1) / 2;
            if (ids[jmed] >= imax) {
                j1 = jmed;
            } else {
                j0 = jmed;
            }
        }
        *jmax_out = j1;
    }
}

/***********************************************************************
 * IDSelectorArray
 ***********************************************************************/

IDSelectorArray::IDSelectorArray(size_t n_in, const idx_t* ids_in)
        : n(n_in), ids(ids_in) {}

bool IDSelectorArray::is_member(idx_t id) const {
    for (size_t i = 0; i < n; i++) {
        if (ids[i] == id) {
            return true;
        }
    }
    return false;
}

bool IDSelectorArray::get_members(idx_t ntotal, std::vector<idx_t>& out) const {
    sorted_members(ids, ids + n, ntotal, out);
    return true;
}

/***********************************************************************
 * IDSelectorBatch
 ***********************************************************************/

IDSelectorBatch::IDSelectorBatch(size_t n, const idx_t* indices) {
    nbits = 0;
    while (n > (size_t{1} << nbits)) {
        nbits++;
    }
    nbits += 5;
    // for n = 1M, nbits = 25 is optimal, see P56659518

    mask = ((idx_t)1 << nbits) - 1;
    bloom.resize(size_t{1} << (nbits - 3), 0);
    for (size_t i = 0; i < n; i++) {
        idx_t id = indices[i];
        set.insert(id);
        id &= mask;
        bloom[id >> 3] |= 1 << (id & 7);
    }
}

bool IDSelectorBatch::is_member(idx_t i) const {
    idx_t im = i & mask;
    if (!(bloom[im >> 3] & (1 << (im & 7)))) {
        return false;
    }
    return set.count(i);
}

bool IDSelectorBatch::get_members(idx_t ntotal, std::vector<idx_t>& ids) const {
    sorted_members(set.begin(), set.end(), ntotal, ids);
    return true;
}

/***********************************************************************
 * IDSelectorBitmap
 ***********************************************************************/

IDSelectorBitmap::IDSelectorBitmap(size_t n_in, const uint8_t* bitmap_in)
        : n(n_in), bitmap(bitmap_in) {}

bool IDSelectorBitmap::is_member(idx_t ii) const {
    uint64_t i = ii;
    if ((i >> 3) >= n) {
        return false;
    }
    return (bitmap[i >> 3] >> (i & 7)) & 1;
}

bool IDSelectorBitmap::get_members(idx_t ntotal, std::vector<idx_t>& ids)
        const {
    ids.clear();
    const size_t nbytes =
            std::min(n, size_t(std::max(ntotal, idx_t(0)) + 7) / 8);
    for (size_t b = 0; b < nbytes; b++) {
        if (b % 8 == 0 && b + 8 <= nbytes) {
            uint64_t block;
            std::memcpy(&block, bitmap + b, 8);
            if (block == 0) {
                b += 7;
                continue;
            }
        }
        for (unsigned m = bitmap[b]; m != 0; m &= m - 1) {
            const idx_t id = idx_t(b * 8 + __builtin_ctzll(m));
            if (id >= ntotal) {
                break;
            }
            ids.push_back(id);
        }
    }
    return true;
}

/***********************************************************************
 * Combinations
 ***********************************************************************/

bool IDSelectorNot::get_members(idx_t ntotal, std::vector<idx_t>& ids) const {
    std::vector<idx_t> excluded;
    if (!sel->get_members(ntotal, excluded)) {
        return false;
    }
    ids.clear();
    auto next = excluded.begin();
    for (idx_t id = 0; id < ntotal; id++) {
        if (next != excluded.end() && *next == id) {
            ++next;
        } else {
            ids.push_back(id);
        }
    }
    return true;
}

bool IDSelectorAll::get_members(idx_t ntotal, std::vector<idx_t>& ids) const {
    ids.clear();
    for (idx_t id = 0; id < ntotal; id++) {
        ids.push_back(id);
    }
    return true;
}

bool IDSelectorAnd::get_members(idx_t ntotal, std::vector<idx_t>& ids) const {
    std::vector<idx_t> a, b;
    const bool la = lhs->get_members(ntotal, a);
    const bool lb = rhs->get_members(ntotal, b);
    if (la && lb) {
        ids.clear();
        std::set_intersection(
                a.begin(),
                a.end(),
                b.begin(),
                b.end(),
                std::back_inserter(ids));
    } else if (la) {
        filter_members(a, *rhs, ids);
    } else if (lb) {
        filter_members(b, *lhs, ids);
    } else {
        return false;
    }
    return true;
}

bool IDSelectorOr::get_members(idx_t ntotal, std::vector<idx_t>& ids) const {
    std::vector<idx_t> a, b;
    if (!lhs->get_members(ntotal, a) || !rhs->get_members(ntotal, b)) {
        return false;
    }
    ids.clear();
    std::set_union(
            a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(ids));
    return true;
}

bool IDSelectorXOr::get_members(idx_t ntotal, std::vector<idx_t>& ids) const {
    std::vector<idx_t> a, b;
    if (!lhs->get_members(ntotal, a) || !rhs->get_members(ntotal, b)) {
        return false;
    }
    ids.clear();
    std::set_symmetric_difference(
            a.begin(), a.end(), b.begin(), b.end(), std::back_inserter(ids));
    return true;
}

} // namespace faiss
