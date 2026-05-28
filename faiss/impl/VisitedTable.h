/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef FAISS_VISITED_TABLE_H
#define FAISS_VISITED_TABLE_H

#include <stdint.h>

#include <memory>
#include <optional>
#include <unordered_set>
#include <vector>

#include <faiss/impl/platform_macros.h>
#include <faiss/utils/prefetch.h>

namespace faiss {

FAISS_API extern size_t visited_table_hashset_threshold;

/// Abstract base class for a fast, reusable Visited Set for graph search
/// algorithms.
struct VisitedTable {
    virtual ~VisitedTable() = default;

    /// set flag #no to true, return whether this changed it.
    virtual bool set(size_t no) = 0;

    /// get flag #no
    virtual bool get(size_t no) const = 0;

    /// prefetch flag #no
    virtual void prefetch(size_t no) const = 0;

    /// pre-allocate bucket space to avoid rehashing during repeated set() calls
    virtual void reserve(size_t /*n*/) {}

    /// reset all flags to false
    virtual void advance() = 0;

    /// Factory method to create appropriate implementation.
    /// If use_hashset is nullopt, the use of a hashset will be determined by
    /// size >= visited_table_hashset_threshold.
    static std::unique_ptr<VisitedTable> create(
            size_t size,
            std::optional<bool> use_hashset = std::nullopt);
};

/// Set-based implementation using unordered_set.
/// O(1) to construct and O(visits) to advance.
struct VisitedTableSet FAISS_FINAL : VisitedTable {
    std::unordered_set<size_t> visited_set;

    VisitedTableSet() = default;

    bool set(size_t no) final {
        return visited_set.insert(no).second;
    }

    bool get(size_t no) const final {
        return visited_set.count(no) != 0;
    }

    void prefetch(size_t /*no*/) const final {
        // No-op for set-based implementation
    }

    void reserve(size_t n) final {
        visited_set.reserve(n);
    }

    void advance() final {
        visited_set.clear();
    }
};

/// Vector-based implementation using a versioned byte array.
/// Faster for get()/set(), but O(size) to initialize.
/// advance() is O(1) except every 250 calls, which are O(size).
struct VisitedTableVector FAISS_FINAL : VisitedTable {
    std::vector<uint8_t> visited;
    uint8_t visno{1}; // Version number, 1..254

    explicit VisitedTableVector(size_t size) : visited(size, 0) {}

    bool set(size_t no) final {
        if (visited[no] == visno) {
            return false;
        }
        visited[no] = visno;
        return true;
    }

    bool get(size_t no) const final {
        return visited[no] == visno;
    }

    void prefetch(size_t no) const final {
        prefetch_L2(&visited[no]);
    }

    void advance() final;
};

} // namespace faiss

#endif
