/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/VisitedTable.h>

#include <cstring>

namespace faiss {

// The vector strategy is faster for get()/set(), but O(size) to initialize.
// advance() is O(1) except every 250 calls, which are O(size).
// The hash set strategy is a constant factor slower for get()/set(),
// but O(1) to construct and O(visits) to advance.
// 10M is only a current estimated threshold, not a proven crossover: we are not
// sure the array still wins at 10M. The point where the array stops paying off
// varies by dataset (it shifts with dimension, working-set / cache pressure,
// etc.), so this is a coarse default that should eventually be replaced by
// smarter per-index tuning.
size_t visited_table_hashset_threshold = 10000000;

std::unique_ptr<VisitedTable> VisitedTable::create(
        size_t size,
        std::optional<bool> use_hashset) {
    bool use_set =
            use_hashset.value_or(size >= visited_table_hashset_threshold);
    if (use_set) {
        return std::make_unique<VisitedTableSet>();
    }
    return std::make_unique<VisitedTableVector>(size);
}

void VisitedTableVector::advance() {
    if (visno < 254) {
        // 254 rather than 255 because sometimes we use visno and visno+1
        ++visno;
    } else {
        memset(visited.data(), 0, sizeof(visited[0]) * visited.size());
        visno = 1;
    }
}

} // namespace faiss
