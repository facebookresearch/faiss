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
// A size of ~1M seems to be the threshold where the hash set wins.
size_t visited_table_hashset_threshold = 500000;

VisitedTable::VisitedTable(size_t size, std::optional<bool> use_hashset)
        : visno(use_hashset.value_or(size >= visited_table_hashset_threshold)
                        ? 0
                        : 1) {
    if (visno != 0) {
        visited.resize(size, 0);
    }
}

void VisitedTable::advance() {
    if (visno == 0) {
        visited_set.clear();
    } else if (visno < 254) {
        // 254 rather than 255 because sometimes we use visno and visno+1
        ++visno;
    } else {
        memset(visited.data(), 0, sizeof(visited[0]) * visited.size());
        visno = 1;
    }
}

} // namespace faiss
