/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <unordered_map>
#include <vector>

#include <faiss/MetricType.h>

/** IDDeduper is intended to define a group of vectors to dedupe */

namespace faiss {

/** Encapsulates a set of id groups to handle. */
struct IDDeduper {
    virtual idx_t group_id(idx_t id) const = 0;
    virtual ~IDDeduper() {}
};

/** id to group id mapping */
struct IDDeduperMap : IDDeduper {
    std::unordered_map<idx_t, idx_t>* m;

    IDDeduperMap(std::unordered_map<idx_t, idx_t>* m);

    idx_t group_id(idx_t id) const final;

    ~IDDeduperMap() override {}
};

} // namespace faiss
