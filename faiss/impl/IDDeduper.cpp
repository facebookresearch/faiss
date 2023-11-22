/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/IDDeduper.h>

namespace faiss {

/***********************************************************************
 * IDDeduperMap
 ***********************************************************************/
IDDeduperMap::IDDeduperMap(std::unordered_map<idx_t, idx_t>* m) : m(m) {}

idx_t IDDeduperMap::group_id(idx_t id) const {
    return m->at(id);
}

} // namespace faiss
