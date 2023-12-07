/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <faiss/impl/ResultCollector.h>
namespace faiss {

/** ResultCollectorFactory to create a ResultCollector object */
struct ResultCollectorFactory {
    DefaultCollector default_collector;
    const std::vector<int64_t>* id_map;

    // Create a new ResultCollector object
    virtual ResultCollector* newCollector() {
        return &default_collector;
    }

    // For default case, the factory share single object and no need to delete
    // the object. For other case, the factory can create a new object which
    // need to be deleted later. We have deleteCollector method to handle both
    // case as factory class knows how to release resource that it created
    virtual void deleteCollector(ResultCollector* collector) {
        // Do nothing
    }

    virtual ~ResultCollectorFactory() {}
};

} // namespace faiss
