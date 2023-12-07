/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once
#include <faiss/impl/ResultCollector.h>
namespace faiss {

/** ResultCollector is intended to define how to collect search result */
struct ResultCollectorFactory {
    DefaultCollector defaultCollector;

    // For each result, collect method is called to store result
    virtual ResultCollector* newCollector() {
        return &defaultCollector;
    }

    virtual void deleteCollector(ResultCollector* collector) {
        // Do nothing
    }
    // This method is called after all result is collected
    virtual ~ResultCollectorFactory() {}
};

} // namespace faiss
