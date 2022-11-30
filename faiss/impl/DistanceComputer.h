/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/Index.h>

namespace faiss {

/***********************************************************
 * The distance computer maintains a current query and computes
 * distances to elements in an index that supports random access.
 *
 * The DistanceComputer is not intended to be thread-safe (eg. because
 * it maintains counters) so the distance functions are not const,
 * instantiate one from each thread if needed.
 *
 * Note that the equivalent for IVF indexes is the InvertedListScanner,
 * that has additional methods to handle the inverted list context.
 ***********************************************************/
struct DistanceComputer {
    /// called before computing distances. Pointer x should remain valid
    /// while operator () is called
    virtual void set_query(const float* x) = 0;

    /// compute distance of vector i to current query
    virtual float operator()(idx_t i) = 0;

    /// compute distance between two stored vectors
    virtual float symmetric_dis(idx_t i, idx_t j) = 0;

    virtual ~DistanceComputer() {}
};

/*************************************************************
 * Specialized version of the DistanceComputer when we know that codes are
 * laid out in a flat index.
 */
struct FlatCodesDistanceComputer : DistanceComputer {
    const uint8_t* codes;
    size_t code_size;

    FlatCodesDistanceComputer(const uint8_t* codes, size_t code_size)
            : codes(codes), code_size(code_size) {}

    FlatCodesDistanceComputer() : codes(nullptr), code_size(0) {}

    float operator()(idx_t i) final {
        return distance_to_code(codes + i * code_size);
    }

    /// compute distance of current query to an encoded vector
    virtual float distance_to_code(const uint8_t* code) = 0;

    virtual ~FlatCodesDistanceComputer() {}
};

} // namespace faiss
