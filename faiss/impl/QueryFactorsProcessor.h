/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>

#include "faiss/impl/FaissAssert.h"
#include "faiss/impl/RaBitQUtils.h"

namespace faiss {

// Import QueryFactorsData for specialized methods
using rabitq_utils::QueryFactorsData;

/// Processor for handling RaBitQ query factors during fast scan
/// Specialized for QueryFactorsData - the only type ever used in practice
/// Thread safety: Multiple threads may safely call methods simultaneously
/// as long as they access different query_id values. Designed for parallel
/// access patterns where each thread processes a distinct range of queries.
struct QueryFactorsProcessor {
   private:
    // Query factors data storage - owned by this processor
    std::vector<QueryFactorsData> query_factors_storage;
    size_t num_queries;

    /// Validate query access bounds
    /// @param query_id The query ID to validate
    /// @throws FaissException if query_id is out of range or processor
    /// uninitialized
    void validate_access(size_t query_id) const {
        FAISS_THROW_IF_NOT_MSG(
                !query_factors_storage.empty(),
                "QueryFactorsProcessor: processor not initialized");
        FAISS_THROW_IF_NOT_MSG(
                query_id < num_queries,
                "QueryFactorsProcessor: query_id out of range");
    }

   public:
    /// Constructor optimized for QueryFactorsData (the only type ever used)
    /// @param n_queries Number of queries in the batch
    explicit QueryFactorsProcessor(size_t n_queries) : num_queries(n_queries) {
        FAISS_THROW_IF_NOT_MSG(
                n_queries > 0,
                "QueryFactorsProcessor: n_queries must be positive");

        // Direct allocation - no overflow risk with QueryFactorsData size
        query_factors_storage.resize(n_queries);
    }

    /// Default constructor (no query factors)
    QueryFactorsProcessor() : num_queries(0) {}

    /// Get query factors for a specific query
    /// @param query_id The ID of the query (0-based index)
    /// @return Const reference to query factors data
    /// @throws FaissException if query_id is out of range or processor
    /// uninitialized
    const QueryFactorsData& get_query_factors(size_t query_id) const {
        validate_access(query_id);
        return query_factors_storage[query_id];
    }

    /// Set query factors for a specific query
    /// @param query_id The ID of the query (0-based index)
    /// @param factors The query factors data to store
    /// @throws FaissException if query_id is out of range or processor
    /// uninitialized
    /// @note Thread Safety: Safe for concurrent access with different query_id
    /// values
    void set_query_factors(size_t query_id, const QueryFactorsData& factors) {
        validate_access(query_id);
        query_factors_storage[query_id] = factors;
    }

    /// Check if query factors are available
    /// @return true if query factors data is available
    bool has_query_factors() const noexcept {
        return !query_factors_storage.empty();
    }

    /// Get the number of queries for which factors are available
    /// @return Number of queries with available factors
    size_t get_num_queries() const noexcept {
        return num_queries;
    }
};

} // namespace faiss
