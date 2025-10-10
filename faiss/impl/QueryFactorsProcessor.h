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
/// as long as they access different indices. Designed for parallel access
/// patterns where each thread processes a distinct range of indices.
struct QueryFactorsProcessor {
   private:
    // Query factors data storage - owned by this processor
    std::vector<QueryFactorsData> query_factors_storage;

    /// Validate access bounds
    /// @param index The index to validate
    /// @throws FaissException if index is out of range or processor
    /// uninitialized
    void validate_access(size_t index) const {
        FAISS_THROW_IF_NOT_MSG(
                !query_factors_storage.empty(),
                "QueryFactorsProcessor: processor not initialized");
        FAISS_THROW_IF_NOT_MSG(
                index < query_factors_storage.size(),
                "QueryFactorsProcessor: index out of range");
    }

   public:
    /// Constructor for query factors storage
    /// @param storage_size Number of storage slots needed (for IVF: n_queries *
    /// n_lists)
    explicit QueryFactorsProcessor(size_t storage_size) {
        FAISS_THROW_IF_NOT_MSG(
                storage_size > 0,
                "QueryFactorsProcessor: storage_size must be positive");

        // Direct allocation - no overflow risk with QueryFactorsData size
        query_factors_storage.resize(storage_size);
    }

    /// Default constructor (no query factors)
    QueryFactorsProcessor() = default;

    /// Get query factors for a specific index
    /// @param index The storage index (user computes: query_id for 1D, query_id
    /// * nlist + list_id for IVF)
    /// @return Const reference to query factors data
    /// @throws FaissException if index is out of range or processor
    /// uninitialized
    const QueryFactorsData& get_query_factors(size_t index) const {
        validate_access(index);
        return query_factors_storage[index];
    }

    /// Set query factors for a specific index
    /// @param index The storage index (user computes: query_id for 1D, query_id
    /// * nlist + list_id for IVF)
    /// @param factors The query factors data to store
    /// @throws FaissException if index is out of range or processor
    /// uninitialized
    /// @note Thread Safety: Safe for concurrent access with different indices
    void set_query_factors(size_t index, const QueryFactorsData& factors) {
        validate_access(index);
        query_factors_storage[index] = factors;
    }

    /// Check if query factors are available
    /// @return true if query factors data is available
    bool has_query_factors() const noexcept {
        return !query_factors_storage.empty();
    }

    /// Get the total storage size
    /// @return Number of storage slots available
    size_t size() const noexcept {
        return query_factors_storage.size();
    }
};

} // namespace faiss
