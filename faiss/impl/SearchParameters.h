/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <memory>
#include <vector>

#include <faiss/Index.h>
#include <faiss/impl/IDSelector.h>

namespace faiss {

/** Owns per-call parameter copies and selectors used by composite indexes.
 * The caller's parameter tree and selectors are never modified.
 */
struct SearchParametersStorage {
    std::vector<std::unique_ptr<SearchParameters>> parameters;
    std::vector<std::unique_ptr<IDSelector>> selectors;

    /// Intersect sel with params->sel, preserving all index-specific fields.
    /// With null params, retain the index's configured search defaults.
    const SearchParameters* with_selector(
            const Index* index,
            const SearchParameters* params,
            IDSelector* sel);

    /// Translate selectors in Refine/PreTransform parameter trees to row IDs.
    /// Quantizer parameters are intentionally not visited: their IDs belong
    /// to a different index (the coarse quantizer).
    const SearchParameters* translate_selectors(
            const SearchParameters* params,
            const std::vector<idx_t>& id_map);

   private:
    SearchParameters* copy(const SearchParameters* params);
};

} // namespace faiss
