/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/Index.h>
#include <faiss/IndexBinary.h>
#include <faiss/impl/IDSelector.h>

#include <unordered_map>
#include <vector>

namespace faiss {

/** Index that translates search results to ids */
template <typename IndexT>
struct IndexIDMapTemplate : IndexT {
    using component_t = typename IndexT::component_t;
    using distance_t = typename IndexT::distance_t;

    IndexT* index = nullptr; ///! the sub-index
    bool own_fields = false; ///! whether pointers are deleted in destructo
    std::vector<idx_t> id_map;

    explicit IndexIDMapTemplate(IndexT* index);

    /// @param xids if non-null, ids to store for the vectors (size n)
    void add_with_ids(idx_t n, const component_t* x, const idx_t* xids)
            override;

    /// this will fail. Use add_with_ids
    void add(idx_t n, const component_t* x) override;

    void search(
            idx_t n,
            const component_t* x,
            idx_t k,
            distance_t* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void train(idx_t n, const component_t* x) override;

    void reset() override;

    /// remove ids adapted to IndexFlat
    size_t remove_ids(const IDSelector& sel) override;

    void range_search(
            idx_t n,
            const component_t* x,
            distance_t radius,
            RangeSearchResult* result,
            const SearchParameters* params = nullptr) const override;

    void merge_from(IndexT& otherIndex, idx_t add_id = 0) override;
    void check_compatible_for_merge(const IndexT& otherIndex) const override;

    ~IndexIDMapTemplate() override;
    IndexIDMapTemplate() {
        own_fields = false;
        index = nullptr;
    }
};

using IndexIDMap = IndexIDMapTemplate<Index>;
using IndexBinaryIDMap = IndexIDMapTemplate<IndexBinary>;

/** same as IndexIDMap but also provides an efficient reconstruction
 *  implementation via a 2-way index */
template <typename IndexT>
struct IndexIDMap2Template : IndexIDMapTemplate<IndexT> {
    using component_t = typename IndexT::component_t;
    using distance_t = typename IndexT::distance_t;

    std::unordered_map<idx_t, idx_t> rev_map;

    explicit IndexIDMap2Template(IndexT* index);

    /// make the rev_map from scratch
    void construct_rev_map();

    void add_with_ids(idx_t n, const component_t* x, const idx_t* xids)
            override;

    size_t remove_ids(const IDSelector& sel) override;

    void reconstruct(idx_t key, component_t* recons) const override;

    /// check that the rev_map and the id_map are in sync
    void check_consistency() const;

    void merge_from(IndexT& otherIndex, idx_t add_id = 0) override;

    ~IndexIDMap2Template() override {}
    IndexIDMap2Template() {}
};

using IndexIDMap2 = IndexIDMap2Template<Index>;
using IndexBinaryIDMap2 = IndexIDMap2Template<IndexBinary>;

// IDSelector that translates the ids using an IDMap
struct IDSelectorTranslated : IDSelector {
    const std::vector<int64_t>& id_map;
    const IDSelector* sel;

    IDSelectorTranslated(
            const std::vector<int64_t>& id_map,
            const IDSelector* sel)
            : id_map(id_map), sel(sel) {}

    IDSelectorTranslated(IndexBinaryIDMap& index_idmap, const IDSelector* sel)
            : id_map(index_idmap.id_map), sel(sel) {}

    IDSelectorTranslated(IndexIDMap& index_idmap, const IDSelector* sel)
            : id_map(index_idmap.id_map), sel(sel) {}

    bool is_member(idx_t id) const override {
        return sel->is_member(id_map[id]);
    }
    bool is_member(idx_t id, std::optional<float> d) const override {
        return sel->is_member(id_map[id], d);
    }
};

} // namespace faiss
