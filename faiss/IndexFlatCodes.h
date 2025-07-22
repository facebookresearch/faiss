/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <vector>

#include <faiss/Index.h>
#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/maybe_owned_vector.h>

namespace faiss {

struct CodePacker;

/** Index that encodes all vectors as fixed-size codes (size code_size). Storage
 * is in the codes vector */
struct IndexFlatCodes : Index {
    size_t code_size;

    /// encoded dataset, size ntotal * code_size
    MaybeOwnedVector<uint8_t> codes;

    IndexFlatCodes();

    IndexFlatCodes(size_t code_size, idx_t d, MetricType metric = METRIC_L2);

    /// default add uses sa_encode
    void add(idx_t n, const float* x) override;
    void add(idx_t n, const void* x, NumericType numeric_type) override;

    void reset() override;

    void reconstruct_n(idx_t i0, idx_t ni, float* recons) const override;

    void reconstruct(idx_t key, float* recons) const override;

    size_t sa_code_size() const override;

    /** remove some ids. NB that because of the structure of the
     * index, the semantics of this operation are
     * different from the usual ones: the new ids are shifted */
    size_t remove_ids(const IDSelector& sel) override;

    /** a FlatCodesDistanceComputer offers a distance_to_code method
     *
     * The default implementation explicitly decodes the vector with sa_decode.
     */
    virtual FlatCodesDistanceComputer* get_FlatCodesDistanceComputer() const;

    DistanceComputer* get_distance_computer() const override {
        return get_FlatCodesDistanceComputer();
    }

    /** Search implemented by decoding */
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;
    void search(
            idx_t n,
            const void* x,
            NumericType numeric_type,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void range_search(
            idx_t n,
            const float* x,
            float radius,
            RangeSearchResult* result,
            const SearchParameters* params = nullptr) const override;

    // returns a new instance of a CodePacker
    CodePacker* get_CodePacker() const;

    void check_compatible_for_merge(const Index& otherIndex) const override;

    virtual void merge_from(Index& otherIndex, idx_t add_id = 0) override;

    virtual void add_sa_codes(idx_t n, const uint8_t* x, const idx_t* xids)
            override;

    // permute_entries. perm of size ntotal maps new to old positions
    void permute_entries(const idx_t* perm);
};

} // namespace faiss
