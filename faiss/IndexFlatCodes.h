/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/Index.h>
#include <faiss/impl/DistanceComputer.h>
#include <vector>

namespace faiss {

struct CodePacker;

/** Index that encodes all vectors as fixed-size codes (size code_size). Storage
 * is in the codes vector */
struct IndexFlatCodes : Index {
    size_t code_size;
    static const size_t attr_size = sizeof(float);

    /// encoded dataset, size ntotal * code_size
    std::vector<uint8_t> codes;
    std::vector<uint8_t> attributes;
    std::vector<uint8_t> attributes_first;
    std::vector<uint8_t> attributes_second;

    bool is_include_one_attribute = false;
    bool is_include_two_attribute = false;
    bool mode_two = false;

    IndexFlatCodes();

    IndexFlatCodes(size_t code_size, idx_t d, MetricType metric = METRIC_L2);
    IndexFlatCodes(size_t code_size, idx_t d, bool is_include_one_attribute, MetricType metric = METRIC_L2);
    IndexFlatCodes(size_t code_size, idx_t d, bool is_include_two_attribute, bool mode_two, MetricType metric = METRIC_L2);

    /// default add uses sa_encode
    void add(idx_t n, const float* x) override;
    void add_with_one_attribute(idx_t n, const float* x, const float* attr) override;
    void add_with_two_attribute(idx_t n, const float* x, const float* attr_first, const float* attr_second) override;

    void set_is_include_one_attribute();
    bool get_is_include_one_attribute();
    
    void set_is_include_two_attribute();
    bool get_is_include_two_attribute();

    void reset() override;

    void reconstruct_n(idx_t i0, idx_t ni, float* recons) const override;
    void reconstruct_n_one_attribute(idx_t i0, idx_t ni, float* recons_attr) const override;
    void reconstruct_n_two_attribute(idx_t i0, idx_t ni, float* recons_attr_first, float* recons_attr_second) const override;

    void reconstruct(idx_t key, float* recons) const override;
    void reconstruct_one_attribute(idx_t key, float* recons_attr) const override;
    void reconstruct_two_attribute(idx_t key, float* recons_attr_first, float* recons_attr_second) const override;

    size_t sa_code_size() const override;
    size_t sa_one_attribute_code_size() const override;
    size_t sa_two_attribute_code_size() const override;

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
