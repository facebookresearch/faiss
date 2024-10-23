/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#ifndef INDEX_FLAT_H
#define INDEX_FLAT_H

#include <vector>

#include <faiss/IndexFlatCodes.h>

namespace faiss {

/** Index that stores the full vectors and performs exhaustive search */
struct IndexFlat : IndexFlatCodes {
    explicit IndexFlat(
            idx_t d, ///< dimensionality of the input vectors
            MetricType metric = METRIC_L2);
    explicit IndexFlat(
            idx_t d, ///< dimensionality of the input vectors
            bool is_include_one_attribute,
            MetricType metric = METRIC_L2);
    explicit IndexFlat(
            idx_t d, ///< dimensionality of the input vectors
            bool is_include_two_attribute,
            bool mode_two,
            MetricType metric = METRIC_L2);

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    void search_with_one_attribute(
            idx_t n,
            const float* x,
            const float lower_attribute,
            const float upper_attribute,
            idx_t k,
            float* distances,
            idx_t* labels,
            float* out_attrs,
            const SearchParameters* params = nullptr) const override;
    
    void search_with_two_attribute(
            idx_t n,
            const float* x,
            const float lower_attribute_first,
            const float upper_attribute_first,
            const float lower_attribute_second,
            const float upper_attribute_second,
            idx_t k,
            float* distances,
            idx_t* labels,
            float* out_attrs_first,
            float* out_attrs_second,
            const SearchParameters* params = nullptr) const override;

    void range_search(
            idx_t n,
            const float* x,
            float radius,
            RangeSearchResult* result,
            const SearchParameters* params = nullptr) const override;

    void reconstruct(idx_t key, float* recons) const override;
    void reconstruct_one_attribute(idx_t key, float* recons_attr) const override;
    void reconstruct_two_attribute(idx_t key, float* recons_attr_first, float* recons_attr_second) const override;

    /** compute distance with a subset of vectors
     *
     * @param x       query vectors, size n * d
     * @param labels  indices of the vectors that should be compared
     *                for each query vector, size n * k
     * @param distances
     *                corresponding output distances, size n * k
     */
    void compute_distance_subset(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            const idx_t* labels) const;

    // get pointer to the floating point data
    float* get_xb() {
        return (float*)codes.data();
    }
    const float* get_xb() const {
        return (const float*)codes.data();
    }

    float* get_one_attributes() {
        return (float*)attributes.data();
    }

    const float* get_one_attributes() const {
        return (const float*)attributes.data();
    }

    float* get_two_attributes_first() {
        return (float*)attributes_first.data();
    }

    const float* get_two_attributes_first() const {
        return (const float*)attributes_first.data();
    }

    float* get_two_attributes_second() {
        return (float*)attributes_second.data();
    }

    const float* get_two_attributes_second() const {
        return (const float*)attributes_second.data();
    }

    IndexFlat() {}

    FlatCodesDistanceComputer* get_FlatCodesDistanceComputer() const override;

    /* The stanadlone codec interface (just memcopies in this case) */
    void sa_encode(idx_t n, const float* x, uint8_t* bytes) const override;
    void sa_one_attribute_encode(idx_t n, const float* attr, uint8_t* bytes) const override;
    void sa_two_attribute_encode(idx_t n, const float* attr_first, const float* attr_second,
                                 uint8_t* bytes_first, uint8_t* bytes_second) const override;

    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;
    void sa_one_attribute_decode(idx_t n, const uint8_t* bytes, float* attr) const override;
    void sa_two_attribute_decode(idx_t n, const uint8_t* bytes_first, const uint8_t* bytes_second,
                                 float* attr_first, float* attr_second) const override;
};

struct IndexFlatIP : IndexFlat {
    explicit IndexFlatIP(idx_t d) : IndexFlat(d, METRIC_INNER_PRODUCT) {}
    explicit IndexFlatIP(idx_t d, bool is_include_one_attribute) : IndexFlat(d, is_include_one_attribute, METRIC_INNER_PRODUCT) {}
    explicit IndexFlatIP(idx_t d, bool is_include_two_attribute, bool mode_two) : IndexFlat(d, is_include_two_attribute, mode_two, METRIC_INNER_PRODUCT) {}
    IndexFlatIP() {}
};

struct IndexFlatL2 : IndexFlat {
    // Special cache for L2 norms.
    // If this cache is set, then get_distance_computer() returns
    // a special version that computes the distance using dot products
    // and l2 norms.
    std::vector<float> cached_l2norms;

    /**
     * @param d dimensionality of the input vectors
     */
    explicit IndexFlatL2(idx_t d) : IndexFlat(d, METRIC_L2) {}
    explicit IndexFlatL2(idx_t d, bool is_include_one_attribute) : IndexFlat(d, is_include_one_attribute, METRIC_L2) {}
    explicit IndexFlatL2(idx_t d, bool is_include_two_attribute, bool mode_two) : IndexFlat(d, is_include_two_attribute, mode_two, METRIC_L2) {}
    IndexFlatL2() {}

    // override for l2 norms cache.
    FlatCodesDistanceComputer* get_FlatCodesDistanceComputer() const override;

    // compute L2 norms
    void sync_l2norms();
    // clear L2 norms
    void clear_l2norms();
};

/// optimized version for 1D "vectors".
struct IndexFlat1D : IndexFlatL2 {
    bool continuous_update = true; ///< is the permutation updated continuously?

    std::vector<idx_t> perm; ///< sorted database indices

    explicit IndexFlat1D(bool continuous_update = true);

    /// if not continuous_update, call this between the last add and
    /// the first search
    void update_permutation();

    void add(idx_t n, const float* x) override;
    void add_with_one_attribute(idx_t n, const float* x, const float* attr) override;
    void add_with_two_attribute(idx_t n, const float* x, const float* attr_first, const float* attr_second) override;

    void reset() override;

    /// Warn: the distances returned are L1 not L2
    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;
};

} // namespace faiss

#endif
