/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIDMap.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexPreTransform.h>
#include <faiss/IndexRefine.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissException.h>
#include <faiss/impl/IDSelector.h>

namespace {

using namespace faiss;

// A downstream parameter subclass must survive both selector translation and
// wrapper forwarding. A real Flat search below checks the resulting filter.
struct CustomParameters : IVFPQSearchParameters {
    int tag = 1234;
    SearchParameters* clone() const override {
        return new CustomParameters(*this);
    }
};

struct InspectingFlat : IndexFlatL2 {
    const IndexRefineSearchParameters* original_outer = nullptr;
    const SearchParametersPreTransform* original_middle = nullptr;
    const CustomParameters* original_leaf = nullptr;
    IDSelector* original_sel = nullptr;
    bool fail = false;
    mutable int calls = 0;

    InspectingFlat() : IndexFlatL2(1) {}

    void inspect(const SearchParameters* params) const {
        ++calls;
        EXPECT_EQ(original_outer->sel, original_sel);
        EXPECT_EQ(original_middle->sel, nullptr);
        EXPECT_EQ(original_leaf->sel, nullptr);
        EXPECT_EQ(original_middle->index_params, original_leaf);
        const auto* copied = dynamic_cast<const CustomParameters*>(params);
        ASSERT_NE(copied, nullptr);
        EXPECT_NE(copied, original_leaf);
        EXPECT_EQ(copied->tag, 1234);
        EXPECT_EQ(copied->nprobe, 7);
        EXPECT_EQ(copied->max_codes, 19);
        EXPECT_EQ(copied->scan_table_threshold, 31);
        EXPECT_EQ(copied->polysemous_ht, 11);
        EXPECT_NE(copied->sel, nullptr);
        if (fail) {
            throw FaissException("intentional sub-search failure");
        }
    }

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params) const override {
        inspect(params);
        IndexFlatL2::search(n, x, k, distances, labels, params);
    }

    void range_search(
            idx_t n,
            const float* x,
            float radius,
            RangeSearchResult* result,
            const SearchParameters* params) const override {
        inspect(params);
        IndexFlatL2::range_search(n, x, radius, result, params);
    }
};

TEST(WrapperSearchParameters, PreserveConcreteTypeAndCallerDuringSearch) {
    InspectingFlat base;
    IndexPreTransform transform(&base);
    IndexRefineFlat refine(&transform);
    IndexIDMap index(&refine);
    const float xb[] = {0, 1, 2, 3};
    const idx_t ids[] = {30, 10, 40, 20};
    index.add_with_ids(4, xb, ids);
    IDSelectorRange sel(10, 21);
    const CustomParameters leaf = [] {
        CustomParameters p;
        p.nprobe = 7;
        p.max_codes = 19;
        p.scan_table_threshold = 31;
        p.polysemous_ht = 11;
        return p;
    }();
    SearchParametersPreTransform middle;
    middle.index_params = const_cast<CustomParameters*>(&leaf);
    IndexRefineSearchParameters outer;
    outer.sel = &sel;
    outer.k_factor = 2;
    outer.base_index_params = &middle;
    base.original_outer = &outer;
    base.original_middle = &middle;
    base.original_leaf = &leaf;
    base.original_sel = &sel;
    const float query = 0;
    for (bool fail : {false, true, false}) {
        base.fail = fail;
        float distances[2];
        idx_t labels[2];
        RangeSearchResult result(1);
        if (fail) {
            EXPECT_THROW(
                    index.search(1, &query, 2, distances, labels, &outer),
                    FaissException);
            EXPECT_THROW(
                    index.range_search(1, &query, 10, &result, &outer),
                    FaissException);
        } else {
            index.search(1, &query, 2, distances, labels, &outer);
            EXPECT_EQ(labels[0], 10);
            EXPECT_EQ(labels[1], 20);
            EXPECT_FLOAT_EQ(distances[0], 1);
            EXPECT_FLOAT_EQ(distances[1], 9);
            index.range_search(1, &query, 10, &result, &outer);
            ASSERT_EQ(result.lims[1], 2);
            EXPECT_EQ(result.labels[0], 10);
            EXPECT_EQ(result.labels[1], 20);
        }
        EXPECT_EQ(outer.sel, &sel);
        EXPECT_EQ(outer.base_index_params, &middle);
        EXPECT_EQ(middle.index_params, &leaf);
        EXPECT_EQ(leaf.sel, nullptr);
    }
    EXPECT_EQ(base.calls, 6);
}

TEST(WrapperSearchParameters, RejectSlicingUnclonableSubclass) {
    struct UnclonableParameters : SearchParameters {
        int extra = 99;
    };
    IndexFlatL2 base(1);
    IndexIDMap index(&base);
    const float xb[] = {0, 1};
    const idx_t ids[] = {20, 10};
    index.add_with_ids(2, xb, ids);
    IDSelectorRange sel(10, 11);
    UnclonableParameters params;
    params.sel = &sel;
    float distance;
    idx_t label;
    EXPECT_THROW(
            index.search(1, xb, 1, &distance, &label, &params), FaissException);
    EXPECT_EQ(params.sel, &sel);
    // No copies are needed on the no-selector path, including custom types.
    params.sel = nullptr;
    index.search(1, xb, 1, &distance, &label, &params);
    EXPECT_EQ(label, 20);
}

TEST(WrapperSearchParameters, ExplicitTranslatedSelectorIsNotTranslatedTwice) {
    IndexFlatL2 base(1);
    IndexRefineFlat refine(&base);
    IndexIDMap index(&refine);
    const float xb[] = {0, 1, 2, 3};
    const idx_t ids[] = {30, 10, 40, 20};
    index.add_with_ids(4, xb, ids);
    IDSelectorRange sel(10, 21);
    IDSelectorTranslated translated(index, &sel);
    SearchParameters leaf;
    leaf.sel = &translated;
    IndexRefineSearchParameters params;
    params.base_index_params = &leaf;
    float distances[2];
    idx_t labels[2];
    index.search(1, xb, 2, distances, labels, &params);
    EXPECT_EQ(labels[0], 10);
    EXPECT_EQ(labels[1], 20);
    EXPECT_EQ(leaf.sel, &translated);
}

TEST(WrapperSearchParameters, DisjointSelectorsAndMissingResults) {
    for (auto metric : {METRIC_L2, METRIC_INNER_PRODUCT}) {
        IndexFlat base(1, metric);
        IndexRefineFlat index(&base);
        const float xb[] = {0, 1, 2, 3};
        index.add(4, xb);
        IDSelectorRange outer(0, 1), inner(2, 3);
        SearchParameters leaf;
        leaf.sel = &inner;
        IndexRefineSearchParameters params;
        params.sel = &outer;
        params.base_index_params = &leaf;
        float distances[2];
        idx_t labels[2];
        index.search(1, xb, 2, distances, labels, &params);
        EXPECT_EQ(labels[0], -1);
        EXPECT_EQ(labels[1], -1);
        RangeSearchResult result(1);
        index.range_search(1, xb, 10, &result, &params);
        EXPECT_EQ(result.lims[1], 0);
        // Fewer eligible vectors than k must retain the missing-ID sentinel.
        params.sel = &inner;
        index.search(1, xb, 2, distances, labels, &params);
        EXPECT_EQ(labels[0], 2);
        EXPECT_EQ(labels[1], -1);
    }
}

TEST(WrapperSearchParameters, QuantizerSelectorKeepsItsOwnIdSpace) {
    IndexFlatL2 quantizer(1);
    const float centroids[] = {-2, 2};
    quantizer.add(2, centroids);
    IndexIVFFlat base(&quantizer, 1, 2);
    IndexRefineFlat refine(&base);
    IndexIDMap index(&refine);
    const float xb[] = {-2, -1, 1, 2};
    const idx_t ids[] = {300, 100, 400, 200};
    index.train(4, xb);
    index.add_with_ids(4, xb, ids);
    IDSelectorRange data_sel(100, 201), coarse_sel(0, 1);
    SearchParameters coarse_params;
    coarse_params.sel = &coarse_sel;
    SearchParametersIVF leaf;
    leaf.nprobe = 1;
    leaf.quantizer_params = &coarse_params;
    IndexRefineSearchParameters params;
    params.sel = &data_sel;
    params.base_index_params = &leaf;
    const float query = 0;
    float distances[2];
    idx_t labels[2];
    index.search(1, &query, 2, distances, labels, &params);
    EXPECT_EQ(labels[0], 100);
    EXPECT_EQ(labels[1], -1);
    RangeSearchResult result(1);
    index.range_search(1, &query, 10, &result, &params);
    ASSERT_EQ(result.lims[1], 1);
    EXPECT_EQ(result.labels[0], 100);
    EXPECT_EQ(coarse_params.sel, &coarse_sel);
    EXPECT_EQ(leaf.quantizer_params, &coarse_params);
}

} // namespace
