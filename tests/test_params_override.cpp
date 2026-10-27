/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdio>
#include <cstdlib>

#include <memory>
#include <random>
#include <vector>

#include <gtest/gtest.h>

#include <faiss/IVFlib.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFRaBitQ.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/index_factory.h>

using namespace faiss;

namespace {

int d = 32;
size_t nb = 1000;
size_t nq = 200;

std::mt19937 rng;

std::vector<float> make_data(size_t n) {
    std::vector<float> database(n * d);
    std::uniform_real_distribution<> distrib;
    for (size_t i = 0; i < n * d; i++) {
        database[i] = distrib(rng);
    }
    return database;
}

std::unique_ptr<Index> make_index(
        const char* index_type,
        MetricType metric,
        const std::vector<float>& x) {
    assert(x.size() % d == 0);
    idx_t local_nb = x.size() / d;
    std::unique_ptr<Index> index(index_factory(d, index_type, metric));
    index->train(local_nb, x.data());
    index->add(local_nb, x.data());
    return index;
}

std::vector<idx_t> search_index_with_params(
        Index* index,
        const float* xq,
        IVFSearchParameters* params) {
    int k = 10;
    std::vector<idx_t> I(k * nq);
    std::vector<float> D(k * nq);
    ivflib::search_with_parameters(
            index, nq, xq, k, D.data(), I.data(), params);
    return I;
}

// Delegates to an inner IDSelector; n_context_calls detects a scanner that
// silently drops the is_member_with_context hook.
struct ContextSelector : IDSelectorWithContext {
    const IDSelector& inner;
    mutable size_t n_context_calls = 0;
    explicit ContextSelector(const IDSelector& inner) : inner(inner) {}
    bool is_member(idx_t id) const override {
        return inner.is_member(id);
    }
    bool is_member_with_context(idx_t id, const IDScanContext& ctx)
            const override {
        (void)ctx;
        ++n_context_calls;
        return inner.is_member(id);
    }
};

// Returns 0 if ok, 1 if results differ, 2 if the context hook never fired.
int check_ctx_matches_plain(Index& index) {
    std::vector<float> xq = make_data(nq);

    std::vector<idx_t> kept;
    for (size_t i = 0; i < nb; i++) {
        if (i % 10 == 2) {
            kept.push_back(i);
        }
    }
    IDSelectorBatch batch(kept.size(), kept.data());

    IVFSearchParameters plain_params;
    plain_params.max_codes = 0;
    plain_params.nprobe = 3;
    plain_params.sel = &batch;
    auto plain_result =
            search_index_with_params(&index, xq.data(), &plain_params);

    ContextSelector ctx_sel(batch);
    IVFSearchParameters ctx_params;
    ctx_params.max_codes = 0;
    ctx_params.nprobe = 3;
    ctx_params.sel = &ctx_sel;
    auto ctx_result = search_index_with_params(&index, xq.data(), &ctx_params);

    if (plain_result != ctx_result) {
        return 1;
    }
    if (ctx_sel.n_context_calls == 0) {
        return 2;
    }
    return 0;
}

int test_selector_with_context(const char* index_key) {
    std::vector<float> xb = make_data(nb);
    auto index = make_index(index_key, METRIC_L2, xb);
    return check_ctx_matches_plain(*index);
}

// Multibit RaBitQ (nb_bits >= 2) scans through the two hand-written loops in
// RaBitQuantizer.cpp rather than run_scan_codes1, so it needs the context hook
// wired separately. nb_bits=3 => ex_bits=2 => scan_codes_multibit.
int test_selector_with_context_rabitq_multibit() {
    std::vector<float> xb = make_data(nb);
    IndexFlatL2 quantizer(d);
    IndexIVFRaBitQ index(
            &quantizer, d, 32, METRIC_L2, /*own_invlists=*/true, /*nb_bits=*/3);
    index.train(nb, xb.data());
    index.add(nb, xb.data());
    return check_ctx_matches_plain(index);
}

} // namespace

// An IDSelectorWithContext must be functionally identical to a plain
// IDSelector AND actually exercised (results must not silently fall back to
// is_member) on every applicable IVF scanner. IVFFlat/IVFSQ route through
// run_scan_codes1; IVFPQ through WrappedSearchResult::skip_entry; multibit
// RaBitQ through its two hand-written scan loops.
TEST(TSELCtx, IVFFlat) {
    EXPECT_EQ(test_selector_with_context("PCA16,IVF32,Flat"), 0);
}

TEST(TSELCtx, IVFFPQ) {
    EXPECT_EQ(test_selector_with_context("PCA16,IVF32,PQ4x8np"), 0);
}

TEST(TSELCtx, IVFFSQ) {
    EXPECT_EQ(test_selector_with_context("PCA16,IVF32,SQ8"), 0);
}

TEST(TSELCtx, RaBitQMultibit) {
    EXPECT_EQ(test_selector_with_context_rabitq_multibit(), 0);
}

TEST(TSELCtx, Panorama) {
    EXPECT_EQ(test_selector_with_context("IVF32,FlatPanorama"), 0);
}
