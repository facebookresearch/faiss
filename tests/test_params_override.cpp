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

// dimension of the vectors to index
int d = 32;

// size of the database we plan to index
size_t nb = 1000;

// nb of queries
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

/*************************************************************
 * Test functions for a given index type
 *************************************************************/

// IDSelectorWithContext that delegates the verdict to an inner IDSelector while
// exercising the context parameters. Used to prove that a scanner (a) returns
// results identical to the plain is_member path and (b) actually routes through
// is_member_with_context — n_context_calls counts how often the context path
// fired, so a scanner that silently dropped the hook is detectable.
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

// Runs the same membership set (every id where i % 10 == 2) through a plain
// IDSelectorBatch and through a ContextSelector wrapping it, on an
// already-trained index. Returns:
//   0 if the context path fired AND labels are byte-identical to the plain
//   path, 1 if the results diverged (a correctness regression), 2 if the
//   scanner never routed through is_member_with_context (the hook was
//     silently dropped for this index type).
int check_ctx_matches_plain(Index& index) {
    std::vector<float> xq = make_data(nq);

    std::vector<idx_t> kept;
    for (size_t i = 0; i < nb; i++) {
        if (i % 10 == 2) {
            kept.push_back(i);
        }
    }
    IDSelectorBatch batch(kept.size(), kept.data());

    // Plain IDSelector path.
    IVFSearchParameters plain_params;
    plain_params.max_codes = 0;
    plain_params.nprobe = 3;
    plain_params.sel = &batch;
    auto plain_result =
            search_index_with_params(&index, xq.data(), &plain_params);

    // IDSelectorWithContext path over the same membership set.
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

// Same membership set as test_selector, but driven through an
// IDSelectorWithContext to prove the context scan path is functionally
// identical to — and actually exercised by — the given index type.
int test_selector_with_context(const char* index_key) {
    std::vector<float> xb = make_data(nb);
    auto index = make_index(index_key, METRIC_L2, xb);
    // nprobe is set explicitly in the IVFSearchParameters inside
    // check_ctx_matches_plain, so no need to set it on the index here.
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

/*************************************************************
 * Test entry points
 *************************************************************/

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
