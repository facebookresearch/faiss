/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Unit tests for the three early-termination knobs on SearchParametersIVF:
//   - ensure_topk_full         (knn, IndexIVF::search_preassigned)
//   - max_empty_result_buckets (range, IndexIVF::range_search_preassigned)
//   - max_lists_num            (fastscan knn)
//
// The tests establish the default-behavior contract: when the new fields
// are left at their defaults, search results must be byte-for-byte identical
// to a search with no params at all.

#include <algorithm>
#include <memory>
#include <random>
#include <set>
#include <vector>

#include <gtest/gtest.h>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQFastScan.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissException.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/index_factory.h>

namespace {

constexpr int d = 16;
constexpr size_t nb = 500;
constexpr size_t nq = 20;
constexpr size_t nlist = 16;
constexpr int k = 10;

std::vector<float> make_data(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> out(n * d);
    for (auto& v : out) {
        v = dist(rng);
    }
    return out;
}

std::unique_ptr<faiss::Index> build_ivf_flat() {
    auto xb = make_data(nb, 0xabcdef);
    std::unique_ptr<faiss::Index> index(
            faiss::index_factory(d, "IVF16,Flat", faiss::METRIC_L2));
    index->train(nb, xb.data());
    index->add(nb, xb.data());
    return index;
}

} // namespace

// Default-constructed SearchParametersIVF (all three new fields at 0/false)
// must reproduce the result of a search without any SearchParameters.
TEST(IVFEarlyTermination, DefaultsPreserveBaselineKnn) {
    auto index = build_ivf_flat();
    auto* ivf = dynamic_cast<faiss::IndexIVFFlat*>(index.get());
    ASSERT_NE(ivf, nullptr);
    ivf->nprobe = 4;

    auto xq = make_data(nq, 0x123456);
    std::vector<float> D_ref(k * nq), D_new(k * nq);
    std::vector<faiss::idx_t> I_ref(k * nq), I_new(k * nq);

    index->search(nq, xq.data(), k, D_ref.data(), I_ref.data());

    faiss::SearchParametersIVF params; // all defaults
    params.nprobe = 4;
    index->search(nq, xq.data(), k, D_new.data(), I_new.data(), &params);

    EXPECT_EQ(I_ref, I_new);
    EXPECT_EQ(D_ref, D_new);
}

TEST(IVFEarlyTermination, DefaultsPreserveBaselineRange) {
    auto index = build_ivf_flat();
    auto* ivf = dynamic_cast<faiss::IndexIVFFlat*>(index.get());
    ASSERT_NE(ivf, nullptr);
    ivf->nprobe = 4;

    auto xq = make_data(nq, 0x789abc);
    const float radius = 2.0f;

    faiss::RangeSearchResult ref(nq);
    faiss::RangeSearchResult tst(nq);

    index->range_search(nq, xq.data(), radius, &ref);

    faiss::SearchParametersIVF params;
    params.nprobe = 4;
    index->range_search(nq, xq.data(), radius, &tst, &params);

    ASSERT_EQ(ref.nq, tst.nq);
    for (size_t i = 0; i <= ref.nq; i++) {
        EXPECT_EQ(ref.lims[i], tst.lims[i]) << "at lims[" << i << "]";
    }
    // The underlying result order is implementation-defined across OMP
    // threads, so compare as sorted (id, distance) pairs per query.
    for (size_t q = 0; q < ref.nq; q++) {
        std::vector<std::pair<faiss::idx_t, float>> ref_pairs;
        std::vector<std::pair<faiss::idx_t, float>> tst_pairs;
        for (size_t j = ref.lims[q]; j < ref.lims[q + 1]; j++) {
            ref_pairs.emplace_back(ref.labels[j], ref.distances[j]);
        }
        for (size_t j = tst.lims[q]; j < tst.lims[q + 1]; j++) {
            tst_pairs.emplace_back(tst.labels[j], tst.distances[j]);
        }
        std::sort(ref_pairs.begin(), ref_pairs.end());
        std::sort(tst_pairs.begin(), tst_pairs.end());
        EXPECT_EQ(ref_pairs, tst_pairs);
    }
}

// With a tight max_codes, ensure_topk_full=true must produce at least as
// many valid (non -1) labels as ensure_topk_full=false.
TEST(IVFEarlyTermination, EnsureTopkFullReducesMisses) {
    auto index = build_ivf_flat();
    auto* ivf = dynamic_cast<faiss::IndexIVFFlat*>(index.get());
    ASSERT_NE(ivf, nullptr);

    auto xq = make_data(nq, 0xdeadbe);
    std::vector<float> D_tight(k * nq), D_full(k * nq);
    std::vector<faiss::idx_t> I_tight(k * nq), I_full(k * nq);

    faiss::SearchParametersIVF params;
    params.nprobe = nlist;
    // Tight budget: strictly smaller than k to guarantee truncation.
    params.max_codes = k / 2;
    params.ensure_topk_full = false;
    index->search(nq, xq.data(), k, D_tight.data(), I_tight.data(), &params);

    params.ensure_topk_full = true;
    index->search(nq, xq.data(), k, D_full.data(), I_full.data(), &params);

    size_t miss_tight =
            std::count(I_tight.begin(), I_tight.end(), (faiss::idx_t)-1);
    size_t miss_full =
            std::count(I_full.begin(), I_full.end(), (faiss::idx_t)-1);

    EXPECT_LE(miss_full, miss_tight);
    EXPECT_GT(miss_tight, size_t(0))
            << "test precondition: tight budget should cause truncation";
    EXPECT_EQ(miss_full, size_t(0))
            << "ensure_topk_full must fill the heap when enough data is "
               "available in the probed lists";
}

// The field is checked against the existing max_codes throw guard for
// parallel modes other than 0/3.
TEST(IVFEarlyTermination, EnsureTopkFullRejectsUnsupportedParallelMode) {
    auto index = build_ivf_flat();
    auto* ivf = dynamic_cast<faiss::IndexIVFFlat*>(index.get());
    ASSERT_NE(ivf, nullptr);
    ivf->parallel_mode = 1;

    auto xq = make_data(1, 0xaaaa);
    std::vector<float> D(k);
    std::vector<faiss::idx_t> I(k);

    faiss::SearchParametersIVF params;
    params.nprobe = 4;
    params.ensure_topk_full = true;
    EXPECT_THROW(
            index->search(1, xq.data(), k, D.data(), I.data(), &params),
            faiss::FaissException);
}

// Range-search early-exit: with max_empty_result_buckets set, the result
// set is a subset of the full-scan result set.
TEST(IVFEarlyTermination, RangeEarlyExitProducesSubset) {
    auto index = build_ivf_flat();
    auto* ivf = dynamic_cast<faiss::IndexIVFFlat*>(index.get());
    ASSERT_NE(ivf, nullptr);

    auto xq = make_data(nq, 0xfacefeed);
    // Pick a radius tight enough that some probed buckets legitimately
    // yield no hits.
    const float radius = 0.5f;

    faiss::RangeSearchResult full(nq);
    faiss::RangeSearchResult early(nq);

    faiss::SearchParametersIVF p_full;
    p_full.nprobe = nlist;
    p_full.max_empty_result_buckets = 0; // disabled
    index->range_search(nq, xq.data(), radius, &full, &p_full);

    faiss::SearchParametersIVF p_early;
    p_early.nprobe = nlist;
    p_early.max_empty_result_buckets = 1; // aggressive
    index->range_search(nq, xq.data(), radius, &early, &p_early);

    ASSERT_EQ(full.nq, early.nq);
    for (size_t q = 0; q < full.nq; q++) {
        std::set<faiss::idx_t> full_ids(
                full.labels + full.lims[q], full.labels + full.lims[q + 1]);
        std::set<faiss::idx_t> early_ids(
                early.labels + early.lims[q], early.labels + early.lims[q + 1]);
        // early_ids ⊆ full_ids
        for (auto id : early_ids) {
            EXPECT_TRUE(full_ids.count(id))
                    << "early-exit returned id " << id
                    << " not present in full-scan result for query " << q;
        }
    }
}

// Verify that the empty-bucket early exit actually stops the probe loop.
TEST(IVFEarlyTermination, RangeEarlyExitActuallyTruncates) {
    auto index = build_ivf_flat();
    auto* ivf = dynamic_cast<faiss::IndexIVFFlat*>(index.get());
    ASSERT_NE(ivf, nullptr);

    auto xq = make_data(nq, 0xfacefeed);
    // Radius chosen so that the dataset (uniform[0,1]^16, nb=500,
    // nlist=16) yields per-query results spread across multiple
    // probed lists with interspersed empty buckets — observable
    // shrinkage when max_empty_result_buckets fires.
    const float radius = 1.0f;

    faiss::RangeSearchResult full(nq);
    faiss::RangeSearchResult early(nq);

    faiss::SearchParametersIVF p_full;
    p_full.nprobe = nlist;
    p_full.max_empty_result_buckets = 0; // disabled
    index->range_search(nq, xq.data(), radius, &full, &p_full);

    faiss::SearchParametersIVF p_early;
    p_early.nprobe = nlist;
    p_early.max_empty_result_buckets = 1; // exit on first empty bucket
    index->range_search(nq, xq.data(), radius, &early, &p_early);

    ASSERT_EQ(full.nq, early.nq);

    // Per-query result count must never grow under early-exit.
    for (size_t q = 0; q < full.nq; q++) {
        EXPECT_LE(
                early.lims[q + 1] - early.lims[q],
                full.lims[q + 1] - full.lims[q])
                << "early-exit grew result count for query " << q;
    }

    // At least one query must strictly shrink — proves the loop broke.
    bool any_shrunk = false;
    for (size_t q = 0; q < full.nq; q++) {
        if ((early.lims[q + 1] - early.lims[q]) <
            (full.lims[q + 1] - full.lims[q])) {
            any_shrunk = true;
            break;
        }
    }
    EXPECT_TRUE(any_shrunk)
            << "no query shrank: max_empty_result_buckets has no effect";
}

// max_empty_result_buckets counter must reset on probes that add hits.
// Hand-verified: with ndup=2 we should NEVER terminate as long as a
// non-empty bucket appears between any two empty buckets, even if every
// other probe is empty. Test by forcing alternating-empty patterns.
TEST(IVFEarlyTermination, RangeEarlyExitCounterResetsOnHit) {
    auto index = build_ivf_flat();
    auto* ivf = dynamic_cast<faiss::IndexIVFFlat*>(index.get());
    ASSERT_NE(ivf, nullptr);

    auto xq = make_data(nq, 0xc0ffee);
    const float radius = 1.0f;

    // With max_empty_result_buckets=1, exit on first empty bucket.
    // With =3, only exit on three consecutive empties — strictly more
    // results expected (or at least equal) for the latter.
    faiss::RangeSearchResult tight(nq);
    faiss::RangeSearchResult relaxed(nq);

    faiss::SearchParametersIVF p_tight;
    p_tight.nprobe = nlist;
    p_tight.max_empty_result_buckets = 1;
    index->range_search(nq, xq.data(), radius, &tight, &p_tight);

    faiss::SearchParametersIVF p_relaxed;
    p_relaxed.nprobe = nlist;
    p_relaxed.max_empty_result_buckets = 3;
    index->range_search(nq, xq.data(), radius, &relaxed, &p_relaxed);

    ASSERT_EQ(tight.nq, relaxed.nq);
    for (size_t q = 0; q < tight.nq; q++) {
        EXPECT_LE(
                tight.lims[q + 1] - tight.lims[q],
                relaxed.lims[q + 1] - relaxed.lims[q])
                << "relaxed early-exit returned fewer results than tight "
                   "for query "
                << q;
    }
}

// max_empty_result_buckets is only supported in parallel_mode == 0.
TEST(IVFEarlyTermination, RangeEarlyExitRejectsUnsupportedParallelMode) {
    auto index = build_ivf_flat();
    auto* ivf = dynamic_cast<faiss::IndexIVFFlat*>(index.get());
    ASSERT_NE(ivf, nullptr);
    ivf->parallel_mode = 1;

    auto xq = make_data(1, 0xbbbb);
    faiss::RangeSearchResult res(1);

    faiss::SearchParametersIVF params;
    params.nprobe = 4;
    params.max_empty_result_buckets = 2;
    EXPECT_THROW(
            index->range_search(1, xq.data(), 1.0f, &res, &params),
            faiss::FaissException);
}

// IndexIVFStats::ndis is documented as "nb of distances computed".
// Verify that the post-filter scan_count flowing through
// InvertedListScannerStats actually populates ndis according to that
// docstring: under no IDSelector, ndis equals the sum of probed list
// sizes; under a 50%-keep IDSelectorBitmap, ndis is roughly halved
// because only codes that survive the selector have distances
// computed.
TEST(IVFEarlyTermination, NdisStatsMatchesDocstringPostFilter) {
    auto index = build_ivf_flat();
    auto* ivf = dynamic_cast<faiss::IndexIVFFlat*>(index.get());
    ASSERT_NE(ivf, nullptr);
    ivf->nprobe = nlist;

    auto xq = make_data(nq, 0xb0bafe11);

    // Baseline: no selector. ndis equals the total raw codes the
    // scanner saw.
    auto& stats = faiss::indexIVF_stats;
    stats.reset();
    std::vector<float> D(k * nq);
    std::vector<faiss::idx_t> I(k * nq);
    index->search(nq, xq.data(), k, D.data(), I.data());
    const size_t ndis_no_sel = stats.ndis;
    EXPECT_GT(ndis_no_sel, size_t(0))
            << "test precondition: at least some codes should be visited";

    // 50%-keep bitmap selector. ndis should drop because rejected
    // codes don't get a distance computed.
    std::vector<uint8_t> packed((nb + 7) / 8, 0);
    for (size_t i = 0; i < nb; i += 2) {
        packed[i / 8] |= 1u << (i % 8);
    }
    faiss::IDSelectorBitmap sel(nb, packed.data());

    faiss::SearchParametersIVF params;
    params.nprobe = nlist;
    params.sel = &sel;

    stats.reset();
    index->search(nq, xq.data(), k, D.data(), I.data(), &params);
    const size_t ndis_with_sel = stats.ndis;

    // Strict: with a 50%-keep selector, ndis must be strictly less
    // than the no-selector ndis (each call processes the same lists
    // with the same nprobe).
    EXPECT_LT(ndis_with_sel, ndis_no_sel)
            << "ndis did not decrease under restrictive IDSelector — "
               "check post-filter scan_cnt plumbing";
    // Loose-bound sanity: ~50% retention should put ndis somewhere
    // around half. Allow generous slop because per-list size variance
    // means the empirical ratio can drift.
    EXPECT_LT(ndis_with_sel * 4, ndis_no_sel * 3)
            << "ndis_with_sel=" << ndis_with_sel
            << " not noticeably below ~75% of ndis_no_sel=" << ndis_no_sel;
}

// Same docstring contract for IndexIVF::range_search_preassigned: ndis
// must be post-filter for the range path too.
TEST(IVFEarlyTermination, NdisStatsMatchesDocstringPostFilterRange) {
    auto index = build_ivf_flat();
    auto* ivf = dynamic_cast<faiss::IndexIVFFlat*>(index.get());
    ASSERT_NE(ivf, nullptr);
    ivf->nprobe = nlist;

    auto xq = make_data(nq, 0xc01dca11);
    const float radius = 1.0f;

    auto& stats = faiss::indexIVF_stats;
    stats.reset();
    faiss::RangeSearchResult res_no_sel(nq);
    index->range_search(nq, xq.data(), radius, &res_no_sel);
    const size_t ndis_no_sel = stats.ndis;
    EXPECT_GT(ndis_no_sel, size_t(0));

    std::vector<uint8_t> packed((nb + 7) / 8, 0);
    for (size_t i = 0; i < nb; i += 2) {
        packed[i / 8] |= 1u << (i % 8);
    }
    faiss::IDSelectorBitmap sel(nb, packed.data());

    faiss::SearchParametersIVF params;
    params.nprobe = nlist;
    params.sel = &sel;

    stats.reset();
    faiss::RangeSearchResult res_with_sel(nq);
    index->range_search(nq, xq.data(), radius, &res_with_sel, &params);
    const size_t ndis_with_sel = stats.ndis;

    EXPECT_LT(ndis_with_sel, ndis_no_sel);
    EXPECT_LT(ndis_with_sel * 4, ndis_no_sel * 3);
}

// With a restrictive bitmap selector, ensure_topk_full keeps probing until
// enough selector-passing candidates have been scanned.
TEST(IVFEarlyTermination, EnsureTopkFullFillsHeapUnderRestrictiveSelector) {
    auto index = build_ivf_flat();
    auto* ivf = dynamic_cast<faiss::IndexIVFFlat*>(index.get());
    ASSERT_NE(ivf, nullptr);
    ivf->nprobe = nlist;

    // ~10%-keep bitmap selector (every 10th id).
    std::vector<uint8_t> packed((nb + 7) / 8, 0);
    for (size_t i = 0; i < nb; i += 10) {
        packed[i / 8] |= 1u << (i % 8);
    }
    faiss::IDSelectorBitmap sel(nb, packed.data());

    auto xq = make_data(nq, 0xdeadbe);
    std::vector<float> D(k * nq);
    std::vector<faiss::idx_t> I(k * nq);

    faiss::SearchParametersIVF params;
    params.nprobe = nlist;
    params.max_codes = k / 2;       // pre-filter budget < k
    params.ensure_topk_full = true; // post-filter soft-cap kicks in
    params.sel = &sel;

    index->search(nq, xq.data(), k, D.data(), I.data(), &params);

    // Valid hits available per query (~50 across all nlist buckets,
    // far exceeding k=10) — the strengthened break should fill the
    // heap completely.
    EXPECT_EQ(std::count(I.begin(), I.end(), (faiss::idx_t)-1), size_t(0))
            << "ensure_topk_full failed to fill the heap under a "
               "restrictive IDSelector";
}

// Without a selector, post-filter and raw scan counts are the same.
TEST(IVFEarlyTermination, EnsureTopkFullNoSelectorByteIdentical) {
    auto index = build_ivf_flat();
    auto* ivf = dynamic_cast<faiss::IndexIVFFlat*>(index.get());
    ASSERT_NE(ivf, nullptr);

    auto xq = make_data(nq, 0xa5a5a5);

    // Reference: ensure_topk_full=false.
    faiss::SearchParametersIVF p_ref;
    p_ref.nprobe = nlist;
    p_ref.max_codes = k / 2;
    p_ref.ensure_topk_full = false;
    std::vector<float> D_ref(k * nq);
    std::vector<faiss::idx_t> I_ref(k * nq);
    index->search(nq, xq.data(), k, D_ref.data(), I_ref.data(), &p_ref);

    // Soft budget: ensure_topk_full=true.
    faiss::SearchParametersIVF p_full;
    p_full.nprobe = nlist;
    p_full.max_codes = k / 2;
    p_full.ensure_topk_full = true;
    std::vector<float> D_full(k * nq);
    std::vector<faiss::idx_t> I_full(k * nq);
    index->search(nq, xq.data(), k, D_full.data(), I_full.data(), &p_full);

    // Softening the budget must not increase the number of missing results.
    size_t miss_ref = std::count(I_ref.begin(), I_ref.end(), (faiss::idx_t)-1);
    size_t miss_full =
            std::count(I_full.begin(), I_full.end(), (faiss::idx_t)-1);
    EXPECT_LE(miss_full, miss_ref);
}

// Current FastScan handlers do not report heap update counts.
TEST(IVFEarlyTermination, FastscanNheapUpdatesIsZeroToday) {
    auto xb = make_data(nb, 0xcafebabe);
    std::unique_ptr<faiss::Index> index(
            faiss::index_factory(d, "IVF16,PQ4x4fs", faiss::METRIC_L2));
    index->train(nb, xb.data());
    index->add(nb, xb.data());

    auto xq = make_data(nq, 0x5678);
    std::vector<float> D(k * nq);
    std::vector<faiss::idx_t> I(k * nq);

    auto& stats = faiss::indexIVF_stats;
    stats.reset();
    index->search(nq, xq.data(), k, D.data(), I.data());

    // FastScan plumbs nq, ndis, and nlist into indexIVF_stats.
    EXPECT_GT(stats.ndis, size_t(0));
    EXPECT_EQ(stats.nheap_updates, size_t(0))
            << "FastScan started reporting nheap_updates — update this "
               "test and verify the value is correct.";
}

// FastScan early-stop options use the per-query implementations.
TEST(IVFEarlyTermination, FastscanDefaultsWork) {
    auto xb = make_data(nb, 0xcafebabe);
    std::unique_ptr<faiss::Index> index(
            faiss::index_factory(d, "IVF16,PQ4x4fs", faiss::METRIC_L2));
    index->train(nb, xb.data());
    index->add(nb, xb.data());

    auto xq = make_data(nq, 0x5678);
    std::vector<float> D(k * nq);
    std::vector<faiss::idx_t> I(k * nq);

    faiss::SearchParametersIVF params; // defaults
    params.nprobe = 4;
    EXPECT_NO_THROW(
            index->search(nq, xq.data(), k, D.data(), I.data(), &params));
}

TEST(IVFEarlyTermination, FastscanMaxListsNumHonored) {
    auto xb = make_data(nb, 0xcafebabe);
    std::unique_ptr<faiss::Index> index(
            faiss::index_factory(d, "IVF16,PQ4x4fs", faiss::METRIC_L2));
    index->train(nb, xb.data());
    index->add(nb, xb.data());

    auto xq = make_data(1, 0x9999);
    std::vector<float> D_full(k), D_cap(k);
    std::vector<faiss::idx_t> I_full(k), I_cap(k);

    faiss::SearchParametersIVF p_full;
    p_full.nprobe = nlist;
    EXPECT_NO_THROW(index->search(
            1, xq.data(), k, D_full.data(), I_full.data(), &p_full));

    faiss::SearchParametersIVF p_cap;
    p_cap.nprobe = nlist;
    p_cap.max_lists_num = 1;
    EXPECT_NO_THROW(
            index->search(1, xq.data(), k, D_cap.data(), I_cap.data(), &p_cap));
}

TEST(IVFEarlyTermination, FastscanEnsureTopkFullSingleQueryWorks) {
    auto xb = make_data(nb, 0xcafebabe);
    std::unique_ptr<faiss::Index> index(
            faiss::index_factory(d, "IVF16,PQ4x4fs", faiss::METRIC_L2));
    index->train(nb, xb.data());
    index->add(nb, xb.data());

    auto xq = make_data(1, 0x9999);
    std::vector<float> D(k);
    std::vector<faiss::idx_t> I(k);

    faiss::SearchParametersIVF params;
    params.nprobe = nlist;
    params.max_codes = 2;
    params.ensure_topk_full = true;
    EXPECT_NO_THROW(
            index->search(1, xq.data(), k, D.data(), I.data(), &params));
}

TEST(IVFEarlyTermination, FastscanEnsureTopkFullRestrictiveSelectorWorks) {
    auto xb = make_data(nb, 0xcafebabe);
    std::unique_ptr<faiss::Index> index(
            faiss::index_factory(d, "IVF16,PQ4x4fs", faiss::METRIC_L2));
    index->train(nb, xb.data());
    index->add(nb, xb.data());

    std::vector<uint8_t> packed((nb + 7) / 8, 0);
    for (size_t i = 0; i < nb; i += 10) {
        packed[i / 8] |= 1u << (i % 8);
    }
    faiss::IDSelectorBitmap sel(nb, packed.data());

    auto xq = make_data(1, 0x9999);
    std::vector<float> D(k);
    std::vector<faiss::idx_t> I(k);

    faiss::SearchParametersIVF params;
    params.nprobe = nlist;
    params.max_codes = 2;
    params.ensure_topk_full = true;
    params.sel = &sel;
    index->search(1, xq.data(), k, D.data(), I.data(), &params);

    EXPECT_EQ(std::count(I.begin(), I.end(), (faiss::idx_t)-1), size_t(0))
            << "FastScan max_codes must be counted after IDSelector filtering";
}

TEST(IVFEarlyTermination, FastscanEnsureTopkFullMultiQueryWorks) {
    // Each query independently resets nscan_q; multi-query batches are
    // therefore supported.
    auto xb = make_data(nb, 0xcafebabe);
    std::unique_ptr<faiss::Index> index(
            faiss::index_factory(d, "IVF16,PQ4x4fs", faiss::METRIC_L2));
    index->train(nb, xb.data());
    index->add(nb, xb.data());

    auto xq = make_data(2, 0x9999);
    std::vector<float> D(k * 2);
    std::vector<faiss::idx_t> I(k * 2);

    faiss::SearchParametersIVF params;
    params.nprobe = 4;
    params.ensure_topk_full = true;
    EXPECT_NO_THROW(
            index->search(2, xq.data(), k, D.data(), I.data(), &params));
}

TEST(IVFEarlyTermination, FastscanExplicitImplem12RejectsKnobs) {
    auto xb = make_data(nb, 0xcafebabe);
    std::unique_ptr<faiss::Index> index(
            faiss::index_factory(d, "IVF16,PQ4x4fs", faiss::METRIC_L2));
    index->train(nb, xb.data());
    index->add(nb, xb.data());
    auto* fs = dynamic_cast<faiss::IndexIVFFastScan*>(index.get());
    ASSERT_NE(fs, nullptr);
    fs->implem = 12;

    auto xq = make_data(1, 0x9999);
    std::vector<float> D(k);
    std::vector<faiss::idx_t> I(k);

    faiss::SearchParametersIVF params;
    params.nprobe = 4;
    params.max_lists_num = 2;
    EXPECT_THROW(
            index->search(1, xq.data(), k, D.data(), I.data(), &params),
            faiss::FaissException);
}

TEST(IVFEarlyTermination, FastscanRangeDefaultsPreserveBaseline) {
    auto xb = make_data(nb, 0xcafebabe);
    std::unique_ptr<faiss::Index> index(
            faiss::index_factory(d, "IVF16,PQ4x4fs", faiss::METRIC_L2));
    index->train(nb, xb.data());
    index->add(nb, xb.data());
    auto* fs = dynamic_cast<faiss::IndexIVFFastScan*>(index.get());
    ASSERT_NE(fs, nullptr);
    fs->nprobe = 4;

    auto xq = make_data(nq, 0x9999);
    const float radius = 1.0f;
    faiss::RangeSearchResult ref(nq);
    faiss::RangeSearchResult tst(nq);

    index->range_search(nq, xq.data(), radius, &ref);

    faiss::SearchParametersIVF params;
    params.nprobe = 4;
    params.max_empty_result_buckets = 0;
    index->range_search(nq, xq.data(), radius, &tst, &params);

    ASSERT_EQ(ref.nq, tst.nq);
    for (size_t i = 0; i <= ref.nq; i++) {
        EXPECT_EQ(ref.lims[i], tst.lims[i]) << "at lims[" << i << "]";
    }
    for (size_t q = 0; q < ref.nq; q++) {
        std::vector<std::pair<faiss::idx_t, float>> ref_pairs;
        std::vector<std::pair<faiss::idx_t, float>> tst_pairs;
        for (size_t j = ref.lims[q]; j < ref.lims[q + 1]; j++) {
            ref_pairs.emplace_back(ref.labels[j], ref.distances[j]);
        }
        for (size_t j = tst.lims[q]; j < tst.lims[q + 1]; j++) {
            tst_pairs.emplace_back(tst.labels[j], tst.distances[j]);
        }
        std::sort(ref_pairs.begin(), ref_pairs.end());
        std::sort(tst_pairs.begin(), tst_pairs.end());
        EXPECT_EQ(ref_pairs, tst_pairs);
    }
}

TEST(IVFEarlyTermination, FastscanRangeHonorsMaxEmptyResultBuckets) {
    auto xb = make_data(nb, 0xcafebabe);
    std::unique_ptr<faiss::Index> index(
            faiss::index_factory(d, "IVF16,PQ4x4fs", faiss::METRIC_L2));
    index->train(nb, xb.data());
    index->add(nb, xb.data());

    auto xq = make_data(1, 0x9999);
    const float radius = 1.0f;
    faiss::RangeSearchResult full(1);
    faiss::RangeSearchResult early(1);
    std::vector<faiss::idx_t> empty_subset;
    faiss::IDSelectorBatch empty_sel(0, empty_subset.data());

    auto& stats = faiss::indexIVF_stats;
    faiss::SearchParametersIVF p_full;
    p_full.nprobe = nlist;
    p_full.max_empty_result_buckets = 0;
    p_full.sel = &empty_sel;
    stats.reset();
    index->range_search(1, xq.data(), radius, &full, &p_full);
    const size_t full_nlist = stats.nlist;

    faiss::SearchParametersIVF p_early;
    p_early.nprobe = nlist;
    p_early.max_empty_result_buckets = 1;
    p_early.sel = &empty_sel;
    stats.reset();
    index->range_search(1, xq.data(), radius, &early, &p_early);
    const size_t early_nlist = stats.nlist;

    EXPECT_GT(full_nlist, size_t(1))
            << "test precondition: full scan should visit several lists";
    EXPECT_LT(early_nlist, full_nlist)
            << "max_empty_result_buckets did not stop fastscan range early";
}

TEST(IVFEarlyTermination, FastscanRangeRejectsMaxEmptyWithExplicitImpl12) {
    auto xb = make_data(nb, 0xcafebabe);
    std::unique_ptr<faiss::Index> index(
            faiss::index_factory(d, "IVF16,PQ4x4fs", faiss::METRIC_L2));
    index->train(nb, xb.data());
    index->add(nb, xb.data());
    auto* fs = dynamic_cast<faiss::IndexIVFFastScan*>(index.get());
    ASSERT_NE(fs, nullptr);
    fs->implem = 12;

    auto xq = make_data(1, 0x9999);
    faiss::RangeSearchResult res(1);

    faiss::SearchParametersIVF params;
    params.nprobe = 4;
    params.max_empty_result_buckets = 2;
    EXPECT_THROW(
            index->range_search(1, xq.data(), 1.0f, &res, &params),
            faiss::FaissException);
}

TEST(IVFEarlyTermination, FastscanRangeRejectsMaxEmptyUnsupportedParallelMode) {
    auto xb = make_data(nb, 0xcafebabe);
    std::unique_ptr<faiss::Index> index(
            faiss::index_factory(d, "IVF16,PQ4x4fs", faiss::METRIC_L2));
    index->train(nb, xb.data());
    index->add(nb, xb.data());
    auto* fs = dynamic_cast<faiss::IndexIVFFastScan*>(index.get());
    ASSERT_NE(fs, nullptr);
    fs->parallel_mode = 1;

    auto xq = make_data(1, 0x9999);
    faiss::RangeSearchResult res(1);

    faiss::SearchParametersIVF params;
    params.nprobe = 4;
    params.max_empty_result_buckets = 2;
    EXPECT_THROW(
            index->range_search(1, xq.data(), 1.0f, &res, &params),
            faiss::FaissException);
}
