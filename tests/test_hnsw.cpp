/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <bit>
#include <cstddef>
#include <functional>
#include <limits>
#include <optional>
#include <random>
#include <unordered_set>
#include <vector>

#include <faiss/IndexBinaryHNSW.h>
#include <faiss/IndexBinaryIVF.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/HNSW.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/impl/VisitedTable.h>
#include <faiss/impl/hnsw/MinimaxHeap.h>
#include <faiss/utils/fp16.h>
#include <faiss/utils/random.h>
#include <omp.h>

int reference_pop_min(faiss::MinimaxHeap& heap, float* vmin_out) {
    assert(heap.k > 0);
    int i = heap.k - 1;
    while (i >= 0) {
        if (heap.ids[i] != -1) {
            break;
        }
        i--;
    }
    if (i == -1) {
        return -1;
    }
    int imin = i;
    float vmin = heap.dis[i];
    i--;
    while (i >= 0) {
        if (heap.ids[i] != -1 && heap.dis[i] < vmin) {
            vmin = heap.dis[i];
            imin = i;
        }
        i--;
    }
    if (vmin_out) {
        *vmin_out = vmin;
    }
    int ret = heap.ids[imin];
    heap.ids[imin] = -1;
    --heap.nvalid;

    return ret;
}

void test_popmin(
        int heap_size,
        int amount_to_put,
        std::optional<float> fixed_distance = std::nullopt) {
    faiss::MinimaxHeap mm_heap(heap_size);

    using storage_idx_t = faiss::HNSW::storage_idx_t;

    std::default_random_engine rng(123 + heap_size * amount_to_put);
    std::uniform_int_distribution<storage_idx_t> u(0, 65536);
    std::uniform_real_distribution<float> uf(0, 1);

    std::unordered_set<storage_idx_t> indices;
    while (static_cast<int>(indices.size()) < amount_to_put) {
        const storage_idx_t index = u(rng);
        indices.insert(index);
    }

    for (const auto index : indices) {
        float distance;
        if (fixed_distance) {
            distance = *fixed_distance;
        } else {
            distance = uf(rng);
            if (distance >= 0.7f) {
                distance = std::numeric_limits<float>::infinity();
            }
        }
        mm_heap.push(index, distance);
    }

    faiss::MinimaxHeap cloned_mm_heap = mm_heap;

    while (mm_heap.size() > 0) {
        ASSERT_EQ(mm_heap.k, cloned_mm_heap.k);
        ASSERT_EQ(mm_heap.nvalid, cloned_mm_heap.nvalid);
        ASSERT_EQ(mm_heap.ids, cloned_mm_heap.ids);
        ASSERT_EQ(mm_heap.dis, cloned_mm_heap.dis);

        float cloned_vmin_dis = std::numeric_limits<float>::quiet_NaN();
        storage_idx_t cloned_vmin_idx =
                reference_pop_min(cloned_mm_heap, &cloned_vmin_dis);

        float vmin_dis = std::numeric_limits<float>::quiet_NaN();
        storage_idx_t vmin_idx = mm_heap.pop_min(&vmin_dis);

        ASSERT_EQ(vmin_dis, cloned_vmin_dis);
        ASSERT_EQ(vmin_idx, cloned_vmin_idx);
    }

    ASSERT_EQ(mm_heap.k, cloned_mm_heap.k);
    ASSERT_EQ(mm_heap.nvalid, cloned_mm_heap.nvalid);
    ASSERT_EQ(mm_heap.ids, cloned_mm_heap.ids);
    ASSERT_EQ(mm_heap.dis, cloned_mm_heap.dis);
}

TEST(HNSW, Test_popmin) {
    std::vector<size_t> sizes = {1, 2, 3, 4, 5, 7, 9, 11, 16, 27, 32, 64, 128};
    for (const size_t size : sizes) {
        for (size_t amount = size; amount > 0; amount /= 2) {
            test_popmin(size, amount);
        }
    }
}

TEST(HNSW, Test_popmin_identical_distances) {
    std::vector<size_t> sizes = {1, 2, 3, 4, 5, 7, 9, 11, 16, 27, 32};
    for (const size_t size : sizes) {
        for (size_t amount = size; amount > 0; amount /= 2) {
            test_popmin(size, amount, 1.0f);
        }
    }
}

TEST(HNSW, Test_popmin_infinite_distances) {
    std::vector<size_t> sizes = {1, 2, 3, 4, 5, 7, 9, 11, 16, 27, 32};
    for (const size_t size : sizes) {
        for (size_t amount = size; amount > 0; amount /= 2) {
            test_popmin(size, amount, std::numeric_limits<float>::infinity());
        }
    }
}

class HNSWTest : public testing::Test {
   protected:
    HNSWTest() {
        xb = std::make_unique<std::vector<float>>(d * nb);
        faiss::float_rand(xb->data(), d * nb, 12345);
        index = std::make_unique<faiss::IndexHNSWFlat>(d, M);
        index->add(nb, xb->data());
        xq = std::unique_ptr<std::vector<float>>(
                new std::vector<float>(d * nq));
        faiss::float_rand(xq->data(), d * nq, 12345);
        dis = std::unique_ptr<faiss::DistanceComputer>(
                index->storage->get_distance_computer());
        dis->set_query(xq->data() + 0 * index->d);
    }

    const int d = 64;
    const int nb = 2000;
    const int M = 4;
    const int nq = 10;
    const int k = 10;
    std::unique_ptr<std::vector<float>> xb;
    std::unique_ptr<std::vector<float>> xq;
    std::unique_ptr<faiss::DistanceComputer> dis;
    std::unique_ptr<faiss::IndexHNSWFlat> index;
};

/** Do a BFS on the candidates list */
int reference_search_from_candidates(
        const faiss::HNSW& hnsw,
        faiss::DistanceComputer& qdis,
        faiss::ResultHandler& res,
        faiss::MinimaxHeap& candidates,
        faiss::VisitedTable& vt,
        faiss::HNSWStats& stats,
        int level,
        int nres_in,
        const faiss::SearchParametersHNSW* params) {
    int nres = nres_in;
    int ndis = 0;

    bool do_dis_check = params ? params->check_relative_distance
                               : hnsw.check_relative_distance;
    int efSearch = params ? params->efSearch : hnsw.efSearch;
    const faiss::IDSelector* sel = params ? params->sel : nullptr;

    faiss::HNSW::C::T threshold = res.threshold;
    for (int i = 0; i < candidates.size(); i++) {
        faiss::idx_t v1 = candidates.ids[i];
        float d = candidates.dis[i];
        FAISS_ASSERT(v1 >= 0);
        if (!sel || sel->is_member(v1)) {
            if (d < threshold) {
                if (res.add_result(d, v1)) {
                    threshold = res.threshold;
                }
            }
        }
        vt.set(v1);
    }

    int nstep = 0;

    while (candidates.size() > 0) {
        float d0 = 0;
        int v0 = candidates.pop_min(&d0);

        if (do_dis_check) {
            // tricky stopping condition: there are more that ef
            // distances that are processed already that are smaller
            // than d0

            int n_dis_below = candidates.count_below(d0);
            if (n_dis_below >= efSearch) {
                break;
            }
        }

        size_t begin, end;
        hnsw.neighbor_range(v0, level, &begin, &end);

        for (size_t j = begin; j < end; j++) {
            int v1 = hnsw.neighbors[j];
            if (v1 < 0) {
                break;
            }
            if (vt.get(v1)) {
                continue;
            }
            vt.set(v1);
            ndis++;
            float d = qdis(v1);
            if (!sel || sel->is_member(v1)) {
                if (d < threshold) {
                    if (res.add_result(d, v1)) {
                        threshold = res.threshold;
                        nres += 1;
                    }
                }
            }

            candidates.push(v1, d);
        }

        nstep++;
        if (!do_dis_check && nstep > efSearch) {
            break;
        }
    }

    if (level == 0) {
        stats.n1++;
        if (candidates.size() == 0) {
            stats.n2++;
        }
        stats.ndis += ndis;
        stats.nhops += nstep;
    }

    return nres;
}

std::priority_queue<faiss::HNSW::Node> reference_search_from_candidate_unbounded(
        const faiss::HNSW& hnsw,
        const faiss::HNSW::Node& node,
        faiss::DistanceComputer& qdis,
        int ef,
        faiss::VisitedTable* vt,
        faiss::HNSWStats& stats) {
    int ndis = 0;
    std::priority_queue<faiss::HNSW::Node> top_candidates;
    std::priority_queue<
            faiss::HNSW::Node,
            std::vector<faiss::HNSW::Node>,
            std::greater<faiss::HNSW::Node>>
            candidates;

    top_candidates.push(node);
    candidates.push(node);

    vt->set(node.second);

    while (!candidates.empty()) {
        float d0;
        faiss::HNSW::storage_idx_t v0;
        std::tie(d0, v0) = candidates.top();

        if (d0 > top_candidates.top().first) {
            break;
        }

        candidates.pop();

        size_t begin, end;
        hnsw.neighbor_range(v0, 0, &begin, &end);

        for (size_t j = begin; j < end; ++j) {
            int v1 = hnsw.neighbors[j];

            if (v1 < 0) {
                break;
            }
            if (vt->get(v1)) {
                continue;
            }

            vt->set(v1);

            float d1 = qdis(v1);
            ++ndis;

            if (top_candidates.top().first > d1 ||
                static_cast<int>(top_candidates.size()) < ef) {
                candidates.emplace(d1, v1);
                top_candidates.emplace(d1, v1);

                if (static_cast<int>(top_candidates.size()) > ef) {
                    top_candidates.pop();
                }
            }
        }

        stats.nhops += 1;
    }

    ++stats.n1;
    if (candidates.size() == 0) {
        ++stats.n2;
    }
    stats.ndis += ndis;

    return top_candidates;
}

TEST_F(HNSWTest, TEST_search_from_candidate_unbounded) {
    auto nearest = index->hnsw.entry_point;
    float d_nearest = (*dis)(nearest);
    auto node = faiss::HNSW::Node(d_nearest, nearest);
    std::unique_ptr<faiss::VisitedTable> vt =
            faiss::VisitedTable::create(index->ntotal);
    faiss::HNSWStats stats;

    auto top_candidates = faiss::hnsw_detail::search_from_candidate_unbounded(
            index->hnsw, node, *dis, k, vt.get(), stats);

    auto reference_nearest = index->hnsw.entry_point;
    float reference_d_nearest = (*dis)(nearest);
    auto reference_node =
            faiss::HNSW::Node(reference_d_nearest, reference_nearest);
    std::unique_ptr<faiss::VisitedTable> reference_vt =
            faiss::VisitedTable::create(index->ntotal);
    faiss::HNSWStats reference_stats;

    auto reference_top_candidates = reference_search_from_candidate_unbounded(
            index->hnsw,
            reference_node,
            *dis,
            k,
            reference_vt.get(),
            reference_stats);
    EXPECT_EQ(stats.ndis, reference_stats.ndis);
    EXPECT_EQ(stats.nhops, reference_stats.nhops);
    EXPECT_EQ(stats.n1, reference_stats.n1);
    EXPECT_EQ(stats.n2, reference_stats.n2);
    EXPECT_EQ(top_candidates.size(), reference_top_candidates.size());
}

TEST_F(HNSWTest, TEST_search_from_candidates) {
    std::vector<faiss::idx_t> I(k * nq);
    std::vector<float> D(k * nq);
    std::vector<faiss::idx_t> reference_I(k * nq);
    std::vector<float> reference_D(k * nq);
    using RH = faiss::HeapBlockResultHandler<faiss::HNSW::C>;

    std::unique_ptr<faiss::VisitedTable> vt =
            faiss::VisitedTable::create(index->ntotal);
    std::unique_ptr<faiss::VisitedTable> reference_vt =
            faiss::VisitedTable::create(index->ntotal);
    int num_candidates = 10;
    faiss::MinimaxHeap candidates(num_candidates);
    faiss::MinimaxHeap reference_candidates(num_candidates);

    for (int i = 0; i < num_candidates; i++) {
        vt->set(i);
        reference_vt->set(i);
        candidates.push(i, (*dis)(i));
        reference_candidates.push(i, (*dis)(i));
    }

    faiss::HNSWStats stats;
    RH bres(nq, D.data(), I.data(), k);
    faiss::HeapBlockResultHandler<faiss::HNSW::C>::SingleResultHandler res(
            bres);

    res.begin(0);
    faiss::hnsw_detail::search_from_candidates(
            index->hnsw, *dis, res, candidates, *vt, stats, 0, 0, nullptr);
    res.end();

    faiss::HNSWStats reference_stats;
    RH reference_bres(nq, reference_D.data(), reference_I.data(), k);
    faiss::HeapBlockResultHandler<faiss::HNSW::C>::SingleResultHandler
            reference_res(reference_bres);
    reference_res.begin(0);
    reference_search_from_candidates(
            index->hnsw,
            *dis,
            reference_res,
            reference_candidates,
            *reference_vt,
            reference_stats,
            0,
            0,
            nullptr);
    reference_res.end();
    for (int j = 0; j < k; j++) {
        EXPECT_NEAR(I[j], reference_I[j], 0.1);
        EXPECT_NEAR(D[j], reference_D[j], 0.1);
    }
    EXPECT_EQ(reference_stats.ndis, stats.ndis);
    EXPECT_EQ(reference_stats.nhops, stats.nhops);
    EXPECT_EQ(reference_stats.n1, stats.n1);
    EXPECT_EQ(reference_stats.n2, stats.n2);
}

TEST_F(HNSWTest, TEST_search_neighbors_to_add) {
    std::unique_ptr<faiss::VisitedTable> vt =
            faiss::VisitedTable::create(index->ntotal);
    std::unique_ptr<faiss::VisitedTable> reference_vt =
            faiss::VisitedTable::create(index->ntotal);

    std::priority_queue<faiss::HNSW::NodeDistCloser> link_targets;
    std::priority_queue<faiss::HNSW::NodeDistCloser> reference_link_targets;

    faiss::hnsw_detail::search_neighbors_to_add(
            index->hnsw,
            *dis,
            link_targets,
            index->hnsw.entry_point,
            (*dis)(index->hnsw.entry_point),
            index->hnsw.max_level,
            *vt,
            false);

    faiss::hnsw_detail::search_neighbors_to_add(
            index->hnsw,
            *dis,
            reference_link_targets,
            index->hnsw.entry_point,
            (*dis)(index->hnsw.entry_point),
            index->hnsw.max_level,
            *reference_vt,
            true);

    EXPECT_EQ(link_targets.size(), reference_link_targets.size());
    while (!link_targets.empty()) {
        auto val = link_targets.top();
        auto reference_val = reference_link_targets.top();
        EXPECT_EQ(val.d, reference_val.d);
        EXPECT_EQ(val.id, reference_val.id);
        link_targets.pop();
        reference_link_targets.pop();
    }
}

TEST_F(HNSWTest, TEST_search_level_0) {
    std::vector<faiss::idx_t> I(k * nq);
    std::vector<float> D(k * nq);

    using RH = faiss::HeapBlockResultHandler<faiss::HNSW::C>;
    RH bres1(nq, D.data(), I.data(), k);
    faiss::HeapBlockResultHandler<faiss::HNSW::C>::SingleResultHandler res1(
            bres1);
    RH bres2(nq, D.data(), I.data(), k);
    faiss::HeapBlockResultHandler<faiss::HNSW::C>::SingleResultHandler res2(
            bres2);

    faiss::HNSWStats stats1, stats2;
    std::unique_ptr<faiss::VisitedTable> vt1 =
            faiss::VisitedTable::create(index->ntotal);
    std::unique_ptr<faiss::VisitedTable> vt2 =
            faiss::VisitedTable::create(index->ntotal);
    auto nprobe = 5;
    const faiss::HNSW::storage_idx_t values[] = {1, 2, 3, 4, 5};
    const faiss::HNSW::storage_idx_t* nearest_i = values;
    const float distances[] = {0.1, 0.2, 0.3, 0.4, 0.5};
    const float* nearest_d = distances;

    // search_type == 1
    res1.begin(0);
    index->hnsw.search_level_0(
            *dis, res1, nprobe, nearest_i, nearest_d, 1, stats1, *vt1, nullptr);
    res1.end();

    // search_type == 2
    res2.begin(0);
    index->hnsw.search_level_0(
            *dis, res2, nprobe, nearest_i, nearest_d, 2, stats2, *vt2, nullptr);
    res2.end();

    // search_type 1 calls search_from_candidates in a loop nprobe times.
    // search_type 2 pushes the candidates and just calls search_from_candidates
    // once, so those stats will be much less.
    EXPECT_GT(stats1.ndis, stats2.ndis);
    EXPECT_GT(stats1.nhops, stats2.nhops);
    EXPECT_GT(stats1.n1, stats2.n1);
    EXPECT_GT(stats1.n2, stats2.n2);
}

// Restores the global OpenMP thread count on scope exit.
struct ScopedOmpThreads {
    int saved = omp_get_max_threads();
    ~ScopedOmpThreads() {
        omp_set_num_threads(saved);
    }
};

TEST_F(HNSWTest, TEST_search_advances_one_generation_per_query) {
    ScopedOmpThreads omp_guard;
    omp_set_num_threads(1);
    index->hnsw.use_visited_hashset = false;

    auto& reusable = dynamic_cast<faiss::VisitedTableVector&>(
            faiss::VisitedTable::get_reusable(index->ntotal, false));
    std::fill(reusable.visited.begin(), reusable.visited.end(), 0);
    reusable.visno = 10;

    std::vector<faiss::idx_t> labels(k);
    std::vector<float> distances(k);
    index->search(1, xq->data(), k, distances.data(), labels.data());

    // Acquiring the reusable table advances 10 -> 11 to clear stale state.
    // HNSW::search then advances 11 -> 12 after the query. The outer wrapper
    // must not perform a third advance.
    EXPECT_EQ(12, reusable.visno);
}

TEST_F(HNSWTest, TEST_search_reuse_correctness) {
    ScopedOmpThreads omp_guard;

    std::vector<faiss::idx_t> I1(k * nq);
    std::vector<float> D1(k * nq);

    omp_set_num_threads(1);
    index->search(nq, xq->data(), k, D1.data(), I1.data());

    // The reused versioned-array path must match a hash-set reference.
    std::vector<faiss::idx_t> Iref(k * nq);
    std::vector<float> Dref(k * nq);
    index->hnsw.use_visited_hashset = true;
    index->search(nq, xq->data(), k, Dref.data(), Iref.data());
    index->hnsw.use_visited_hashset = false;
    std::vector<faiss::idx_t> Iarr(k * nq);
    std::vector<float> Darr(k * nq);
    index->search(nq, xq->data(), k, Darr.data(), Iarr.data());
    EXPECT_EQ(Iref, Iarr);
    EXPECT_EQ(Dref, Darr);

    // Multi-threaded search must match single-threaded: each thread reuses its
    // own thread-local table.
    index->hnsw.use_visited_hashset = std::nullopt;
    omp_set_num_threads(4);
    std::vector<faiss::idx_t> Imt(k * nq);
    std::vector<float> Dmt(k * nq);
    index->search(nq, xq->data(), k, Dmt.data(), Imt.data());
    EXPECT_EQ(I1, Imt);
    EXPECT_EQ(D1, Dmt);
}

// These tests override virtual functions (the add() of an IVF quantizer, the
// sample searches), which the swig module cannot do in Python. The other tests
// of add_level_0_links_from_sample() are in test_hnsw.py.
namespace {

constexpr int kSinkDim = 8;
constexpr size_t kSinkNodes = 4096;

std::vector<float> sink_test_vectors(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::normal_distribution<float> normal;
    std::vector<float> x(n * kSinkDim);
    for (auto& v : x) {
        v = normal(rng);
    }
    return x;
}

// Vectors near the nodes, cycling through all of them.
std::vector<float> near_nodes(
        const std::vector<float>& nodes,
        size_t n,
        uint32_t seed) {
    const std::vector<float> noise = sink_test_vectors(n, seed);
    std::vector<float> x(n * kSinkDim);
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < kSinkDim; j++) {
            x[i * kSinkDim + j] = nodes[(i % kSinkNodes) * kSinkDim + j] +
                    0.01f * noise[i * kSinkDim + j];
        }
    }
    return x;
}

// Removes all links of the entry point, so that every search stops there.
void plant_sink_at_entry_point(faiss::HNSW& hnsw) {
    for (int level = 0; level <= hnsw.max_level; level++) {
        size_t begin, end;
        hnsw.neighbor_range(hnsw.entry_point, level, &begin, &end);
        for (size_t k = begin; k < end; k++) {
            hnsw.neighbors[k] = -1;
        }
    }
}

std::vector<uint16_t> to_fp16(const std::vector<float>& x) {
    std::vector<uint16_t> x16(x.size());
    for (size_t i = 0; i < x.size(); i++) {
        x16[i] = faiss::encode_fp16(x[i]);
    }
    return x16;
}

// Fraction of the queries for which the index returns the exact nearest node.
double recall_at_1(
        const faiss::IndexHNSW& index,
        const std::vector<float>& xq) {
    const size_t nq = xq.size() / kSinkDim;
    std::vector<float> distances(nq);
    std::vector<faiss::idx_t> labels(nq), exact(nq);
    index.search(nq, xq.data(), 1, distances.data(), labels.data());
    index.storage->search(nq, xq.data(), 1, distances.data(), exact.data());
    size_t correct = 0;
    for (size_t i = 0; i < nq; i++) {
        correct += labels[i] == exact[i];
    }
    return double(correct) / nq;
}

// An HNSW quantizer whose add() leaves the entry point without links, so that
// every search stops there.
struct SinkHNSWFlat : faiss::IndexHNSWFlat {
    SinkHNSWFlat() : faiss::IndexHNSWFlat(kSinkDim, 32) {}

    void add(faiss::idx_t n, const float* x) override {
        faiss::IndexHNSWFlat::add(n, x);
        plant_sink_at_entry_point(hnsw);
    }
};

// Trains an IVF index with a SinkHNSWFlat quantizer and returns the recall@1
// of the quantizer on vectors that training did not see.
double ivf_quantizer_recall(
        int samples_per_list,
        faiss::NumericType numeric_type = faiss::NumericType::Float32) {
    const std::vector<float> nodes = sink_test_vectors(kSinkNodes, 1);
    SinkHNSWFlat quantizer;
    faiss::IndexIVFFlat ivf(&quantizer, kSinkDim, kSinkNodes);
    ivf.quantizer_trains_alone = 2;
    quantizer.level_0_link_samples_per_node = samples_per_list;
    const std::vector<float> train = near_nodes(nodes, 40000, 2);
    if (numeric_type == faiss::NumericType::Float16) {
        ivf.train_ex(40000, to_fp16(train).data(), numeric_type);
    } else {
        ivf.train(40000, train.data());
    }
    return recall_at_1(quantizer, near_nodes(nodes, 8192, 3));
}

constexpr int kBinaryDim = 128;
constexpr size_t kBinaryCodeSize = kBinaryDim / 8;

std::vector<uint8_t> random_codes(size_t n, uint32_t seed) {
    std::mt19937 rng(seed);
    std::vector<uint8_t> x(n * kBinaryCodeSize);
    for (auto& b : x) {
        b = rng() & 0xff;
    }
    return x;
}

// Codes near the nodes, cycling through all of them: 4 random bits differ.
std::vector<uint8_t> near_codes(
        const std::vector<uint8_t>& nodes,
        size_t n,
        uint32_t seed) {
    const size_t nn = nodes.size() / kBinaryCodeSize;
    std::mt19937 rng(seed);
    std::vector<uint8_t> x(n * kBinaryCodeSize);
    for (size_t i = 0; i < n; i++) {
        uint8_t* code = x.data() + i * kBinaryCodeSize;
        std::copy_n(
                nodes.data() + (i % nn) * kBinaryCodeSize,
                kBinaryCodeSize,
                code);
        for (int f = 0; f < 4; f++) {
            const int bit = rng() % kBinaryDim;
            code[bit / 8] ^= 1 << (bit % 8);
        }
    }
    return x;
}

// Fraction of the queries for which the index returns a node at the exact
// nearest distance. Hamming distances have ties, so the IDs can differ.
double binary_recall_at_1(
        const faiss::IndexBinaryHNSW& index,
        const std::vector<uint8_t>& xq) {
    const size_t nq = xq.size() / kBinaryCodeSize;
    std::vector<int32_t> distances(nq), exact(nq);
    std::vector<faiss::idx_t> labels(nq);
    index.search(nq, xq.data(), 1, distances.data(), labels.data());
    index.storage->search(nq, xq.data(), 1, exact.data(), labels.data());
    size_t correct = 0;
    for (size_t i = 0; i < nq; i++) {
        correct += distances[i] == exact[i];
    }
    return double(correct) / nq;
}

// A binary HNSW quantizer whose add() leaves the entry point without links.
struct SinkBinaryHNSW : faiss::IndexBinaryHNSW {
    SinkBinaryHNSW() : faiss::IndexBinaryHNSW(kBinaryDim, 32) {}

    void add(faiss::idx_t n, const uint8_t* x) override {
        faiss::IndexBinaryHNSW::add(n, x);
        plant_sink_at_entry_point(hnsw);
    }
};

// Trains an IndexBinaryIVF with a SinkBinaryHNSW quantizer and returns the
// recall@1 of the quantizer on vectors that training did not see.
double binary_ivf_quantizer_recall(int samples_per_node) {
    constexpr size_t nlist = 1024;
    const std::vector<uint8_t> nodes = random_codes(nlist, 1);
    SinkBinaryHNSW quantizer;
    quantizer.level_0_link_samples_per_node = samples_per_node;
    faiss::IndexBinaryIVF ivf(&quantizer, kBinaryDim, nlist);
    const std::vector<uint8_t> train = near_codes(nodes, 40000, 2);
    ivf.train(40000, train.data());
    return binary_recall_at_1(quantizer, near_codes(nodes, 8192, 3));
}

// Leaves the entry point of a graph with only level 0 with links to two nodes,
// and the first of them without links. Thus a search with efSearch 1 stops at
// that node when it is nearer than the second one, and the other searches
// continue through the second node.
void plant_partial_sink(faiss::HNSW& hnsw) {
    size_t begin, end;
    hnsw.neighbor_range(hnsw.entry_point, 0, &begin, &end);
    const faiss::HNSW::storage_idx_t sink = hnsw.neighbors[begin];
    for (size_t k = begin + 2; k < end; k++) {
        hnsw.neighbors[k] = -1;
    }
    hnsw.neighbor_range(sink, 0, &begin, &end);
    for (size_t k = begin; k < end; k++) {
        hnsw.neighbors[k] = -1;
    }
}

// An IndexHNSWFlat with a partial sink.
void build_partial_sink(
        faiss::IndexHNSWFlat& index,
        const std::vector<float>& nodes) {
    index.hnsw.levels.assign(kSinkNodes, 1);
    index.add(kSinkNodes, nodes.data());
    plant_partial_sink(index.hnsw);
}

// Level-0 link parameters that search with efSearch 1.
struct GreedyLinkParameters {
    faiss::SearchParametersHNSW search_params;
    faiss::HNSWLevel0LinkParameters params;

    GreedyLinkParameters() {
        search_params.efSearch = 1;
        params.search_params = &search_params;
    }
    GreedyLinkParameters(const GreedyLinkParameters&) = delete;
    GreedyLinkParameters& operator=(const GreedyLinkParameters&) = delete;
};

// The sample searches of an IndexHNSWFlat, without the classes of
// IndexHNSW.cpp. It records the number of vectors of the last HNSW search, the
// exact searches, and the exact searches that repeat the label of the previous
// exact search of the same vector.
struct RecordingLinkSample : faiss::HNSWLinkSampleSearch {
    const faiss::IndexHNSWFlat& index;
    const std::vector<float>& x;
    mutable size_t last_m = 0;
    mutable size_t n_exact = 0;
    mutable size_t n_repeated = 0;
    mutable std::vector<faiss::idx_t> last_found;

    RecordingLinkSample(
            const faiss::IndexHNSWFlat& index,
            const std::vector<float>& x)
            : index(index), x(x), last_found(x.size() / kSinkDim, -1) {
        n = x.size() / kSinkDim;
    }

    std::vector<float> rows(size_t m, const faiss::idx_t* ids) const {
        std::vector<float> out(m * kSinkDim);
        for (size_t j = 0; j < m; j++) {
            const faiss::idx_t i = ids ? ids[j] : faiss::idx_t(j);
            std::copy_n(
                    x.data() + i * kSinkDim,
                    kSinkDim,
                    out.data() + j * kSinkDim);
        }
        return out;
    }

    void search(
            const faiss::SearchParametersHNSW* params,
            size_t m,
            const faiss::idx_t* ids,
            faiss::idx_t* labels) const override {
        last_m = m;
        std::vector<float> distances(m);
        index.search(
                m, rows(m, ids).data(), 1, distances.data(), labels, params);
    }

    void search_exact(
            size_t m,
            const faiss::idx_t* ids,
            const faiss::idx_t* found,
            faiss::idx_t* exact,
            uint8_t* nearer) const override {
        std::vector<float> distances(m);
        index.storage->search(
                m, rows(m, ids).data(), 1, distances.data(), exact);
        for (size_t j = 0; j < m; j++) {
            n_repeated += last_found[ids[j]] == found[j];
            last_found[ids[j]] = found[j];
            // The samples are near distinct nodes, so no distances tie.
            nearer[j] = exact[j] != found[j];
        }
        n_exact += m;
    }
};

int hamming(const uint8_t* a, const uint8_t* b) {
    int d = 0;
    for (size_t k = 0; k < kBinaryCodeSize; k++) {
        d += std::popcount(static_cast<unsigned>(a[k] ^ b[k]));
    }
    return d;
}

// The sample searches of an IndexBinaryHNSW, without the classes of
// IndexBinaryHNSW.cpp.
struct BinaryReferenceLinkSample : faiss::HNSWLinkSampleSearch {
    const faiss::IndexBinaryHNSW& index;
    const std::vector<uint8_t>& nodes;
    const std::vector<uint8_t>& x;

    BinaryReferenceLinkSample(
            const faiss::IndexBinaryHNSW& index,
            const std::vector<uint8_t>& nodes,
            const std::vector<uint8_t>& x)
            : index(index), nodes(nodes), x(x) {
        n = x.size() / kBinaryCodeSize;
    }

    std::vector<uint8_t> rows(size_t m, const faiss::idx_t* ids) const {
        std::vector<uint8_t> out(m * kBinaryCodeSize);
        for (size_t j = 0; j < m; j++) {
            const faiss::idx_t i = ids ? ids[j] : faiss::idx_t(j);
            std::copy_n(
                    x.data() + i * kBinaryCodeSize,
                    kBinaryCodeSize,
                    out.data() + j * kBinaryCodeSize);
        }
        return out;
    }

    void search(
            const faiss::SearchParametersHNSW* params,
            size_t m,
            const faiss::idx_t* ids,
            faiss::idx_t* labels) const override {
        std::vector<int32_t> distances(m);
        index.search(
                m, rows(m, ids).data(), 1, distances.data(), labels, params);
    }

    void search_exact(
            size_t m,
            const faiss::idx_t* ids,
            const faiss::idx_t* found,
            faiss::idx_t* exact,
            uint8_t* nearer) const override {
        const std::vector<uint8_t> xq = rows(m, ids);
        std::vector<int32_t> distances(m);
        index.storage->search(m, xq.data(), 1, distances.data(), exact);
        for (size_t j = 0; j < m; j++) {
            const uint8_t* q = xq.data() + j * kBinaryCodeSize;
            nearer[j] = exact[j] >= 0 &&
                    hamming(q, nodes.data() + exact[j] * kBinaryCodeSize) <
                            hamming(q,
                                    nodes.data() + found[j] * kBinaryCodeSize);
        }
    }
};

using LinkStep = std::function<faiss::HNSWLevel0LinkStats(
        faiss::IndexHNSWFlat&,
        const faiss::HNSWLevel0LinkParameters&)>;

// The graph after `add_links` on an IndexHNSWFlat with a partial sink. The
// step must make at least 2 rounds, so that the links of the last round come
// from the vectors that were searched again.
std::vector<faiss::HNSW::storage_idx_t> linked_partial_sink(
        const std::vector<float>& nodes,
        const LinkStep& add_links) {
    faiss::IndexHNSWFlat index(kSinkDim, 32);
    build_partial_sink(index, nodes);
    const GreedyLinkParameters greedy;
    const faiss::HNSWLevel0LinkStats stats = add_links(index, greedy.params);
    EXPECT_GE(stats.n_rounds, 2);
    return {index.hnsw.neighbors.begin(), index.hnsw.neighbors.end()};
}

} // namespace

TEST(HNSWLevel0LinksFromSample, SearchesAgainOnlyTheVectorsAtSinks) {
    const std::vector<float> nodes = sink_test_vectors(kSinkNodes, 1);
    faiss::IndexHNSWFlat index(kSinkDim, 32);
    build_partial_sink(index, nodes);
    const std::vector<float> x = near_nodes(nodes, 40000, 2);
    RecordingLinkSample sample(index, x);
    const GreedyLinkParameters greedy;
    std::unique_ptr<faiss::DistanceComputer> dis(
            index.storage->get_distance_computer());

    const faiss::HNSWLevel0LinkStats stats =
            faiss::hnsw_add_level_0_links_from_sample(
                    index.hnsw,
                    index.ntotal,
                    *dis,
                    false,
                    sample,
                    greedy.params);

    EXPECT_EQ(stats.n_sinks_remaining, 0);
    EXPECT_LT(sample.last_m, x.size() / kSinkDim);
    EXPECT_EQ(sample.n_repeated, 0);
    EXPECT_EQ(stats.n_exact_searches, sample.n_exact);
}

// With fp32, fp16 and encoded samples, IndexHNSW searches the sample vectors
// that the step asks for: the links are the same as with RecordingLinkSample.
TEST(HNSWLevel0LinksFromSample, SampleSearchesGatherTheRequestedVectors) {
    const std::vector<float> nodes = sink_test_vectors(kSinkNodes, 1);
    // Values that fp16 represents exactly, so that every sample type decodes
    // to the same vectors.
    const std::vector<uint16_t> x16 = to_fp16(near_nodes(nodes, 40000, 2));
    std::vector<float> x(x16.size());
    for (size_t i = 0; i < x.size(); i++) {
        x[i] = faiss::decode_fp16(x16[i]);
    }
    const faiss::idx_t n = x.size() / kSinkDim;
    faiss::IndexScalarQuantizer codec(
            kSinkDim, faiss::ScalarQuantizer::QT_fp16);
    std::vector<uint8_t> codes(n * codec.sa_code_size());
    codec.sa_encode(n, x.data(), codes.data());

    const auto expected = linked_partial_sink(
            nodes, [&](faiss::IndexHNSWFlat& index, const auto& params) {
                RecordingLinkSample sample(index, x);
                std::unique_ptr<faiss::DistanceComputer> dis(
                        index.storage->get_distance_computer());
                return faiss::hnsw_add_level_0_links_from_sample(
                        index.hnsw, index.ntotal, *dis, false, sample, params);
            });

    EXPECT_EQ(
            linked_partial_sink(
                    nodes,
                    [&](faiss::IndexHNSWFlat& index, const auto& params) {
                        return index.add_level_0_links_from_sample(
                                n, x.data(), params);
                    }),
            expected);
    EXPECT_EQ(
            linked_partial_sink(
                    nodes,
                    [&](faiss::IndexHNSWFlat& index, const auto& params) {
                        return index.add_level_0_links_from_sample_ex(
                                n,
                                x16.data(),
                                faiss::NumericType::Float16,
                                params);
                    }),
            expected);
    EXPECT_EQ(
            linked_partial_sink(
                    nodes,
                    [&](faiss::IndexHNSWFlat& index, const auto& params) {
                        return index.add_level_0_links_from_sample_encoded(
                                n, codes.data(), &codec, params);
                    }),
            expected);
}

// IndexBinaryHNSW searches the sample vectors that the step asks for: the
// links are the same as with BinaryReferenceLinkSample.
TEST(HNSWLevel0LinksFromSample, BinarySampleSearchesGatherTheRequestedVectors) {
    const std::vector<uint8_t> nodes = random_codes(kSinkNodes, 1);
    const std::vector<uint8_t> x = near_codes(nodes, 40000, 2);
    auto linked = [&](bool reference) {
        faiss::IndexBinaryHNSW index(kBinaryDim, 32);
        index.hnsw.levels.assign(kSinkNodes, 1);
        index.add(kSinkNodes, nodes.data());
        plant_partial_sink(index.hnsw);
        const GreedyLinkParameters greedy;
        faiss::HNSWLevel0LinkStats stats;
        if (reference) {
            BinaryReferenceLinkSample sample(index, nodes, x);
            std::unique_ptr<faiss::DistanceComputer> dis(
                    index.get_distance_computer());
            stats = faiss::hnsw_add_level_0_links_from_sample(
                    index.hnsw,
                    index.ntotal,
                    *dis,
                    false,
                    sample,
                    greedy.params);
        } else {
            stats = index.add_level_0_links_from_sample(
                    x.size() / kBinaryCodeSize, x.data(), greedy.params);
        }
        EXPECT_GE(stats.n_rounds, 2);
        return std::vector<faiss::HNSW::storage_idx_t>(
                index.hnsw.neighbors.begin(), index.hnsw.neighbors.end());
    };

    EXPECT_EQ(linked(false), linked(true));
}

TEST(HNSWLevel0LinksFromSample, IVFTrainingLinksHNSWQuantizer) {
    EXPECT_GT(ivf_quantizer_recall(3), 0.95);
}

TEST(HNSWLevel0LinksFromSample, IVFTrainingLinksHNSWQuantizerFloat16) {
    EXPECT_GT(ivf_quantizer_recall(3, faiss::NumericType::Float16), 0.95);
}

TEST(HNSWLevel0LinksFromSample, IVFTrainingSkipsLinksWhenDisabled) {
    EXPECT_LT(ivf_quantizer_recall(0), 0.01);
}

TEST(HNSWLevel0LinksFromSample, BinaryIVFTrainingLinksHNSWQuantizer) {
    EXPECT_GT(binary_ivf_quantizer_recall(3), 0.95);
}

TEST(HNSWLevel0LinksFromSample, BinaryIVFTrainingSkipsLinksWhenDisabled) {
    EXPECT_LT(binary_ivf_quantizer_recall(0), 0.01);
}
