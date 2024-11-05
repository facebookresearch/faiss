/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cstddef>
#include <limits>
#include <random>
#include <unordered_set>
#include <vector>

#include <faiss/IndexHNSW.h>
#include <faiss/impl/HNSW.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/utils/random.h>
#include <faiss/utils/utils.h>

int reference_pop_min(faiss::HNSW::MinimaxHeap& heap, float* vmin_out) {
    assert(heap.k > 0);
    // returns min. This is an O(n) operation
    int i = heap.k - 1;
    while (i >= 0) {
        if (heap.ids[i] != -1)
            break;
        i--;
    }
    if (i == -1)
        return -1;
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
    if (vmin_out)
        *vmin_out = vmin;
    int ret = heap.ids[imin];
    heap.ids[imin] = -1;
    --heap.nvalid;

    return ret;
}

void test_popmin(int heap_size, int amount_to_put) {
    // create a heap
    faiss::HNSW::MinimaxHeap mm_heap(heap_size);

    using storage_idx_t = faiss::HNSW::storage_idx_t;

    std::default_random_engine rng(123 + heap_size * amount_to_put);
    std::uniform_int_distribution<storage_idx_t> u(0, 65536);
    std::uniform_real_distribution<float> uf(0, 1);

    // generate random unique indices
    std::unordered_set<storage_idx_t> indices;
    while (indices.size() < amount_to_put) {
        const storage_idx_t index = u(rng);
        indices.insert(index);
    }

    // put ones into the heap
    for (const auto index : indices) {
        float distance = uf(rng);
        if (distance >= 0.7f) {
            // add infinity values from time to time
            distance = std::numeric_limits<float>::infinity();
        }
        mm_heap.push(index, distance);
    }

    // clone the heap
    faiss::HNSW::MinimaxHeap cloned_mm_heap = mm_heap;

    // takes ones out one by one
    while (mm_heap.size() > 0) {
        // compare heaps
        ASSERT_EQ(mm_heap.n, cloned_mm_heap.n);
        ASSERT_EQ(mm_heap.k, cloned_mm_heap.k);
        ASSERT_EQ(mm_heap.nvalid, cloned_mm_heap.nvalid);
        ASSERT_EQ(mm_heap.ids, cloned_mm_heap.ids);
        ASSERT_EQ(mm_heap.dis, cloned_mm_heap.dis);

        // use the reference pop_min for the cloned heap
        float cloned_vmin_dis = std::numeric_limits<float>::quiet_NaN();
        storage_idx_t cloned_vmin_idx =
                reference_pop_min(cloned_mm_heap, &cloned_vmin_dis);

        float vmin_dis = std::numeric_limits<float>::quiet_NaN();
        storage_idx_t vmin_idx = mm_heap.pop_min(&vmin_dis);

        // compare returns
        ASSERT_EQ(vmin_dis, cloned_vmin_dis);
        ASSERT_EQ(vmin_idx, cloned_vmin_idx);
    }

    // compare heaps again
    ASSERT_EQ(mm_heap.n, cloned_mm_heap.n);
    ASSERT_EQ(mm_heap.k, cloned_mm_heap.k);
    ASSERT_EQ(mm_heap.nvalid, cloned_mm_heap.nvalid);
    ASSERT_EQ(mm_heap.ids, cloned_mm_heap.ids);
    ASSERT_EQ(mm_heap.dis, cloned_mm_heap.dis);
}

void test_popmin_identical_distances(
        int heap_size,
        int amount_to_put,
        const float distance) {
    // create a heap
    faiss::HNSW::MinimaxHeap mm_heap(heap_size);

    using storage_idx_t = faiss::HNSW::storage_idx_t;

    std::default_random_engine rng(123 + heap_size * amount_to_put);
    std::uniform_int_distribution<storage_idx_t> u(0, 65536);

    // generate random unique indices
    std::unordered_set<storage_idx_t> indices;
    while (indices.size() < amount_to_put) {
        const storage_idx_t index = u(rng);
        indices.insert(index);
    }

    // put ones into the heap
    for (const auto index : indices) {
        mm_heap.push(index, distance);
    }

    // clone the heap
    faiss::HNSW::MinimaxHeap cloned_mm_heap = mm_heap;

    // takes ones out one by one
    while (mm_heap.size() > 0) {
        // compare heaps
        ASSERT_EQ(mm_heap.n, cloned_mm_heap.n);
        ASSERT_EQ(mm_heap.k, cloned_mm_heap.k);
        ASSERT_EQ(mm_heap.nvalid, cloned_mm_heap.nvalid);
        ASSERT_EQ(mm_heap.ids, cloned_mm_heap.ids);
        ASSERT_EQ(mm_heap.dis, cloned_mm_heap.dis);

        // use the reference pop_min for the cloned heap
        float cloned_vmin_dis = std::numeric_limits<float>::quiet_NaN();
        storage_idx_t cloned_vmin_idx =
                reference_pop_min(cloned_mm_heap, &cloned_vmin_dis);

        float vmin_dis = std::numeric_limits<float>::quiet_NaN();
        storage_idx_t vmin_idx = mm_heap.pop_min(&vmin_dis);

        // compare returns
        ASSERT_EQ(vmin_dis, cloned_vmin_dis);
        ASSERT_EQ(vmin_idx, cloned_vmin_idx);
    }

    // compare heaps again
    ASSERT_EQ(mm_heap.n, cloned_mm_heap.n);
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
            test_popmin_identical_distances(size, amount, 1.0f);
        }
    }
}

TEST(HNSW, Test_popmin_infinite_distances) {
    std::vector<size_t> sizes = {1, 2, 3, 4, 5, 7, 9, 11, 16, 27, 32};
    for (const size_t size : sizes) {
        for (size_t amount = size; amount > 0; amount /= 2) {
            test_popmin_identical_distances(
                    size, amount, std::numeric_limits<float>::infinity());
        }
    }
}

class HNSWTest : public testing::Test {
   protected:
    HNSWTest() {
        xb = std::make_unique<std::vector<float>>(d * nb);
        xb->reserve(d * nb);
        faiss::float_rand(xb->data(), d * nb, 12345);
        index = std::make_unique<faiss::IndexHNSWFlat>(d, M);
        index->add(nb, xb->data());
        xq = std::unique_ptr<std::vector<float>>(
                new std::vector<float>(d * nq));
        xq->reserve(d * nq);
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
        faiss::ResultHandler<faiss::HNSW::C>& res,
        faiss::HNSW::MinimaxHeap& candidates,
        faiss::VisitedTable& vt,
        faiss::HNSWStats& stats,
        int level,
        int nres_in,
        const faiss::SearchParametersHNSW* params) {
    int nres = nres_in;
    int ndis = 0;

    // can be overridden by search params
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

        // a reference version
        for (size_t j = begin; j < end; j++) {
            int v1 = hnsw.neighbors[j];
            if (v1 < 0)
                break;
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

faiss::HNSWStats reference_greedy_update_nearest(
        const faiss::HNSW& hnsw,
        faiss::DistanceComputer& qdis,
        int level,
        faiss::HNSW::storage_idx_t& nearest,
        float& d_nearest) {
    faiss::HNSWStats stats;

    for (;;) {
        faiss::HNSW::storage_idx_t prev_nearest = nearest;

        size_t begin, end;
        hnsw.neighbor_range(nearest, level, &begin, &end);

        size_t ndis = 0;

        for (size_t i = begin; i < end; i++) {
            faiss::HNSW::storage_idx_t v = hnsw.neighbors[i];
            if (v < 0)
                break;
            ndis += 1;
            float dis = qdis(v);
            if (dis < d_nearest) {
                nearest = v;
                d_nearest = dis;
            }
        }
        // update stats
        stats.ndis += ndis;
        stats.nhops += 1;

        if (nearest == prev_nearest) {
            return stats;
        }
    }
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

            if (top_candidates.top().first > d1 || top_candidates.size() < ef) {
                candidates.emplace(d1, v1);
                top_candidates.emplace(d1, v1);

                if (top_candidates.size() > ef) {
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
    omp_set_num_threads(1);
    auto nearest = index->hnsw.entry_point;
    float d_nearest = (*dis)(nearest);
    auto node = faiss::HNSW::Node(d_nearest, nearest);
    faiss::VisitedTable vt(index->ntotal);
    faiss::HNSWStats stats;

    // actual version
    auto top_candidates = faiss::search_from_candidate_unbounded(
            index->hnsw, node, *dis, k, &vt, stats);

    auto reference_nearest = index->hnsw.entry_point;
    float reference_d_nearest = (*dis)(nearest);
    auto reference_node =
            faiss::HNSW::Node(reference_d_nearest, reference_nearest);
    faiss::VisitedTable reference_vt(index->ntotal);
    faiss::HNSWStats reference_stats;

    // reference version
    auto reference_top_candidates = reference_search_from_candidate_unbounded(
            index->hnsw,
            reference_node,
            *dis,
            k,
            &reference_vt,
            reference_stats);
    EXPECT_EQ(stats.ndis, reference_stats.ndis);
    EXPECT_EQ(stats.nhops, reference_stats.nhops);
    EXPECT_EQ(stats.n1, reference_stats.n1);
    EXPECT_EQ(stats.n2, reference_stats.n2);
    EXPECT_EQ(top_candidates.size(), reference_top_candidates.size());
}

TEST_F(HNSWTest, TEST_greedy_update_nearest) {
    omp_set_num_threads(1);

    auto nearest = index->hnsw.entry_point;
    float d_nearest = (*dis)(nearest);
    auto reference_nearest = index->hnsw.entry_point;
    float reference_d_nearest = (*dis)(reference_nearest);

    // actual version
    auto stats = faiss::greedy_update_nearest(
            index->hnsw, *dis, 0, nearest, d_nearest);

    // reference version
    auto reference_stats = reference_greedy_update_nearest(
            index->hnsw, *dis, 0, reference_nearest, reference_d_nearest);
    EXPECT_EQ(stats.ndis, reference_stats.ndis);
    EXPECT_EQ(stats.nhops, reference_stats.nhops);
    EXPECT_EQ(stats.n1, reference_stats.n1);
    EXPECT_EQ(stats.n2, reference_stats.n2);
    EXPECT_NEAR(d_nearest, reference_d_nearest, 0.01);
    EXPECT_EQ(nearest, reference_nearest);
}

TEST_F(HNSWTest, TEST_search_from_candidates) {
    omp_set_num_threads(1);

    std::vector<faiss::idx_t> I(k * nq);
    std::vector<float> D(k * nq);
    std::vector<faiss::idx_t> reference_I(k * nq);
    std::vector<float> reference_D(k * nq);
    using RH = faiss::HeapBlockResultHandler<faiss::HNSW::C>;

    faiss::VisitedTable vt(index->ntotal);
    faiss::VisitedTable reference_vt(index->ntotal);
    int num_candidates = 10;
    faiss::HNSW::MinimaxHeap candidates(num_candidates);
    faiss::HNSW::MinimaxHeap reference_candidates(num_candidates);

    for (int i = 0; i < num_candidates; i++) {
        vt.set(i);
        reference_vt.set(i);
        candidates.push(i, (*dis)(i));
        reference_candidates.push(i, (*dis)(i));
    }

    faiss::HNSWStats stats;
    RH bres(nq, D.data(), I.data(), k);
    faiss::HeapBlockResultHandler<faiss::HNSW::C>::SingleResultHandler res(
            bres);

    res.begin(0);
    faiss::search_from_candidates(
            index->hnsw, *dis, res, candidates, vt, stats, 0, 0, nullptr);
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
            reference_vt,
            reference_stats,
            0,
            0,
            nullptr);
    reference_res.end();
    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < k; j++) {
            EXPECT_NEAR(I[i * k + j], reference_I[i * k + j], 0.1);
            EXPECT_NEAR(D[i * k + j], reference_D[i * k + j], 0.1);
        }
    }
    EXPECT_EQ(reference_stats.ndis, stats.ndis);
    EXPECT_EQ(reference_stats.nhops, stats.nhops);
    EXPECT_EQ(reference_stats.n1, stats.n1);
    EXPECT_EQ(reference_stats.n2, stats.n2);
}
