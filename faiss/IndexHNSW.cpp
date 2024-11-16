/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexHNSW.h>

#include <omp.h>
#include <cassert>
#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include <limits>
#include <memory>
#include <queue>
#include <random>

#include <cstdint>

#include <faiss/Index2Layer.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/utils/random.h>
#include <faiss/utils/sorting.h>

namespace faiss {

using MinimaxHeap = HNSW::MinimaxHeap;
using storage_idx_t = HNSW::storage_idx_t;
using NodeDistFarther = HNSW::NodeDistFarther;

HNSWStats hnsw_stats;

/**************************************************************
 * add / search blocks of descriptors
 **************************************************************/

namespace {

DistanceComputer* storage_distance_computer(const Index* storage) {
    if (is_similarity_metric(storage->metric_type)) {
        return new NegativeDistanceComputer(storage->get_distance_computer());
    } else {
        return storage->get_distance_computer();
    }
}

void hnsw_add_vertices(
        IndexHNSW& index_hnsw,
        size_t n0,
        size_t n,
        const float* x,
        bool verbose,
        bool preset_levels = false) {
    size_t d = index_hnsw.d;
    HNSW& hnsw = index_hnsw.hnsw;
    size_t ntotal = n0 + n;
    double t0 = getmillisecs();
    if (verbose) {
        printf("hnsw_add_vertices: adding %zd elements on top of %zd "
               "(preset_levels=%d)\n",
               n,
               n0,
               int(preset_levels));
    }

    if (n == 0) {
        return;
    }

    int max_level = hnsw.prepare_level_tab(n, preset_levels);

    if (verbose) {
        printf("  max_level = %d\n", max_level);
    }

    std::vector<omp_lock_t> locks(ntotal);
    for (int i = 0; i < ntotal; i++)
        omp_init_lock(&locks[i]);

    // add vectors from highest to lowest level
    std::vector<int> hist;
    std::vector<int> order(n);

    { // make buckets with vectors of the same level

        // build histogram
        for (int i = 0; i < n; i++) {
            storage_idx_t pt_id = i + n0;
            int pt_level = hnsw.levels[pt_id] - 1;
            while (pt_level >= hist.size())
                hist.push_back(0);
            hist[pt_level]++;
        }

        // accumulate
        std::vector<int> offsets(hist.size() + 1, 0);
        for (int i = 0; i < hist.size() - 1; i++) {
            offsets[i + 1] = offsets[i] + hist[i];
        }

        // bucket sort
        for (int i = 0; i < n; i++) {
            storage_idx_t pt_id = i + n0;
            int pt_level = hnsw.levels[pt_id] - 1;
            order[offsets[pt_level]++] = pt_id;
        }
    }

    idx_t check_period = InterruptCallback::get_period_hint(
            max_level * index_hnsw.d * hnsw.efConstruction);

    { // perform add
        RandomGenerator rng2(789);

        int i1 = n;

        for (int pt_level = hist.size() - 1;
             pt_level >= !index_hnsw.init_level0;
             pt_level--) {
            int i0 = i1 - hist[pt_level];

            if (verbose) {
                printf("Adding %d elements at level %d\n", i1 - i0, pt_level);
            }

            // random permutation to get rid of dataset order bias
            for (int j = i0; j < i1; j++)
                std::swap(order[j], order[j + rng2.rand_int(i1 - j)]);

            bool interrupt = false;

#pragma omp parallel if (i1 > i0 + 100)
            {
                VisitedTable vt(ntotal);

                std::unique_ptr<DistanceComputer> dis(
                        storage_distance_computer(index_hnsw.storage));
                int prev_display =
                        verbose && omp_get_thread_num() == 0 ? 0 : -1;
                size_t counter = 0;

                // here we should do schedule(dynamic) but this segfaults for
                // some versions of LLVM. The performance impact should not be
                // too large when (i1 - i0) / num_threads >> 1
#pragma omp for schedule(static)
                for (int i = i0; i < i1; i++) {
                    storage_idx_t pt_id = order[i];
                    dis->set_query(x + (pt_id - n0) * d);

                    // cannot break
                    if (interrupt) {
                        continue;
                    }

                    hnsw.add_with_locks(
                            *dis,
                            pt_level,
                            pt_id,
                            locks,
                            vt,
                            index_hnsw.keep_max_size_level0 && (pt_level == 0));

                    if (prev_display >= 0 && i - i0 > prev_display + 10000) {
                        prev_display = i - i0;
                        printf("  %d / %d\r", i - i0, i1 - i0);
                        fflush(stdout);
                    }
                    if (counter % check_period == 0) {
                        if (InterruptCallback::is_interrupted()) {
                            interrupt = true;
                        }
                    }
                    counter++;
                }
            }
            if (interrupt) {
                FAISS_THROW_MSG("computation interrupted");
            }
            i1 = i0;
        }
        if (index_hnsw.init_level0) {
            FAISS_ASSERT(i1 == 0);
        } else {
            FAISS_ASSERT((i1 - hist[0]) == 0);
        }
    }
    if (verbose) {
        printf("Done in %.3f ms\n", getmillisecs() - t0);
    }

    for (int i = 0; i < ntotal; i++) {
        omp_destroy_lock(&locks[i]);
    }
}

} // namespace

/**************************************************************
 * IndexHNSW implementation
 **************************************************************/

IndexHNSW::IndexHNSW(int d, int M, MetricType metric)
        : Index(d, metric), hnsw(M) {}

IndexHNSW::IndexHNSW(Index* storage, int M)
        : Index(storage->d, storage->metric_type), hnsw(M), storage(storage) {}

IndexHNSW::~IndexHNSW() {
    if (own_fields) {
        delete storage;
    }
}

void IndexHNSW::train(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexHNSWFlat (or variants) instead of IndexHNSW directly");
    // hnsw structure does not require training
    storage->train(n, x);
    is_trained = true;
}

namespace {

template <class BlockResultHandler>
void hnsw_search(
        const IndexHNSW* index,
        idx_t n,
        const float* x,
        BlockResultHandler& bres,
        const SearchParameters* params_in) {
    FAISS_THROW_IF_NOT_MSG(
            index->storage,
            "No storage index, please use IndexHNSWFlat (or variants) "
            "instead of IndexHNSW directly");
    const SearchParametersHNSW* params = nullptr;
    const HNSW& hnsw = index->hnsw;

    int efSearch = hnsw.efSearch;
    if (params_in) {
        params = dynamic_cast<const SearchParametersHNSW*>(params_in);
        FAISS_THROW_IF_NOT_MSG(params, "params type invalid");
        efSearch = params->efSearch;
    }
    size_t n1 = 0, n2 = 0, ndis = 0, nhops = 0;

    idx_t check_period = InterruptCallback::get_period_hint(
            hnsw.max_level * index->d * efSearch);

    for (idx_t i0 = 0; i0 < n; i0 += check_period) {
        idx_t i1 = std::min(i0 + check_period, n);

#pragma omp parallel if (i1 - i0 > 1)
        {
            VisitedTable vt(index->ntotal);
            typename BlockResultHandler::SingleResultHandler res(bres);

            std::unique_ptr<DistanceComputer> dis(
                    storage_distance_computer(index->storage));

#pragma omp for reduction(+ : n1, n2, ndis, nhops) schedule(guided)
            for (idx_t i = i0; i < i1; i++) {
                res.begin(i);
                dis->set_query(x + i * index->d);

                HNSWStats stats = hnsw.search(*dis, res, vt, params);
                n1 += stats.n1;
                n2 += stats.n2;
                ndis += stats.ndis;
                nhops += stats.nhops;
                res.end();
            }
        }
        InterruptCallback::check();
    }

    hnsw_stats.combine({n1, n2, ndis, nhops});
}

} // anonymous namespace

void IndexHNSW::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params_in) const {
    FAISS_THROW_IF_NOT(k > 0);

    using RH = HeapBlockResultHandler<HNSW::C>;
    RH bres(n, distances, labels, k);

    hnsw_search(this, n, x, bres, params_in);

    if (is_similarity_metric(this->metric_type)) {
        // we need to revert the negated distances
        for (size_t i = 0; i < k * n; i++) {
            distances[i] = -distances[i];
        }
    }
}

void IndexHNSW::range_search(
        idx_t n,
        const float* x,
        float radius,
        RangeSearchResult* result,
        const SearchParameters* params) const {
    using RH = RangeSearchBlockResultHandler<HNSW::C>;
    RH bres(result, is_similarity_metric(metric_type) ? -radius : radius);

    hnsw_search(this, n, x, bres, params);

    if (is_similarity_metric(this->metric_type)) {
        // we need to revert the negated distances
        for (size_t i = 0; i < result->lims[result->nq]; i++) {
            result->distances[i] = -result->distances[i];
        }
    }
}

void IndexHNSW::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(
            storage,
            "Please use IndexHNSWFlat (or variants) instead of IndexHNSW directly");
    FAISS_THROW_IF_NOT(is_trained);
    int n0 = ntotal;
    storage->add(n, x);
    ntotal = storage->ntotal;

    hnsw_add_vertices(*this, n0, n, x, verbose, hnsw.levels.size() == ntotal);
}

void IndexHNSW::reset() {
    hnsw.reset();
    storage->reset();
    ntotal = 0;
}

void IndexHNSW::reconstruct(idx_t key, float* recons) const {
    storage->reconstruct(key, recons);
}

void IndexHNSW::shrink_level_0_neighbors(int new_size) {
#pragma omp parallel
    {
        std::unique_ptr<DistanceComputer> dis(
                storage_distance_computer(storage));

#pragma omp for
        for (idx_t i = 0; i < ntotal; i++) {
            size_t begin, end;
            hnsw.neighbor_range(i, 0, &begin, &end);

            std::priority_queue<NodeDistFarther> initial_list;

            for (size_t j = begin; j < end; j++) {
                int v1 = hnsw.neighbors[j];
                if (v1 < 0)
                    break;
                initial_list.emplace(dis->symmetric_dis(i, v1), v1);

                // initial_list.emplace(qdis(v1), v1);
            }

            std::vector<NodeDistFarther> shrunk_list;
            HNSW::shrink_neighbor_list(
                    *dis, initial_list, shrunk_list, new_size);

            for (size_t j = begin; j < end; j++) {
                if (j - begin < shrunk_list.size())
                    hnsw.neighbors[j] = shrunk_list[j - begin].id;
                else
                    hnsw.neighbors[j] = -1;
            }
        }
    }
}

void IndexHNSW::search_level_0(
        idx_t n,
        const float* x,
        idx_t k,
        const storage_idx_t* nearest,
        const float* nearest_d,
        float* distances,
        idx_t* labels,
        int nprobe,
        int search_type,
        const SearchParameters* params_in) const {
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(nprobe > 0);

    const SearchParametersHNSW* params = nullptr;

    if (params_in) {
        params = dynamic_cast<const SearchParametersHNSW*>(params_in);
        FAISS_THROW_IF_NOT_MSG(params, "params type invalid");
    }

    storage_idx_t ntotal = hnsw.levels.size();

    using RH = HeapBlockResultHandler<HNSW::C>;
    RH bres(n, distances, labels, k);

#pragma omp parallel
    {
        std::unique_ptr<DistanceComputer> qdis(
                storage_distance_computer(storage));
        HNSWStats search_stats;
        VisitedTable vt(ntotal);
        RH::SingleResultHandler res(bres);

#pragma omp for
        for (idx_t i = 0; i < n; i++) {
            res.begin(i);
            qdis->set_query(x + i * d);

            hnsw.search_level_0(
                    *qdis.get(),
                    res,
                    nprobe,
                    nearest + i * nprobe,
                    nearest_d + i * nprobe,
                    search_type,
                    search_stats,
                    vt,
                    params);
            res.end();
            vt.advance();
        }
#pragma omp critical
        { hnsw_stats.combine(search_stats); }
    }
    if (is_similarity_metric(this->metric_type)) {
// we need to revert the negated distances
#pragma omp parallel for
        for (int64_t i = 0; i < k * n; i++) {
            distances[i] = -distances[i];
        }
    }
}

void IndexHNSW::init_level_0_from_knngraph(
        int k,
        const float* D,
        const idx_t* I) {
    int dest_size = hnsw.nb_neighbors(0);

#pragma omp parallel for
    for (idx_t i = 0; i < ntotal; i++) {
        DistanceComputer* qdis = storage_distance_computer(storage);
        std::vector<float> vec(d);
        storage->reconstruct(i, vec.data());
        qdis->set_query(vec.data());

        std::priority_queue<NodeDistFarther> initial_list;

        for (size_t j = 0; j < k; j++) {
            int v1 = I[i * k + j];
            if (v1 == i)
                continue;
            if (v1 < 0)
                break;
            initial_list.emplace(D[i * k + j], v1);
        }

        std::vector<NodeDistFarther> shrunk_list;
        HNSW::shrink_neighbor_list(*qdis, initial_list, shrunk_list, dest_size);

        size_t begin, end;
        hnsw.neighbor_range(i, 0, &begin, &end);

        for (size_t j = begin; j < end; j++) {
            if (j - begin < shrunk_list.size())
                hnsw.neighbors[j] = shrunk_list[j - begin].id;
            else
                hnsw.neighbors[j] = -1;
        }
    }
}

void IndexHNSW::init_level_0_from_entry_points(
        int n,
        const storage_idx_t* points,
        const storage_idx_t* nearests) {
    std::vector<omp_lock_t> locks(ntotal);
    for (int i = 0; i < ntotal; i++)
        omp_init_lock(&locks[i]);

#pragma omp parallel
    {
        VisitedTable vt(ntotal);

        std::unique_ptr<DistanceComputer> dis(
                storage_distance_computer(storage));
        std::vector<float> vec(storage->d);

#pragma omp for schedule(dynamic)
        for (int i = 0; i < n; i++) {
            storage_idx_t pt_id = points[i];
            storage_idx_t nearest = nearests[i];
            storage->reconstruct(pt_id, vec.data());
            dis->set_query(vec.data());

            hnsw.add_links_starting_from(
                    *dis, pt_id, nearest, (*dis)(nearest), 0, locks.data(), vt);

            if (verbose && i % 10000 == 0) {
                printf("  %d / %d\r", i, n);
                fflush(stdout);
            }
        }
    }
    if (verbose) {
        printf("\n");
    }

    for (int i = 0; i < ntotal; i++)
        omp_destroy_lock(&locks[i]);
}

void IndexHNSW::reorder_links() {
    int M = hnsw.nb_neighbors(0);

#pragma omp parallel
    {
        std::vector<float> distances(M);
        std::vector<size_t> order(M);
        std::vector<storage_idx_t> tmp(M);
        std::unique_ptr<DistanceComputer> dis(
                storage_distance_computer(storage));

#pragma omp for
        for (storage_idx_t i = 0; i < ntotal; i++) {
            size_t begin, end;
            hnsw.neighbor_range(i, 0, &begin, &end);

            for (size_t j = begin; j < end; j++) {
                storage_idx_t nj = hnsw.neighbors[j];
                if (nj < 0) {
                    end = j;
                    break;
                }
                distances[j - begin] = dis->symmetric_dis(i, nj);
                tmp[j - begin] = nj;
            }

            fvec_argsort(end - begin, distances.data(), order.data());
            for (size_t j = begin; j < end; j++) {
                hnsw.neighbors[j] = tmp[order[j - begin]];
            }
        }
    }
}

void IndexHNSW::link_singletons() {
    printf("search for singletons\n");

    std::vector<bool> seen(ntotal);

    for (size_t i = 0; i < ntotal; i++) {
        size_t begin, end;
        hnsw.neighbor_range(i, 0, &begin, &end);
        for (size_t j = begin; j < end; j++) {
            storage_idx_t ni = hnsw.neighbors[j];
            if (ni >= 0)
                seen[ni] = true;
        }
    }

    int n_sing = 0, n_sing_l1 = 0;
    std::vector<storage_idx_t> singletons;
    for (storage_idx_t i = 0; i < ntotal; i++) {
        if (!seen[i]) {
            singletons.push_back(i);
            n_sing++;
            if (hnsw.levels[i] > 1)
                n_sing_l1++;
        }
    }

    printf("  Found %d / %" PRId64 " singletons (%d appear in a level above)\n",
           n_sing,
           ntotal,
           n_sing_l1);

    std::vector<float> recons(singletons.size() * d);
    for (int i = 0; i < singletons.size(); i++) {
        FAISS_ASSERT(!"not implemented");
    }
}

void IndexHNSW::permute_entries(const idx_t* perm) {
    auto flat_storage = dynamic_cast<IndexFlatCodes*>(storage);
    FAISS_THROW_IF_NOT_MSG(
            flat_storage, "don't know how to permute this index");
    flat_storage->permute_entries(perm);
    hnsw.permute_entries(perm);
}

DistanceComputer* IndexHNSW::get_distance_computer() const {
    return storage->get_distance_computer();
}

/**************************************************************
 * IndexHNSWFlat implementation
 **************************************************************/

IndexHNSWFlat::IndexHNSWFlat() {
    is_trained = true;
}

IndexHNSWFlat::IndexHNSWFlat(int d, int M, MetricType metric)
        : IndexHNSW(
                  (metric == METRIC_L2) ? new IndexFlatL2(d)
                                        : new IndexFlat(d, metric),
                  M) {
    own_fields = true;
    is_trained = true;
}

/**************************************************************
 * IndexHNSWPQ implementation
 **************************************************************/

IndexHNSWPQ::IndexHNSWPQ() = default;

IndexHNSWPQ::IndexHNSWPQ(
        int d,
        int pq_m,
        int M,
        int pq_nbits,
        MetricType metric)
        : IndexHNSW(new IndexPQ(d, pq_m, pq_nbits, metric), M) {
    own_fields = true;
    is_trained = false;
}

void IndexHNSWPQ::train(idx_t n, const float* x) {
    IndexHNSW::train(n, x);
    (dynamic_cast<IndexPQ*>(storage))->pq.compute_sdc_table();
}

/**************************************************************
 * IndexHNSWSQ implementation
 **************************************************************/

IndexHNSWSQ::IndexHNSWSQ(
        int d,
        ScalarQuantizer::QuantizerType qtype,
        int M,
        MetricType metric)
        : IndexHNSW(new IndexScalarQuantizer(d, qtype, metric), M) {
    is_trained = this->storage->is_trained;
    own_fields = true;
}

IndexHNSWSQ::IndexHNSWSQ() = default;

/**************************************************************
 * IndexHNSW2Level implementation
 **************************************************************/

IndexHNSW2Level::IndexHNSW2Level(
        Index* quantizer,
        size_t nlist,
        int m_pq,
        int M)
        : IndexHNSW(new Index2Layer(quantizer, nlist, m_pq), M) {
    own_fields = true;
    is_trained = false;
}

IndexHNSW2Level::IndexHNSW2Level() = default;

namespace {

// same as search_from_candidates but uses v
// visno -> is in result list
// visno + 1 -> in result list + in candidates
int search_from_candidates_2(
        const HNSW& hnsw,
        DistanceComputer& qdis,
        int k,
        idx_t* I,
        float* D,
        MinimaxHeap& candidates,
        VisitedTable& vt,
        HNSWStats& stats,
        int level,
        int nres_in = 0) {
    int nres = nres_in;
    for (int i = 0; i < candidates.size(); i++) {
        idx_t v1 = candidates.ids[i];
        FAISS_ASSERT(v1 >= 0);
        vt.visited[v1] = vt.visno + 1;
    }

    int nstep = 0;

    while (candidates.size() > 0) {
        float d0 = 0;
        int v0 = candidates.pop_min(&d0);

        size_t begin, end;
        hnsw.neighbor_range(v0, level, &begin, &end);

        for (size_t j = begin; j < end; j++) {
            int v1 = hnsw.neighbors[j];
            if (v1 < 0)
                break;
            if (vt.visited[v1] == vt.visno + 1) {
                // nothing to do
            } else {
                float d = qdis(v1);
                candidates.push(v1, d);

                // never seen before --> add to heap
                if (vt.visited[v1] < vt.visno) {
                    if (nres < k) {
                        faiss::maxheap_push(++nres, D, I, d, v1);
                    } else if (d < D[0]) {
                        faiss::maxheap_replace_top(nres, D, I, d, v1);
                    }
                }
                vt.visited[v1] = vt.visno + 1;
            }
        }

        nstep++;
        if (nstep > hnsw.efSearch) {
            break;
        }
    }

    stats.n1++;
    if (candidates.size() == 0)
        stats.n2++;

    return nres;
}

} // namespace

void IndexHNSW2Level::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT_MSG(
            !params, "search params not supported for this index");

    if (dynamic_cast<const Index2Layer*>(storage)) {
        IndexHNSW::search(n, x, k, distances, labels);

    } else { // "mixed" search
        size_t n1 = 0, n2 = 0, ndis = 0, nhops = 0;

        const IndexIVFPQ* index_ivfpq =
                dynamic_cast<const IndexIVFPQ*>(storage);

        int nprobe = index_ivfpq->nprobe;

        std::unique_ptr<idx_t[]> coarse_assign(new idx_t[n * nprobe]);
        std::unique_ptr<float[]> coarse_dis(new float[n * nprobe]);

        index_ivfpq->quantizer->search(
                n, x, nprobe, coarse_dis.get(), coarse_assign.get());

        index_ivfpq->search_preassigned(
                n,
                x,
                k,
                coarse_assign.get(),
                coarse_dis.get(),
                distances,
                labels,
                false);

#pragma omp parallel
        {
            VisitedTable vt(ntotal);
            std::unique_ptr<DistanceComputer> dis(
                    storage_distance_computer(storage));

            constexpr int candidates_size = 1;
            MinimaxHeap candidates(candidates_size);

#pragma omp for reduction(+ : n1, n2, ndis, nhops)
            for (idx_t i = 0; i < n; i++) {
                idx_t* idxi = labels + i * k;
                float* simi = distances + i * k;
                dis->set_query(x + i * d);

                // mark all inverted list elements as visited

                for (int j = 0; j < nprobe; j++) {
                    idx_t key = coarse_assign[j + i * nprobe];
                    if (key < 0)
                        break;
                    size_t list_length = index_ivfpq->get_list_size(key);
                    const idx_t* ids = index_ivfpq->invlists->get_ids(key);

                    for (int jj = 0; jj < list_length; jj++) {
                        vt.set(ids[jj]);
                    }
                }

                candidates.clear();

                for (int j = 0; j < k; j++) {
                    if (idxi[j] < 0)
                        break;
                    candidates.push(idxi[j], simi[j]);
                }

                // reorder from sorted to heap
                maxheap_heapify(k, simi, idxi, simi, idxi, k);

                HNSWStats search_stats;
                search_from_candidates_2(
                        hnsw,
                        *dis,
                        k,
                        idxi,
                        simi,
                        candidates,
                        vt,
                        search_stats,
                        0,
                        k);
                n1 += search_stats.n1;
                n2 += search_stats.n2;
                ndis += search_stats.ndis;
                nhops += search_stats.nhops;

                vt.advance();
                vt.advance();

                maxheap_reorder(k, simi, idxi);
            }
        }

        hnsw_stats.combine({n1, n2, ndis, nhops});
    }
}

void IndexHNSW2Level::flip_to_ivf() {
    Index2Layer* storage2l = dynamic_cast<Index2Layer*>(storage);

    FAISS_THROW_IF_NOT(storage2l);

    IndexIVFPQ* index_ivfpq = new IndexIVFPQ(
            storage2l->q1.quantizer,
            d,
            storage2l->q1.nlist,
            storage2l->pq.M,
            8);
    index_ivfpq->pq = storage2l->pq;
    index_ivfpq->is_trained = storage2l->is_trained;
    index_ivfpq->precompute_table();
    index_ivfpq->own_fields = storage2l->q1.own_fields;
    storage2l->transfer_to_IVFPQ(*index_ivfpq);
    index_ivfpq->make_direct_map(true);

    storage = index_ivfpq;
    delete storage2l;
}

/**************************************************************
 * IndexHNSWCagra implementation
 **************************************************************/

IndexHNSWCagra::IndexHNSWCagra() {
    is_trained = true;
}

IndexHNSWCagra::IndexHNSWCagra(int d, int M, MetricType metric)
        : IndexHNSW(
                  (metric == METRIC_L2)
                          ? static_cast<IndexFlat*>(new IndexFlatL2(d))
                          : static_cast<IndexFlat*>(new IndexFlatIP(d)),
                  M) {
    FAISS_THROW_IF_NOT_MSG(
            ((metric == METRIC_L2) || (metric == METRIC_INNER_PRODUCT)),
            "unsupported metric type for IndexHNSWCagra");
    own_fields = true;
    is_trained = true;
    init_level0 = true;
    keep_max_size_level0 = true;
}

void IndexHNSWCagra::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT_MSG(
            !base_level_only,
            "Cannot add vectors when base_level_only is set to True");

    IndexHNSW::add(n, x);
}

void IndexHNSWCagra::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    if (!base_level_only) {
        IndexHNSW::search(n, x, k, distances, labels, params);
    } else {
        std::vector<storage_idx_t> nearest(n);
        std::vector<float> nearest_d(n);

#pragma omp for
        for (idx_t i = 0; i < n; i++) {
            std::unique_ptr<DistanceComputer> dis(
                    storage_distance_computer(this->storage));
            dis->set_query(x + i * d);
            nearest[i] = -1;
            nearest_d[i] = std::numeric_limits<float>::max();

            std::random_device rd;
            std::mt19937 gen(rd());
            std::uniform_int_distribution<idx_t> distrib(0, this->ntotal - 1);

            for (idx_t j = 0; j < num_base_level_search_entrypoints; j++) {
                auto idx = distrib(gen);
                auto distance = (*dis)(idx);
                if (distance < nearest_d[i]) {
                    nearest[i] = idx;
                    nearest_d[i] = distance;
                }
            }
            FAISS_THROW_IF_NOT_MSG(
                    nearest[i] >= 0, "Could not find a valid entrypoint.");
        }

        search_level_0(
                n,
                x,
                k,
                nearest.data(),
                nearest_d.data(),
                distances,
                labels,
                1, // n_probes
                1, // search_type
                params);
    }
}

} // namespace faiss
