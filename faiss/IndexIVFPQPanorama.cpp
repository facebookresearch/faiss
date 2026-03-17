#include <faiss/IndexIVFPQPanorama.h>
#include <omp.h>
#include <cstdint>
#include <memory>
#include <mutex>

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <iostream>
#include <limits>
#include <numeric>

#include <faiss/utils/hamming.h>
#include <faiss/utils/utils.h>

#include <faiss/IndexFlat.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/CodePacker.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>

namespace faiss {

static uint64_t total_active = 0;
static uint64_t total_points = 0;

IndexIVFPQPanorama::IndexIVFPQPanorama(
        Index* quantizer,
        size_t d,
        size_t nlist,
        size_t M,
        size_t nbits_per_idx,
        int n_levels,
        float epsilon,
        size_t batch_size,
        MetricType metric,
        bool own_invlists)
        : IndexIVFPQ(
                  quantizer,
                  d,
                  nlist,
                  M,
                  nbits_per_idx,
                  metric,
                  own_invlists),
          n_levels(n_levels),
          added(false),
          chunk_size(code_size / n_levels),
          levels_size(d / n_levels),
          nbits_per_idx(nbits_per_idx),
          m_level_width(M / n_levels),
          epsilon(epsilon),
          batch_size(batch_size) {
    FAISS_ASSERT(M % n_levels == 0);
    FAISS_ASSERT(batch_size % 64 == 0);

    printf("N levels = %d\n", n_levels);
    printf("M = code_size = %zu\n", M);
    printf("Nbits per idx = %u (fixed)\n", 8);
    printf("Nlist = %zu\n", nlist);
    printf("Batch size = %zuB\n", batch_size);

    FAISS_ASSERT(m_level_width > 0);
    FAISS_ASSERT(nbits_per_idx == 8);
    FAISS_ASSERT(M == code_size);
    FAISS_ASSERT(metric == METRIC_L2);
}

void IndexIVFPQPanorama::add(idx_t n, const float* x) {
    FAISS_ASSERT(!added);
    added = true;

    num_points = n;
    IndexIVFPQ::add(n, x);

    size_t new_n = 0;
    column_offsets = new size_t[nlist];
    for (size_t i = 0; i < nlist; i++) {
        column_offsets[i] = new_n;
        size_t batch_n = (invlists->list_size(i) + batch_size - 1) / batch_size;
        size_t rounded_n = batch_n * batch_size;
        new_n += rounded_n * code_size;
    }

    column_storage = new uint8_t[code_size * new_n];

    for (size_t list_no = 0; list_no < nlist; list_no++) {
        size_t col_offset = column_offsets[list_no];
        size_t list_size = invlists->list_size(list_no);
        size_t n_batches = (list_size + batch_size - 1) / batch_size;
        for (size_t batch_no = 0; batch_no < n_batches; batch_no++) {
            size_t batch_offset = batch_no * batch_size * code_size;
            size_t curr_batch_size =
                    std::min(list_size - batch_no * batch_size, batch_size);
            for (size_t m = 0; m < pq.M; m++) {
                size_t m_offset = m * batch_size;
                for (size_t point_idx = 0; point_idx < batch_size;
                     point_idx++) {
                    uint8_t* dest = column_storage + col_offset + batch_offset +
                            m_offset + point_idx;
                    const uint8_t* codes = invlists->get_codes(list_no);

                    if (point_idx < curr_batch_size) {
                        const uint8_t* src = codes + batch_offset +
                                point_idx * code_size + m;
                        memcpy(dest, src, 1);
                    } else {
                        *dest = 0;
                    }
                }
            }
        }
    }

    cum_sums = new float[(n_levels + 1) * n];
    cum_sum_offsets = new size_t[nlist];

    init_exact_distances = new float[n];
    init_exact_distances_offsets = new size_t[nlist];

    size_t cum_size = 0;
    size_t init_size = 0;
    for (size_t list_no = 0; list_no < nlist; list_no++) {
        cum_sum_offsets[list_no] = cum_size;
        cum_size += invlists->list_size(list_no) * (n_levels + 1);

        init_exact_distances_offsets[list_no] = init_size;
        init_size += invlists->list_size(list_no);
    }

    for (size_t list_no = 0; list_no < nlist; list_no++) {
        const idx_t* idx = invlists->get_ids(list_no);
        size_t list_size = invlists->list_size(list_no);

        std::vector<float> centroid(d);
        quantizer->reconstruct(list_no, centroid.data());

        size_t n_batches = (list_size + batch_size - 1) / batch_size;

        for (size_t batch_no = 0; batch_no < n_batches; batch_no++) {
            size_t b_offset = batch_no * batch_size;
            size_t curr_batch_size =
                    std::min(list_size - batch_no * batch_size, batch_size);

            for (size_t point_idx = 0; point_idx < curr_batch_size;
                 point_idx++) {
                float init_exact_distance = 0.0f;

                std::vector<float> vector(d);
                const uint8_t* code =
                        invlists->get_single_code(list_no, b_offset + point_idx);
                pq.decode(code, vector.data());

                std::vector<float> suffix_sums(d + 1);
                suffix_sums[d] = 0.0f;

                for (int j = d - 1; j >= 0; j--) {
                    init_exact_distance +=
                            vector[j] * vector[j] + 2 * vector[j] * centroid[j];
                    float squaredVal = vector[j] * vector[j];
                    suffix_sums[j] = suffix_sums[j + 1] + squaredVal;
                }

                for (int level = 0; level < n_levels; level++) {
                    int start_idx = level * levels_size;
                    size_t offset = cum_sum_offsets[list_no] +
                            b_offset * (n_levels + 1) +
                            level * curr_batch_size + point_idx;
                    if (start_idx < (int)d) {
                        cum_sums[offset] = sqrt(suffix_sums[start_idx]);
                    } else {
                        cum_sums[offset] = 0.0f;
                    }
                }

                size_t offset = cum_sum_offsets[list_no] +
                        b_offset * (n_levels + 1) +
                        n_levels * curr_batch_size + point_idx;
                cum_sums[offset] = 0.0f;

                size_t init_offset = init_exact_distances_offsets[list_no];
                init_exact_distances[init_offset + b_offset + point_idx] =
                        init_exact_distance;
            }
        }
    }
}

void IndexIVFPQPanorama::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params_in) const {
    FAISS_THROW_IF_NOT(k > 0);
    const IVFSearchParameters* params = nullptr;
    if (params_in) {
        params = dynamic_cast<const IVFSearchParameters*>(params_in);
        FAISS_THROW_IF_NOT_MSG(params, "IndexIVF params have incorrect type");
    }
    const size_t nprobe =
            std::min(nlist, params ? params->nprobe : this->nprobe);
    FAISS_THROW_IF_NOT(nprobe > 0);

    auto sub_search_func = [this, k, nprobe, params](
                                   idx_t n,
                                   const float* x,
                                   float* distances,
                                   idx_t* labels,
                                   IndexIVFStats* ivf_stats) {
        std::unique_ptr<idx_t[]> idx(new idx_t[n * nprobe]);
        std::unique_ptr<float[]> coarse_dis(new float[n * nprobe]);

        quantizer->search(
                n,
                x,
                nprobe,
                coarse_dis.get(),
                idx.get(),
                params ? params->quantizer_params : nullptr);

        invlists->prefetch_lists(idx.get(), n * nprobe);

        search_preassigned(
                n,
                x,
                k,
                idx.get(),
                coarse_dis.get(),
                distances,
                labels,
                false,
                params,
                ivf_stats);
    };

    if ((parallel_mode & ~PARALLEL_MODE_NO_HEAP_INIT) == 0) {
        int nt = std::min(omp_get_max_threads(), int(n));
        std::vector<IndexIVFStats> stats(nt);
        std::mutex exception_mutex;
        std::string exception_string;

#pragma omp parallel for if (nt > 1)
        for (idx_t slice = 0; slice < nt; slice++) {
            IndexIVFStats local_stats;
            idx_t i0 = n * slice / nt;
            idx_t i1 = n * (slice + 1) / nt;
            if (i1 > i0) {
                try {
                    sub_search_func(
                            i1 - i0,
                            x + i0 * d,
                            distances + i0 * k,
                            labels + i0 * k,
                            &stats[slice]);
                } catch (const std::exception& e) {
                    std::lock_guard<std::mutex> lock(exception_mutex);
                    exception_string = e.what();
                }
            }
        }

        if (!exception_string.empty()) {
            FAISS_THROW_FMT(
                    "search error: %s", exception_string.c_str());
        }
    } else {
        sub_search_func(n, x, distances, labels, &indexIVF_stats);
    }
}

void IndexIVFPQPanorama::search_preassigned(
        idx_t n,
        const float* x,
        idx_t k,
        const idx_t* keys,
        const float* coarse_dis,
        float* distances,
        idx_t* labels,
        bool store_pairs,
        const IVFSearchParameters* params,
        IndexIVFStats* ivf_stats) const {
    FAISS_THROW_IF_NOT(k > 0);

    idx_t nprobe = params ? params->nprobe : this->nprobe;
    nprobe = std::min((idx_t)nlist, nprobe);
    FAISS_THROW_IF_NOT(nprobe > 0);

    const idx_t unlimited_list_size = std::numeric_limits<idx_t>::max();
    idx_t max_codes = params ? params->max_codes : this->max_codes;
    IDSelector* sel = params ? params->sel : nullptr;
    const IDSelectorRange* selr = dynamic_cast<const IDSelectorRange*>(sel);
    if (selr) {
        if (selr->assume_sorted) {
            sel = nullptr;
        } else {
            selr = nullptr;
        }
    }

    FAISS_THROW_IF_NOT_MSG(
            !(sel && store_pairs),
            "selector and store_pairs cannot be combined");

    FAISS_THROW_IF_NOT_MSG(
            !invlists->use_iterator || (max_codes == 0 && store_pairs == false),
            "iterable inverted lists don't support max_codes and store_pairs");

    size_t nlistv = 0, ndis = 0, nheap = 0;

    using HeapForIP = CMin<float, idx_t>;
    using HeapForL2 = CMax<float, idx_t>;

    bool interrupt = false;
    std::mutex exception_mutex;
    std::string exception_string;

    int pmode = this->parallel_mode & ~PARALLEL_MODE_NO_HEAP_INIT;
    bool do_heap_init = !(this->parallel_mode & PARALLEL_MODE_NO_HEAP_INIT);

    FAISS_THROW_IF_NOT_MSG(
            max_codes == 0 || pmode == 0 || pmode == 3,
            "max_codes supported only for parallel_mode = 0 or 3");

    if (max_codes == 0) {
        max_codes = unlimited_list_size;
    }

    [[maybe_unused]] bool do_parallel = omp_get_max_threads() >= 2 &&
            (pmode == 0           ? false
                     : pmode == 3 ? n > 1
                     : pmode == 1 ? nprobe > 1
                                  : nprobe * n > 1);

    void* inverted_list_context =
            params ? params->inverted_list_context : nullptr;

    const size_t sim_table_size = pq.ksub * pq.M;
    std::vector<float> sim_table_cache(nprobe * sim_table_size);
    std::vector<float> dis0s_cache(nprobe);

    std::vector<float> suffixSums(d + 1);
    std::vector<float> query_cum_norms(n_levels + 1);
    std::vector<float> query(d);
    std::vector<float> exact_distances(batch_size);
    std::vector<uint8_t> bitset(batch_size);
    std::vector<uint32_t> active_indices(batch_size);
    std::vector<uint8_t> compressed_codes(batch_size * chunk_size);

#pragma omp parallel if (do_parallel) reduction(+ : nlistv, ndis, nheap)
    {
        std::unique_ptr<InvertedListScanner> scanner(
                get_InvertedListScanner(store_pairs, sel, params));

        auto init_result = [&](float* simi, idx_t* idxi) {
            if (!do_heap_init)
                return;
            if (metric_type == METRIC_INNER_PRODUCT) {
                heap_heapify<HeapForIP>(k, simi, idxi);
            } else {
                heap_heapify<HeapForL2>(k, simi, idxi);
            }
        };

        auto reorder_result = [&](float* simi, idx_t* idxi) {
            if (!do_heap_init)
                return;
            if (metric_type == METRIC_INNER_PRODUCT) {
                heap_reorder<HeapForIP>(k, simi, idxi);
            } else {
                heap_reorder<HeapForL2>(k, simi, idxi);
            }
        };

        FAISS_ASSERT(pmode == 0);
        if (pmode == 0) {
#pragma omp for
            for (idx_t i = 0; i < n; i++) {
                if (interrupt) {
                    continue;
                }

                scanner->set_query(x + i * d);
                suffixSums[d] = 0.0f;

                const float* q = x + i * d;

                for (int j = d - 1; j >= 0; --j) {
                    float squaredVal = q[j] * q[j];
                    suffixSums[j] = suffixSums[j + 1] + squaredVal;
                }

                for (int level_idx = 0; level_idx < n_levels; level_idx++) {
                    int startIdx = level_idx * levels_size;
                    if (startIdx < (int)d) {
                        query_cum_norms[level_idx] = sqrt(suffixSums[startIdx]);
                    } else {
                        query_cum_norms[level_idx] = 0.0f;
                    }
                }
                query_cum_norms[n_levels] = 0.0f;

                float* simi = distances + i * k;
                idx_t* idxi = labels + i * k;

                init_result(simi, idxi);

                idx_t nscan = 0;

                for (size_t list_no = 0; list_no < (size_t)nprobe; list_no++) {
                    idx_t cluster_id = keys[i * nprobe + list_no];
                    size_t list_size = invlists->list_size(cluster_id);
                    size_t n_batches =
                            (list_size + batch_size - 1) / batch_size;

                    std::unique_ptr<InvertedLists::ScopedIds> sids;
                    const idx_t* ids =
                            std::make_unique<InvertedLists::ScopedIds>(
                                    invlists, cluster_id)
                                    ->get();

                    for (size_t batch_no = 0; batch_no < n_batches;
                         batch_no++) {
                        size_t curr_batch_size = std::min(
                                list_size - batch_no * batch_size, batch_size);
                        size_t b_offset = batch_no * batch_size;

                        std::iota(
                                active_indices.begin(),
                                active_indices.begin() + curr_batch_size,
                                b_offset);
                        std::fill(
                                bitset.begin(),
                                bitset.begin() + curr_batch_size,
                                1);
                        std::fill(
                                bitset.begin() + curr_batch_size,
                                bitset.end(),
                                0);
                        std::fill(
                                compressed_codes.begin(),
                                compressed_codes.end(),
                                0);

                        for (size_t idx = 0; idx < curr_batch_size; idx++) {
                            exact_distances[idx] = init_exact_distances
                                    [init_exact_distances_offsets[cluster_id] +
                                     b_offset + idx];
                        }

                        const uint8_t* codes = column_storage +
                                column_offsets[cluster_id] +
                                b_offset * code_size;
                        float* cums = cum_sums + cum_sum_offsets[cluster_id] +
                                b_offset * (n_levels + 1);

                        total_points += curr_batch_size * n_levels;

                        total_active += scanner->process_batch(
                                pq,
                                compressed_codes.data(),
                                cluster_id,
                                batch_no,
                                coarse_dis[i * nprobe + list_no],
                                curr_batch_size,
                                batch_size,
                                chunk_size,
                                epsilon,
                                n_levels,
                                codes,
                                cums,
                                query_cum_norms.data(),
                                active_indices.data(),
                                bitset.data(),
                                exact_distances.data(),
                                ids,
                                simi,
                                idxi,
                                k,
                                &dis0s_cache[list_no],
                                sim_table_cache.data() +
                                        list_no * sim_table_size);
                    }
                }

                reorder_result(simi, idxi);

                if (InterruptCallback::is_interrupted()) {
                    interrupt = true;
                }
            }
        }
    }

    if (interrupt) {
        if (!exception_string.empty()) {
            FAISS_THROW_FMT(
                    "search interrupted with: %s", exception_string.c_str());
        } else {
            FAISS_THROW_MSG("computation interrupted");
        }
    }

    printf("v0: total_active: %f\n", (float)total_active / total_points);
}

} // namespace faiss
