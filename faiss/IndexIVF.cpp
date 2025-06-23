/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/IndexIVF.h>

#include <omp.h>
#include <cstdint>
#include <memory>
#include <mutex>

#include <algorithm>
#include <cinttypes>
#include <cstdio>
#include <limits>

#include <faiss/utils/hamming.h>
#include <faiss/utils/utils.h>

#include <faiss/IndexFlat.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/CodePacker.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>

namespace faiss {

using ScopedIds = InvertedLists::ScopedIds;
using ScopedCodes = InvertedLists::ScopedCodes;

/*****************************************
 * Level1Quantizer implementation
 ******************************************/

Level1Quantizer::Level1Quantizer(Index* quantizer, size_t nlist)
        : quantizer(quantizer), nlist(nlist) {
    // here we set a low # iterations because this is typically used
    // for large clusterings (nb this is not used for the MultiIndex,
    // for which quantizer_trains_alone = true)
    cp.niter = 10;
}

Level1Quantizer::Level1Quantizer() = default;

Level1Quantizer::~Level1Quantizer() {
    if (own_fields) {
        delete quantizer;
    }
}

void Level1Quantizer::train_q1(
        size_t n,
        const float* x,
        bool verbose,
        MetricType metric_type) {
    size_t d = quantizer->d;
    if (quantizer->is_trained && (quantizer->ntotal == nlist)) {
        if (verbose)
            printf("IVF quantizer does not need training.\n");
    } else if (quantizer_trains_alone == 1) {
        if (verbose)
            printf("IVF quantizer trains alone...\n");
        quantizer->verbose = verbose;
        quantizer->train(n, x);
        FAISS_THROW_IF_NOT_MSG(
                quantizer->ntotal == nlist,
                "nlist not consistent with quantizer size");
    } else if (quantizer_trains_alone == 0) {
        if (verbose)
            printf("Training level-1 quantizer on %zd vectors in %zdD\n", n, d);

        Clustering clus(d, nlist, cp);
        quantizer->reset();
        if (clustering_index) {
            clus.train(n, x, *clustering_index);
            quantizer->add(nlist, clus.centroids.data());
        } else {
            clus.train(n, x, *quantizer);
        }
        quantizer->is_trained = true;
    } else if (quantizer_trains_alone == 2) {
        if (verbose) {
            printf("Training L2 quantizer on %zd vectors in %zdD%s\n",
                   n,
                   d,
                   clustering_index ? "(user provided index)" : "");
        }
        // also accept spherical centroids because in that case
        // L2 and IP are equivalent
        FAISS_THROW_IF_NOT(
                metric_type == METRIC_L2 ||
                (metric_type == METRIC_INNER_PRODUCT && cp.spherical));

        Clustering clus(d, nlist, cp);
        if (!clustering_index) {
            IndexFlatL2 assigner(d);
            clus.train(n, x, assigner);
        } else {
            clus.train(n, x, *clustering_index);
        }
        if (verbose) {
            printf("Adding centroids to quantizer\n");
        }
        if (!quantizer->is_trained) {
            if (verbose) {
                printf("But training it first on centroids table...\n");
            }
            quantizer->train(nlist, clus.centroids.data());
        }
        quantizer->add(nlist, clus.centroids.data());
    }
}

size_t Level1Quantizer::coarse_code_size() const {
    size_t nl = nlist - 1;
    size_t nbyte = 0;
    while (nl > 0) {
        nbyte++;
        nl >>= 8;
    }
    return nbyte;
}

void Level1Quantizer::encode_listno(idx_t list_no, uint8_t* code) const {
    // little endian
    size_t nl = nlist - 1;
    while (nl > 0) {
        *code++ = list_no & 0xff;
        list_no >>= 8;
        nl >>= 8;
    }
}

idx_t Level1Quantizer::decode_listno(const uint8_t* code) const {
    size_t nl = nlist - 1;
    int64_t list_no = 0;
    int nbit = 0;
    while (nl > 0) {
        list_no |= int64_t(*code++) << nbit;
        nbit += 8;
        nl >>= 8;
    }
    FAISS_THROW_IF_NOT(list_no >= 0 && list_no < nlist);
    return list_no;
}

/*****************************************
 * IndexIVF implementation
 ******************************************/

IndexIVF::IndexIVF(
        Index* quantizer,
        size_t d,
        size_t nlist,
        size_t code_size,
        MetricType metric,
        bool own_invlists)
        : Index(d, metric),
          IndexIVFInterface(quantizer, nlist),
          invlists(
                  own_invlists ? new ArrayInvertedLists(nlist, code_size)
                               : nullptr),
          own_invlists(own_invlists),
          code_size(code_size) {
    FAISS_THROW_IF_NOT(d == quantizer->d);
    is_trained = quantizer->is_trained && (quantizer->ntotal == nlist);
    // Spherical by default if the metric is inner_product
    if (metric_type == METRIC_INNER_PRODUCT) {
        cp.spherical = true;
    }
}

IndexIVF::IndexIVF() = default;

void IndexIVF::add(idx_t n, const float* x) {
    add_with_ids(n, x, nullptr);
}

void IndexIVF::add_with_ids(idx_t n, const float* x, const idx_t* xids) {
    std::unique_ptr<idx_t[]> coarse_idx(new idx_t[n]);
    quantizer->assign(n, x, coarse_idx.get());
    add_core(n, x, xids, coarse_idx.get());
}

void IndexIVF::add_sa_codes(idx_t n, const uint8_t* codes, const idx_t* xids) {
    size_t coarse_size = coarse_code_size();
    DirectMapAdd dm_adder(direct_map, n, xids);

    for (idx_t i = 0; i < n; i++) {
        const uint8_t* code = codes + (code_size + coarse_size) * i;
        idx_t list_no = decode_listno(code);
        idx_t id = xids ? xids[i] : ntotal + i;
        size_t ofs = invlists->add_entry(list_no, id, code + coarse_size);
        dm_adder.add(i, list_no, ofs);
    }
    ntotal += n;
}

void IndexIVF::add_core(
        idx_t n,
        const float* x,
        const idx_t* xids,
        const idx_t* coarse_idx,
        void* inverted_list_context) {
    // do some blocking to avoid excessive allocs
    idx_t bs = 65536;
    if (n > bs) {
        for (idx_t i0 = 0; i0 < n; i0 += bs) {
            idx_t i1 = std::min(n, i0 + bs);
            if (verbose) {
                printf("   IndexIVF::add_with_ids %" PRId64 ":%" PRId64 "\n",
                       i0,
                       i1);
            }
            add_core(
                    i1 - i0,
                    x + i0 * d,
                    xids ? xids + i0 : nullptr,
                    coarse_idx + i0,
                    inverted_list_context);
        }
        return;
    }
    FAISS_THROW_IF_NOT(coarse_idx);
    FAISS_THROW_IF_NOT(is_trained);
    direct_map.check_can_add(xids);

    size_t nadd = 0, nminus1 = 0;

    for (size_t i = 0; i < n; i++) {
        if (coarse_idx[i] < 0)
            nminus1++;
    }

    std::unique_ptr<uint8_t[]> flat_codes(new uint8_t[n * code_size]);
    encode_vectors(n, x, coarse_idx, flat_codes.get());

    DirectMapAdd dm_adder(direct_map, n, xids);

#pragma omp parallel reduction(+ : nadd)
    {
        int nt = omp_get_num_threads();
        int rank = omp_get_thread_num();

        // each thread takes care of a subset of lists
        for (size_t i = 0; i < n; i++) {
            idx_t list_no = coarse_idx[i];
            if (list_no >= 0 && list_no % nt == rank) {
                idx_t id = xids ? xids[i] : ntotal + i;
                size_t ofs = invlists->add_entry(
                        list_no,
                        id,
                        flat_codes.get() + i * code_size,
                        inverted_list_context);

                dm_adder.add(i, list_no, ofs);

                nadd++;
            } else if (rank == 0 && list_no == -1) {
                dm_adder.add(i, -1, 0);
            }
        }
    }

    if (verbose) {
        printf("    added %zd / %" PRId64 " vectors (%zd -1s)\n",
               nadd,
               n,
               nminus1);
    }

    ntotal += n;
}

void IndexIVF::make_direct_map(bool b) {
    if (b) {
        direct_map.set_type(DirectMap::Array, invlists, ntotal);
    } else {
        direct_map.set_type(DirectMap::NoMap, invlists, ntotal);
    }
}

void IndexIVF::set_direct_map_type(DirectMap::Type type) {
    direct_map.set_type(type, invlists, ntotal);
}

/** It is a sad fact of software that a conceptually simple function like this
 * becomes very complex when you factor in several ways of parallelizing +
 * interrupt/error handling + collecting stats + min/max collection. The
 * codepath that is used 95% of time is the one for parallel_mode = 0 */
void IndexIVF::search(
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

    // search function for a subset of queries
    auto sub_search_func = [this, k, nprobe, params](
                                   idx_t n,
                                   const float* x,
                                   float* distances,
                                   idx_t* labels,
                                   IndexIVFStats* ivf_stats) {
        std::unique_ptr<idx_t[]> idx(new idx_t[n * nprobe]);
        std::unique_ptr<float[]> coarse_dis(new float[n * nprobe]);

        double t0 = getmillisecs();
        quantizer->search(
                n,
                x,
                nprobe,
                coarse_dis.get(),
                idx.get(),
                params ? params->quantizer_params : nullptr);

        double t1 = getmillisecs();
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
        double t2 = getmillisecs();
        ivf_stats->quantization_time += t1 - t0;
        ivf_stats->search_time += t2 - t0;
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
            FAISS_THROW_MSG(exception_string.c_str());
        }

        // collect stats
        for (idx_t slice = 0; slice < nt; slice++) {
            indexIVF_stats.add(stats[slice]);
        }
    } else {
        // handle parallelization at level below (or don't run in parallel at
        // all)
        sub_search_func(n, x, distances, labels, &indexIVF_stats);
    }
}

void IndexIVF::search_preassigned(
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
            sel = nullptr; // use special IDSelectorRange processing
        } else {
            selr = nullptr; // use generic processing
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

#pragma omp parallel if (do_parallel) reduction(+ : nlistv, ndis, nheap)
    {
        std::unique_ptr<InvertedListScanner> scanner(
                get_InvertedListScanner(store_pairs, sel, params));

        /*****************************************************
         * Depending on parallel_mode, there are two possible ways
         * to organize the search. Here we define local functions
         * that are in common between the two
         ******************************************************/

        // initialize + reorder a result heap

        auto init_result = [&](float* simi, idx_t* idxi) {
            if (!do_heap_init)
                return;
            if (metric_type == METRIC_INNER_PRODUCT) {
                heap_heapify<HeapForIP>(k, simi, idxi);
            } else {
                heap_heapify<HeapForL2>(k, simi, idxi);
            }
        };

        auto add_local_results = [&](const float* local_dis,
                                     const idx_t* local_idx,
                                     float* simi,
                                     idx_t* idxi) {
            if (metric_type == METRIC_INNER_PRODUCT) {
                heap_addn<HeapForIP>(k, simi, idxi, local_dis, local_idx, k);
            } else {
                heap_addn<HeapForL2>(k, simi, idxi, local_dis, local_idx, k);
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

        // single list scan using the current scanner (with query
        // set porperly) and storing results in simi and idxi
        auto scan_one_list = [&](idx_t key,
                                 float coarse_dis_i,
                                 float* simi,
                                 idx_t* idxi,
                                 idx_t list_size_max) {
            if (key < 0) {
                // not enough centroids for multiprobe
                return (size_t)0;
            }
            FAISS_THROW_IF_NOT_FMT(
                    key < (idx_t)nlist,
                    "Invalid key=%" PRId64 " nlist=%zd\n",
                    key,
                    nlist);

            // don't waste time on empty lists
            if (invlists->is_empty(key, inverted_list_context)) {
                return (size_t)0;
            }

            scanner->set_list(key, coarse_dis_i);

            nlistv++;

            try {
                if (invlists->use_iterator) {
                    size_t list_size = 0;

                    std::unique_ptr<InvertedListsIterator> it(
                            invlists->get_iterator(key, inverted_list_context));

                    nheap += scanner->iterate_codes(
                            it.get(), simi, idxi, k, list_size);

                    return list_size;
                } else {
                    size_t list_size = invlists->list_size(key);
                    if (list_size > list_size_max) {
                        list_size = list_size_max;
                    }

                    InvertedLists::ScopedCodes scodes(invlists, key);
                    const uint8_t* codes = scodes.get();

                    std::unique_ptr<InvertedLists::ScopedIds> sids;
                    const idx_t* ids = nullptr;

                    if (!store_pairs) {
                        sids = std::make_unique<InvertedLists::ScopedIds>(
                                invlists, key);
                        ids = sids->get();
                    }

                    if (selr) { // IDSelectorRange
                        // restrict search to a section of the inverted list
                        size_t jmin, jmax;
                        selr->find_sorted_ids_bounds(
                                list_size, ids, &jmin, &jmax);
                        list_size = jmax - jmin;
                        if (list_size == 0) {
                            return (size_t)0;
                        }
                        codes += jmin * code_size;
                        ids += jmin;
                    }

                    nheap += scanner->scan_codes(
                            list_size, codes, ids, simi, idxi, k);

                    return list_size;
                }
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lock(exception_mutex);
                exception_string =
                        demangle_cpp_symbol(typeid(e).name()) + "  " + e.what();
                interrupt = true;
                return size_t(0);
            }
        };

        /****************************************************
         * Actual loops, depending on parallel_mode
         ****************************************************/

        if (pmode == 0 || pmode == 3) {
#pragma omp for
            for (idx_t i = 0; i < n; i++) {
                if (interrupt) {
                    continue;
                }

                // loop over queries
                scanner->set_query(x + i * d);
                float* simi = distances + i * k;
                idx_t* idxi = labels + i * k;

                init_result(simi, idxi);

                idx_t nscan = 0;

                // loop over probes
                for (size_t ik = 0; ik < nprobe; ik++) {
                    nscan += scan_one_list(
                            keys[i * nprobe + ik],
                            coarse_dis[i * nprobe + ik],
                            simi,
                            idxi,
                            max_codes - nscan);
                    if (nscan >= max_codes) {
                        break;
                    }
                }

                ndis += nscan;
                reorder_result(simi, idxi);

                if (InterruptCallback::is_interrupted()) {
                    interrupt = true;
                }

            } // parallel for
        } else if (pmode == 1) {
            std::vector<idx_t> local_idx(k);
            std::vector<float> local_dis(k);

            for (size_t i = 0; i < n; i++) {
                scanner->set_query(x + i * d);
                init_result(local_dis.data(), local_idx.data());

#pragma omp for schedule(dynamic)
                for (idx_t ik = 0; ik < nprobe; ik++) {
                    ndis += scan_one_list(
                            keys[i * nprobe + ik],
                            coarse_dis[i * nprobe + ik],
                            local_dis.data(),
                            local_idx.data(),
                            unlimited_list_size);

                    // can't do the test on max_codes
                }
                // merge thread-local results

                float* simi = distances + i * k;
                idx_t* idxi = labels + i * k;
#pragma omp single
                init_result(simi, idxi);

#pragma omp barrier
#pragma omp critical
                {
                    add_local_results(
                            local_dis.data(), local_idx.data(), simi, idxi);
                }
#pragma omp barrier
#pragma omp single
                reorder_result(simi, idxi);
            }
        } else if (pmode == 2) {
            std::vector<idx_t> local_idx(k);
            std::vector<float> local_dis(k);

#pragma omp single
            for (int64_t i = 0; i < n; i++) {
                init_result(distances + i * k, labels + i * k);
            }

#pragma omp for schedule(dynamic)
            for (int64_t ij = 0; ij < n * nprobe; ij++) {
                size_t i = ij / nprobe;

                scanner->set_query(x + i * d);
                init_result(local_dis.data(), local_idx.data());
                ndis += scan_one_list(
                        keys[ij],
                        coarse_dis[ij],
                        local_dis.data(),
                        local_idx.data(),
                        unlimited_list_size);
#pragma omp critical
                {
                    add_local_results(
                            local_dis.data(),
                            local_idx.data(),
                            distances + i * k,
                            labels + i * k);
                }
            }
#pragma omp single
            for (int64_t i = 0; i < n; i++) {
                reorder_result(distances + i * k, labels + i * k);
            }
        } else {
            FAISS_THROW_FMT("parallel_mode %d not supported\n", pmode);
        }
    } // parallel section

    if (interrupt) {
        if (!exception_string.empty()) {
            FAISS_THROW_FMT(
                    "search interrupted with: %s", exception_string.c_str());
        } else {
            FAISS_THROW_MSG("computation interrupted");
        }
    }

    if (ivf_stats == nullptr) {
        ivf_stats = &indexIVF_stats;
    }
    ivf_stats->nq += n;
    ivf_stats->nlist += nlistv;
    ivf_stats->ndis += ndis;
    ivf_stats->nheap_updates += nheap;
}

void IndexIVF::range_search(
        idx_t nx,
        const float* x,
        float radius,
        RangeSearchResult* result,
        const SearchParameters* params_in) const {
    const IVFSearchParameters* params = nullptr;
    const SearchParameters* quantizer_params = nullptr;
    if (params_in) {
        params = dynamic_cast<const IVFSearchParameters*>(params_in);
        FAISS_THROW_IF_NOT_MSG(params, "IndexIVF params have incorrect type");
        quantizer_params = params->quantizer_params;
    }
    const size_t nprobe =
            std::min(nlist, params ? params->nprobe : this->nprobe);
    std::unique_ptr<idx_t[]> keys(new idx_t[nx * nprobe]);
    std::unique_ptr<float[]> coarse_dis(new float[nx * nprobe]);

    double t0 = getmillisecs();
    quantizer->search(
            nx, x, nprobe, coarse_dis.get(), keys.get(), quantizer_params);
    indexIVF_stats.quantization_time += getmillisecs() - t0;

    t0 = getmillisecs();
    invlists->prefetch_lists(keys.get(), nx * nprobe);

    range_search_preassigned(
            nx,
            x,
            radius,
            keys.get(),
            coarse_dis.get(),
            result,
            false,
            params,
            &indexIVF_stats);

    indexIVF_stats.search_time += getmillisecs() - t0;
}

void IndexIVF::range_search_preassigned(
        idx_t nx,
        const float* x,
        float radius,
        const idx_t* keys,
        const float* coarse_dis,
        RangeSearchResult* result,
        bool store_pairs,
        const IVFSearchParameters* params,
        IndexIVFStats* stats) const {
    idx_t nprobe = params ? params->nprobe : this->nprobe;
    nprobe = std::min((idx_t)nlist, nprobe);
    FAISS_THROW_IF_NOT(nprobe > 0);

    idx_t max_codes = params ? params->max_codes : this->max_codes;
    IDSelector* sel = params ? params->sel : nullptr;

    FAISS_THROW_IF_NOT_MSG(
            !invlists->use_iterator || (max_codes == 0 && store_pairs == false),
            "iterable inverted lists don't support max_codes and store_pairs");

    size_t nlistv = 0, ndis = 0;

    bool interrupt = false;
    std::mutex exception_mutex;
    std::string exception_string;

    std::vector<RangeSearchPartialResult*> all_pres(omp_get_max_threads());

    int pmode = this->parallel_mode & ~PARALLEL_MODE_NO_HEAP_INIT;
    // don't start parallel section if single query
    [[maybe_unused]] bool do_parallel = omp_get_max_threads() >= 2 &&
            (pmode == 3           ? false
                     : pmode == 0 ? nx > 1
                     : pmode == 1 ? nprobe > 1
                                  : nprobe * nx > 1);

    void* inverted_list_context =
            params ? params->inverted_list_context : nullptr;

#pragma omp parallel if (do_parallel) reduction(+ : nlistv, ndis)
    {
        RangeSearchPartialResult pres(result);
        std::unique_ptr<InvertedListScanner> scanner(
                get_InvertedListScanner(store_pairs, sel, params));
        FAISS_THROW_IF_NOT(scanner.get());
        all_pres[omp_get_thread_num()] = &pres;

        // prepare the list scanning function

        auto scan_list_func = [&](size_t i, size_t ik, RangeQueryResult& qres) {
            idx_t key = keys[i * nprobe + ik]; /* select the list  */
            if (key < 0)
                return;
            FAISS_THROW_IF_NOT_FMT(
                    key < (idx_t)nlist,
                    "Invalid key=%" PRId64 " at ik=%zd nlist=%zd\n",
                    key,
                    ik,
                    nlist);

            if (invlists->is_empty(key, inverted_list_context)) {
                return;
            }

            try {
                size_t list_size = 0;
                scanner->set_list(key, coarse_dis[i * nprobe + ik]);
                if (invlists->use_iterator) {
                    std::unique_ptr<InvertedListsIterator> it(
                            invlists->get_iterator(key, inverted_list_context));

                    scanner->iterate_codes_range(
                            it.get(), radius, qres, list_size);
                } else {
                    InvertedLists::ScopedCodes scodes(invlists, key);
                    InvertedLists::ScopedIds ids(invlists, key);
                    list_size = invlists->list_size(key);

                    scanner->scan_codes_range(
                            list_size, scodes.get(), ids.get(), radius, qres);
                }
                nlistv++;
                ndis += list_size;
            } catch (const std::exception& e) {
                std::lock_guard<std::mutex> lock(exception_mutex);
                exception_string =
                        demangle_cpp_symbol(typeid(e).name()) + "  " + e.what();
                interrupt = true;
            }
        };

        if (parallel_mode == 0) {
#pragma omp for
            for (idx_t i = 0; i < nx; i++) {
                scanner->set_query(x + i * d);

                RangeQueryResult& qres = pres.new_result(i);

                for (size_t ik = 0; ik < nprobe; ik++) {
                    scan_list_func(i, ik, qres);
                }
            }

        } else if (parallel_mode == 1) {
            for (size_t i = 0; i < nx; i++) {
                scanner->set_query(x + i * d);

                RangeQueryResult& qres = pres.new_result(i);

#pragma omp for schedule(dynamic)
                for (int64_t ik = 0; ik < nprobe; ik++) {
                    scan_list_func(i, ik, qres);
                }
            }
        } else if (parallel_mode == 2) {
            RangeQueryResult* qres = nullptr;

#pragma omp for schedule(dynamic)
            for (idx_t iik = 0; iik < nx * (idx_t)nprobe; iik++) {
                idx_t i = iik / (idx_t)nprobe;
                idx_t ik = iik % (idx_t)nprobe;
                if (qres == nullptr || qres->qno != i) {
                    qres = &pres.new_result(i);
                    scanner->set_query(x + i * d);
                }
                scan_list_func(i, ik, *qres);
            }
        } else {
            FAISS_THROW_FMT("parallel_mode %d not supported\n", parallel_mode);
        }
        if (parallel_mode == 0) {
            pres.finalize();
        } else {
#pragma omp barrier
#pragma omp single
            RangeSearchPartialResult::merge(all_pres, false);
#pragma omp barrier
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

    if (stats == nullptr) {
        stats = &indexIVF_stats;
    }
    stats->nq += nx;
    stats->nlist += nlistv;
    stats->ndis += ndis;
}

InvertedListScanner* IndexIVF::get_InvertedListScanner(
        bool /*store_pairs*/,
        const IDSelector* /* sel */,
        const IVFSearchParameters* /* params */) const {
    FAISS_THROW_MSG("get_InvertedListScanner not implemented");
}

void IndexIVF::reconstruct(idx_t key, float* recons) const {
    idx_t lo = direct_map.get(key);
    reconstruct_from_offset(lo_listno(lo), lo_offset(lo), recons);
}

void IndexIVF::reconstruct_n(idx_t i0, idx_t ni, float* recons) const {
    FAISS_THROW_IF_NOT(ni == 0 || (i0 >= 0 && i0 + ni <= ntotal));

    for (idx_t list_no = 0; list_no < nlist; list_no++) {
        size_t list_size = invlists->list_size(list_no);
        ScopedIds idlist(invlists, list_no);

        for (idx_t offset = 0; offset < list_size; offset++) {
            idx_t id = idlist[offset];
            if (!(id >= i0 && id < i0 + ni)) {
                continue;
            }

            float* reconstructed = recons + (id - i0) * d;
            reconstruct_from_offset(list_no, offset, reconstructed);
        }
    }
}

bool IndexIVF::check_ids_sorted() const {
    size_t nflip = 0;

    for (size_t i = 0; i < nlist; i++) {
        size_t list_size = invlists->list_size(i);
        InvertedLists::ScopedIds ids(invlists, i);
        for (size_t j = 0; j + 1 < list_size; j++) {
            if (ids[j + 1] < ids[j]) {
                nflip++;
            }
        }
    }
    return nflip == 0;
}

/* standalone codec interface */
size_t IndexIVF::sa_code_size() const {
    size_t coarse_size = coarse_code_size();
    return code_size + coarse_size;
}

void IndexIVF::sa_encode(idx_t n, const float* x, uint8_t* bytes) const {
    FAISS_THROW_IF_NOT(is_trained);
    std::unique_ptr<int64_t[]> idx(new int64_t[n]);
    quantizer->assign(n, x, idx.get());
    encode_vectors(n, x, idx.get(), bytes, true);
}

void IndexIVF::search_and_reconstruct(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        float* recons,
        const SearchParameters* params_in) const {
    const IVFSearchParameters* params = nullptr;
    if (params_in) {
        params = dynamic_cast<const IVFSearchParameters*>(params_in);
        FAISS_THROW_IF_NOT_MSG(params, "IndexIVF params have incorrect type");
    }
    const size_t nprobe =
            std::min(nlist, params ? params->nprobe : this->nprobe);
    FAISS_THROW_IF_NOT(nprobe > 0);

    std::unique_ptr<idx_t[]> idx(new idx_t[n * nprobe]);
    std::unique_ptr<float[]> coarse_dis(new float[n * nprobe]);

    quantizer->search(n, x, nprobe, coarse_dis.get(), idx.get());

    invlists->prefetch_lists(idx.get(), n * nprobe);

    // search_preassigned() with `store_pairs` enabled to obtain the list_no
    // and offset into `codes` for reconstruction
    search_preassigned(
            n,
            x,
            k,
            idx.get(),
            coarse_dis.get(),
            distances,
            labels,
            true /* store_pairs */,
            params);
#pragma omp parallel for if (n * k > 1000)
    for (idx_t ij = 0; ij < n * k; ij++) {
        idx_t key = labels[ij];
        float* reconstructed = recons + ij * d;
        if (key < 0) {
            // Fill with NaNs
            memset(reconstructed, -1, sizeof(*reconstructed) * d);
        } else {
            int list_no = lo_listno(key);
            int offset = lo_offset(key);

            // Update label to the actual id
            labels[ij] = invlists->get_single_id(list_no, offset);

            reconstruct_from_offset(list_no, offset, reconstructed);
        }
    }
}

void IndexIVF::search_and_return_codes(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        uint8_t* codes,
        bool include_listno,
        const SearchParameters* params_in) const {
    const IVFSearchParameters* params = nullptr;
    if (params_in) {
        params = dynamic_cast<const IVFSearchParameters*>(params_in);
        FAISS_THROW_IF_NOT_MSG(params, "IndexIVF params have incorrect type");
    }
    const size_t nprobe =
            std::min(nlist, params ? params->nprobe : this->nprobe);
    FAISS_THROW_IF_NOT(nprobe > 0);

    std::unique_ptr<idx_t[]> idx(new idx_t[n * nprobe]);
    std::unique_ptr<float[]> coarse_dis(new float[n * nprobe]);

    quantizer->search(n, x, nprobe, coarse_dis.get(), idx.get());

    invlists->prefetch_lists(idx.get(), n * nprobe);

    // search_preassigned() with `store_pairs` enabled to obtain the list_no
    // and offset into `codes` for reconstruction
    search_preassigned(
            n,
            x,
            k,
            idx.get(),
            coarse_dis.get(),
            distances,
            labels,
            true /* store_pairs */,
            params);

    size_t code_size_1 = code_size;
    if (include_listno) {
        code_size_1 += coarse_code_size();
    }

#pragma omp parallel for if (n * k > 1000)
    for (idx_t ij = 0; ij < n * k; ij++) {
        idx_t key = labels[ij];
        uint8_t* code1 = codes + ij * code_size_1;

        if (key < 0) {
            // Fill with 0xff
            memset(code1, -1, code_size_1);
        } else {
            int list_no = lo_listno(key);
            int offset = lo_offset(key);
            const uint8_t* cc = invlists->get_single_code(list_no, offset);

            labels[ij] = invlists->get_single_id(list_no, offset);

            if (include_listno) {
                encode_listno(list_no, code1);
                code1 += code_size_1 - code_size;
            }
            memcpy(code1, cc, code_size);
        }
    }
}

void IndexIVF::reconstruct_from_offset(
        int64_t /*list_no*/,
        int64_t /*offset*/,
        float* /*recons*/) const {
    FAISS_THROW_MSG("reconstruct_from_offset not implemented");
}

void IndexIVF::reset() {
    direct_map.clear();
    invlists->reset();
    ntotal = 0;
}

size_t IndexIVF::remove_ids(const IDSelector& sel) {
    size_t nremove = direct_map.remove_ids(sel, invlists);
    ntotal -= nremove;
    return nremove;
}

void IndexIVF::update_vectors(int n, const idx_t* new_ids, const float* x) {
    if (direct_map.type == DirectMap::Hashtable) {
        // just remove then add
        IDSelectorArray sel(n, new_ids);
        size_t nremove = remove_ids(sel);
        FAISS_THROW_IF_NOT_MSG(
                nremove == n, "did not find all entries to remove");
        add_with_ids(n, x, new_ids);
        return;
    }

    FAISS_THROW_IF_NOT(direct_map.type == DirectMap::Array);
    // here it is more tricky because we don't want to introduce holes
    // in continuous range of ids

    FAISS_THROW_IF_NOT(is_trained);
    std::vector<idx_t> assign(n);
    quantizer->assign(n, x, assign.data());

    std::vector<uint8_t> flat_codes(n * code_size);
    encode_vectors(n, x, assign.data(), flat_codes.data());

    direct_map.update_codes(
            invlists, n, new_ids, assign.data(), flat_codes.data());
}

void IndexIVF::train(idx_t n, const float* x) {
    if (verbose) {
        printf("Training level-1 quantizer\n");
    }

    train_q1(n, x, verbose, metric_type);

    if (verbose) {
        printf("Training IVF residual\n");
    }

    // optional subsampling
    idx_t max_nt = train_encoder_num_vectors();
    if (max_nt <= 0) {
        max_nt = (size_t)1 << 35;
    }

    TransformedVectors tv(
            x, fvecs_maybe_subsample(d, (size_t*)&n, max_nt, x, verbose));

    if (by_residual) {
        std::vector<idx_t> assign(n);
        quantizer->assign(n, tv.x, assign.data());

        std::vector<float> residuals(n * d);
        quantizer->compute_residual_n(n, tv.x, residuals.data(), assign.data());

        train_encoder(n, residuals.data(), assign.data());
    } else {
        train_encoder(n, tv.x, nullptr);
    }

    is_trained = true;
}

idx_t IndexIVF::train_encoder_num_vectors() const {
    return 0;
}

void IndexIVF::train_encoder(
        idx_t /*n*/,
        const float* /*x*/,
        const idx_t* assign) {
    // does nothing by default
    if (verbose) {
        printf("IndexIVF: no residual training\n");
    }
}

bool check_compatible_for_merge_expensive_check = true;

void IndexIVF::check_compatible_for_merge(const Index& otherIndex) const {
    // minimal sanity checks
    const IndexIVF* other = dynamic_cast<const IndexIVF*>(&otherIndex);
    FAISS_THROW_IF_NOT(other);
    FAISS_THROW_IF_NOT(other->d == d);
    FAISS_THROW_IF_NOT(other->nlist == nlist);
    FAISS_THROW_IF_NOT(quantizer->ntotal == other->quantizer->ntotal);
    FAISS_THROW_IF_NOT(other->code_size == code_size);
    FAISS_THROW_IF_NOT_MSG(
            typeid(*this) == typeid(*other),
            "can only merge indexes of the same type");
    FAISS_THROW_IF_NOT_MSG(
            this->direct_map.no() && other->direct_map.no(),
            "merge direct_map not implemented");

    if (check_compatible_for_merge_expensive_check) {
        std::vector<float> v(d), v2(d);
        for (size_t i = 0; i < nlist; i++) {
            quantizer->reconstruct(i, v.data());
            other->quantizer->reconstruct(i, v2.data());
            FAISS_THROW_IF_NOT_MSG(
                    v == v2, "coarse quantizers should be the same");
        }
    }
}

void IndexIVF::merge_from(Index& otherIndex, idx_t add_id) {
    check_compatible_for_merge(otherIndex);
    IndexIVF* other = static_cast<IndexIVF*>(&otherIndex);
    invlists->merge_from(other->invlists, add_id);

    ntotal += other->ntotal;
    other->ntotal = 0;
}

CodePacker* IndexIVF::get_CodePacker() const {
    return new CodePackerFlat(code_size);
}

void IndexIVF::replace_invlists(InvertedLists* il, bool own) {
    if (own_invlists) {
        delete invlists;
        invlists = nullptr;
    }
    // FAISS_THROW_IF_NOT (ntotal == 0);
    if (il) {
        FAISS_THROW_IF_NOT(il->nlist == nlist);
        FAISS_THROW_IF_NOT(
                il->code_size == code_size ||
                il->code_size == InvertedLists::INVALID_CODE_SIZE);
    }
    invlists = il;
    own_invlists = own;
}

void IndexIVF::copy_subset_to(
        IndexIVF& other,
        InvertedLists::subset_type_t subset_type,
        idx_t a1,
        idx_t a2) const {
    other.ntotal +=
            invlists->copy_subset_to(*other.invlists, subset_type, a1, a2);
}

IndexIVF::~IndexIVF() {
    if (own_invlists) {
        delete invlists;
    }
}

/*************************************************************************
 * IndexIVFStats
 *************************************************************************/

void IndexIVFStats::reset() {
    memset((void*)this, 0, sizeof(*this));
}

void IndexIVFStats::add(const IndexIVFStats& other) {
    nq += other.nq;
    nlist += other.nlist;
    ndis += other.ndis;
    nheap_updates += other.nheap_updates;
    quantization_time += other.quantization_time;
    search_time += other.search_time;
}

IndexIVFStats indexIVF_stats;

/*************************************************************************
 * InvertedListScanner
 *************************************************************************/

size_t InvertedListScanner::scan_codes(
        size_t list_size,
        const uint8_t* codes,
        const idx_t* ids,
        float* simi,
        idx_t* idxi,
        size_t k) const {
    size_t nup = 0;

    if (!keep_max) {
        for (size_t j = 0; j < list_size; j++) {
            if (sel != nullptr) {
                int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                if (!sel->is_member(id)) {
                    codes += code_size;
                    continue;
                }
            }

            float dis = distance_to_code(codes);
            if (dis < simi[0]) {
                int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                maxheap_replace_top(k, simi, idxi, dis, id);
                nup++;
            }
            codes += code_size;
        }
    } else {
        for (size_t j = 0; j < list_size; j++) {
            if (sel != nullptr) {
                int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                if (!sel->is_member(id)) {
                    codes += code_size;
                    continue;
                }
            }

            float dis = distance_to_code(codes);
            if (dis > simi[0]) {
                int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                minheap_replace_top(k, simi, idxi, dis, id);
                nup++;
            }
            codes += code_size;
        }
    }
    return nup;
}

size_t InvertedListScanner::iterate_codes(
        InvertedListsIterator* it,
        float* simi,
        idx_t* idxi,
        size_t k,
        size_t& list_size) const {
    size_t nup = 0;
    list_size = 0;

    if (!keep_max) {
        for (; it->is_available(); it->next()) {
            auto id_and_codes = it->get_id_and_codes();
            float dis = distance_to_code(id_and_codes.second);
            if (dis < simi[0]) {
                maxheap_replace_top(k, simi, idxi, dis, id_and_codes.first);
                nup++;
            }
            list_size++;
        }
    } else {
        for (; it->is_available(); it->next()) {
            auto id_and_codes = it->get_id_and_codes();
            float dis = distance_to_code(id_and_codes.second);
            if (dis > simi[0]) {
                minheap_replace_top(k, simi, idxi, dis, id_and_codes.first);
                nup++;
            }
            list_size++;
        }
    }
    return nup;
}

void InvertedListScanner::scan_codes_range(
        size_t list_size,
        const uint8_t* codes,
        const idx_t* ids,
        float radius,
        RangeQueryResult& res) const {
    for (size_t j = 0; j < list_size; j++) {
        float dis = distance_to_code(codes);
        bool keep = !keep_max
                ? dis < radius
                : dis > radius; // TODO templatize to remove this test
        if (keep) {
            int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
            res.add(dis, id);
        }
        codes += code_size;
    }
}

void InvertedListScanner::iterate_codes_range(
        InvertedListsIterator* it,
        float radius,
        RangeQueryResult& res,
        size_t& list_size) const {
    list_size = 0;
    for (; it->is_available(); it->next()) {
        auto id_and_codes = it->get_id_and_codes();
        float dis = distance_to_code(id_and_codes.second);
        bool keep = !keep_max
                ? dis < radius
                : dis > radius; // TODO templatize to remove this test
        if (keep) {
            res.add(dis, id_and_codes.first);
        }
        list_size++;
    }
}

} // namespace faiss
