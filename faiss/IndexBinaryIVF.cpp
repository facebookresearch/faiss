/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/IndexBinaryIVF.h>

#include <omp.h>
#include <cinttypes>
#include <cstdio>

#include <algorithm>
#include <memory>

#include <faiss/IndexFlat.h>
#include <faiss/IndexLSH.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/hamming.h>
#include <faiss/utils/sorting.h>
#include <faiss/utils/utils.h>

namespace faiss {

IndexBinaryIVF::IndexBinaryIVF(IndexBinary* quantizer, size_t d, size_t nlist)
        : IndexBinary(d),
          invlists(new ArrayInvertedLists(nlist, code_size)),
          quantizer(quantizer),
          nlist(nlist) {
    FAISS_THROW_IF_NOT(d == quantizer->d);
    is_trained = quantizer->is_trained && (quantizer->ntotal == nlist);
    cp.niter = 10;
}

IndexBinaryIVF::IndexBinaryIVF() = default;

void IndexBinaryIVF::add(idx_t n, const uint8_t* x) {
    add_with_ids(n, x, nullptr);
}

void IndexBinaryIVF::add_with_ids(
        idx_t n,
        const uint8_t* x,
        const idx_t* xids) {
    add_core(n, x, xids, nullptr);
}

void IndexBinaryIVF::add_core(
        idx_t n,
        const uint8_t* x,
        const idx_t* xids,
        const idx_t* precomputed_idx) {
    FAISS_THROW_IF_NOT(is_trained);
    assert(invlists);
    direct_map.check_can_add(xids);

    const idx_t* idx;

    std::unique_ptr<idx_t[]> scoped_idx;

    if (precomputed_idx) {
        idx = precomputed_idx;
    } else {
        scoped_idx.reset(new idx_t[n]);
        quantizer->assign(n, x, scoped_idx.get());
        idx = scoped_idx.get();
    }

    idx_t n_add = 0;
    for (size_t i = 0; i < n; i++) {
        idx_t id = xids ? xids[i] : ntotal + i;
        idx_t list_no = idx[i];

        if (list_no < 0) {
            direct_map.add_single_id(id, -1, 0);
        } else {
            const uint8_t* xi = x + i * code_size;
            size_t offset = invlists->add_entry(list_no, id, xi);

            direct_map.add_single_id(id, list_no, offset);
        }

        n_add++;
    }
    if (verbose) {
        printf("IndexBinaryIVF::add_with_ids: added "
               "%" PRId64 " / %" PRId64 " vectors\n",
               n_add,
               n);
    }
    ntotal += n_add;
}

void IndexBinaryIVF::make_direct_map(bool b) {
    if (b) {
        direct_map.set_type(DirectMap::Array, invlists, ntotal);
    } else {
        direct_map.set_type(DirectMap::NoMap, invlists, ntotal);
    }
}

void IndexBinaryIVF::set_direct_map_type(DirectMap::Type type) {
    direct_map.set_type(type, invlists, ntotal);
}

void IndexBinaryIVF::search(
        idx_t n,
        const uint8_t* x,
        idx_t k,
        int32_t* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(
            !params, "search params not supported for this index");
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(nprobe > 0);

    const size_t nprobe_2 = std::min(nlist, this->nprobe);
    std::unique_ptr<idx_t[]> idx(new idx_t[n * nprobe_2]);
    std::unique_ptr<int32_t[]> coarse_dis(new int32_t[n * nprobe_2]);

    double t0 = getmillisecs();
    quantizer->search(n, x, nprobe_2, coarse_dis.get(), idx.get());
    indexIVF_stats.quantization_time += getmillisecs() - t0;

    t0 = getmillisecs();
    invlists->prefetch_lists(idx.get(), n * nprobe_2);

    search_preassigned(
            n, x, k, idx.get(), coarse_dis.get(), distances, labels, false);
    indexIVF_stats.search_time += getmillisecs() - t0;
}

void IndexBinaryIVF::reconstruct(idx_t key, uint8_t* recons) const {
    idx_t lo = direct_map.get(key);
    reconstruct_from_offset(lo_listno(lo), lo_offset(lo), recons);
}

void IndexBinaryIVF::reconstruct_n(idx_t i0, idx_t ni, uint8_t* recons) const {
    FAISS_THROW_IF_NOT(ni == 0 || (i0 >= 0 && i0 + ni <= ntotal));

    for (idx_t list_no = 0; list_no < nlist; list_no++) {
        size_t list_size = invlists->list_size(list_no);
        const idx_t* idlist = invlists->get_ids(list_no);

        for (idx_t offset = 0; offset < list_size; offset++) {
            idx_t id = idlist[offset];
            if (!(id >= i0 && id < i0 + ni)) {
                continue;
            }

            uint8_t* reconstructed = recons + (id - i0) * d;
            reconstruct_from_offset(list_no, offset, reconstructed);
        }
    }
}

void IndexBinaryIVF::search_and_reconstruct(
        idx_t n,
        const uint8_t* __restrict x,
        idx_t k,
        int32_t* __restrict distances,
        idx_t* __restrict labels,
        uint8_t* __restrict recons,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(
            !params, "search params not supported for this index");
    const size_t nprobe_2 = std::min(nlist, this->nprobe);
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(nprobe_2 > 0);

    std::unique_ptr<idx_t[]> idx(new idx_t[n * nprobe_2]);
    std::unique_ptr<int32_t[]> coarse_dis(new int32_t[n * nprobe_2]);

    quantizer->search(n, x, nprobe_2, coarse_dis.get(), idx.get());

    invlists->prefetch_lists(idx.get(), n * nprobe_2);

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
            /* store_pairs */ true);
    for (idx_t i = 0; i < n; ++i) {
        for (idx_t j = 0; j < k; ++j) {
            idx_t ij = i * k + j;
            idx_t key = labels[ij];
            uint8_t* reconstructed = recons + ij * d;
            if (key < 0) {
                // Fill with NaNs
                memset(reconstructed, -1, sizeof(*reconstructed) * d);
            } else {
                int list_no = key >> 32;
                int offset = key & 0xffffffff;

                // Update label to the actual id
                labels[ij] = invlists->get_single_id(list_no, offset);

                reconstruct_from_offset(list_no, offset, reconstructed);
            }
        }
    }
}

void IndexBinaryIVF::reconstruct_from_offset(
        idx_t list_no,
        idx_t offset,
        uint8_t* recons) const {
    memcpy(recons, invlists->get_single_code(list_no, offset), code_size);
}

void IndexBinaryIVF::reset() {
    direct_map.clear();
    invlists->reset();
    ntotal = 0;
}

size_t IndexBinaryIVF::remove_ids(const IDSelector& sel) {
    size_t nremove = direct_map.remove_ids(sel, invlists);
    ntotal -= nremove;
    return nremove;
}

void IndexBinaryIVF::train(idx_t n, const uint8_t* x) {
    if (verbose) {
        printf("Training quantizer\n");
    }

    if (quantizer->is_trained && (quantizer->ntotal == nlist)) {
        if (verbose) {
            printf("IVF quantizer does not need training.\n");
        }
    } else {
        if (verbose) {
            printf("Training quantizer on %" PRId64 " vectors in %dD\n", n, d);
        }

        Clustering clus(d, nlist, cp);
        quantizer->reset();

        IndexFlatL2 index_tmp(d);

        if (clustering_index && verbose) {
            printf("using clustering_index of dimension %d to do the clustering\n",
                   clustering_index->d);
        }

        // LSH codec that is able to convert the binary vectors to floats.
        IndexLSH codec(d, d, false, false);

        clus.train_encoded(
                n, x, &codec, clustering_index ? *clustering_index : index_tmp);

        // convert clusters to binary
        std::unique_ptr<uint8_t[]> x_b(new uint8_t[clus.k * code_size]);
        real_to_binary(d * clus.k, clus.centroids.data(), x_b.get());

        quantizer->add(clus.k, x_b.get());
        quantizer->is_trained = true;
    }

    is_trained = true;
}

void IndexBinaryIVF::check_compatible_for_merge(
        const IndexBinary& otherIndex) const {
    auto other = dynamic_cast<const IndexBinaryIVF*>(&otherIndex);
    FAISS_THROW_IF_NOT(other);
    FAISS_THROW_IF_NOT(other->d == d);
    FAISS_THROW_IF_NOT(other->nlist == nlist);
    FAISS_THROW_IF_NOT(other->code_size == code_size);
    FAISS_THROW_IF_NOT_MSG(
            direct_map.no() && other->direct_map.no(),
            "direct map copy not implemented");
    FAISS_THROW_IF_NOT_MSG(
            typeid(*this) == typeid(other),
            "can only merge indexes of the same type");
}

void IndexBinaryIVF::merge_from(IndexBinary& otherIndex, idx_t add_id) {
    // minimal sanity checks
    check_compatible_for_merge(otherIndex);
    auto other = static_cast<IndexBinaryIVF*>(&otherIndex);
    invlists->merge_from(other->invlists, add_id);
    ntotal += other->ntotal;
    other->ntotal = 0;
}

void IndexBinaryIVF::replace_invlists(InvertedLists* il, bool own) {
    FAISS_THROW_IF_NOT(il->nlist == nlist && il->code_size == code_size);
    if (own_invlists) {
        delete invlists;
    }
    invlists = il;
    own_invlists = own;
}

namespace {

template <class HammingComputer>
struct IVFBinaryScannerL2 : BinaryInvertedListScanner {
    HammingComputer hc;
    size_t code_size;
    bool store_pairs;

    IVFBinaryScannerL2(size_t code_size, bool store_pairs)
            : code_size(code_size), store_pairs(store_pairs) {}

    void set_query(const uint8_t* query_vector) override {
        hc.set(query_vector, code_size);
    }

    idx_t list_no;
    void set_list(idx_t list_no_2, uint8_t /* coarse_dis */) override {
        this->list_no = list_no_2;
    }

    uint32_t distance_to_code(const uint8_t* code) const override {
        return hc.hamming(code);
    }

    size_t scan_codes(
            size_t n,
            const uint8_t* __restrict codes,
            const idx_t* __restrict ids,
            int32_t* __restrict simi,
            idx_t* __restrict idxi,
            size_t k) const override {
        using C = CMax<int32_t, idx_t>;

        size_t nup = 0;
        for (size_t j = 0; j < n; j++) {
            uint32_t dis = hc.hamming(codes);
            if (dis < simi[0]) {
                idx_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                heap_replace_top<C>(k, simi, idxi, dis, id);
                nup++;
            }
            codes += code_size;
        }
        return nup;
    }

    void scan_codes_range(
            size_t n,
            const uint8_t* __restrict codes,
            const idx_t* __restrict ids,
            int radius,
            RangeQueryResult& result) const override {
        for (size_t j = 0; j < n; j++) {
            uint32_t dis = hc.hamming(codes);
            if (dis < radius) {
                int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                result.add(dis, id);
            }
            codes += code_size;
        }
    }
};

void search_knn_hamming_heap(
        const IndexBinaryIVF* ivf,
        size_t n,
        const uint8_t* __restrict x,
        idx_t k,
        const idx_t* __restrict keys,
        const int32_t* __restrict coarse_dis,
        int32_t* __restrict distances,
        idx_t* __restrict labels,
        bool store_pairs,
        const IVFSearchParameters* params) {
    idx_t nprobe = params ? params->nprobe : ivf->nprobe;
    nprobe = std::min((idx_t)ivf->nlist, nprobe);
    idx_t max_codes = params ? params->max_codes : ivf->max_codes;
    MetricType metric_type = ivf->metric_type;

    // almost verbatim copy from IndexIVF::search_preassigned

    size_t nlistv = 0, ndis = 0, nheap = 0;
    using HeapForIP = CMin<int32_t, idx_t>;
    using HeapForL2 = CMax<int32_t, idx_t>;

#pragma omp parallel if (n > 1) reduction(+ : nlistv, ndis, nheap)
    {
        std::unique_ptr<BinaryInvertedListScanner> scanner(
                ivf->get_InvertedListScanner(store_pairs));

#pragma omp for
        for (idx_t i = 0; i < n; i++) {
            const uint8_t* xi = x + i * ivf->code_size;
            scanner->set_query(xi);

            const idx_t* keysi = keys + i * nprobe;
            int32_t* simi = distances + k * i;
            idx_t* idxi = labels + k * i;

            if (metric_type == METRIC_INNER_PRODUCT) {
                heap_heapify<HeapForIP>(k, simi, idxi);
            } else {
                heap_heapify<HeapForL2>(k, simi, idxi);
            }

            size_t nscan = 0;

            for (size_t ik = 0; ik < nprobe; ik++) {
                idx_t key = keysi[ik]; /* select the list  */
                if (key < 0) {
                    // not enough centroids for multiprobe
                    continue;
                }
                FAISS_THROW_IF_NOT_FMT(
                        key < (idx_t)ivf->nlist,
                        "Invalid key=%" PRId64 " at ik=%zd nlist=%zd\n",
                        key,
                        ik,
                        ivf->nlist);

                scanner->set_list(key, coarse_dis[i * nprobe + ik]);

                nlistv++;

                size_t list_size = ivf->invlists->list_size(key);
                InvertedLists::ScopedCodes scodes(ivf->invlists, key);
                std::unique_ptr<InvertedLists::ScopedIds> sids;
                const idx_t* ids = nullptr;

                if (!store_pairs) {
                    sids = std::make_unique<InvertedLists::ScopedIds>(
                            ivf->invlists, key);
                    ids = sids->get();
                }

                nheap += scanner->scan_codes(
                        list_size, scodes.get(), ids, simi, idxi, k);

                nscan += list_size;
                if (max_codes && nscan >= max_codes)
                    break;
            }

            ndis += nscan;
            if (metric_type == METRIC_INNER_PRODUCT) {
                heap_reorder<HeapForIP>(k, simi, idxi);
            } else {
                heap_reorder<HeapForL2>(k, simi, idxi);
            }

        } // parallel for
    } // parallel

    indexIVF_stats.nq += n;
    indexIVF_stats.nlist += nlistv;
    indexIVF_stats.ndis += ndis;
    indexIVF_stats.nheap_updates += nheap;
}

template <class HammingComputer, bool store_pairs>
void search_knn_hamming_count(
        const IndexBinaryIVF* ivf,
        size_t nx,
        const uint8_t* __restrict x,
        const idx_t* __restrict keys,
        int k,
        int32_t* __restrict distances,
        idx_t* __restrict labels,
        const IVFSearchParameters* params) {
    const int nBuckets = ivf->d + 1;
    std::vector<int> all_counters(nx * nBuckets, 0);
    std::unique_ptr<idx_t[]> all_ids_per_dis(new idx_t[nx * nBuckets * k]);

    idx_t nprobe = params ? params->nprobe : ivf->nprobe;
    nprobe = std::min((idx_t)ivf->nlist, nprobe);
    idx_t max_codes = params ? params->max_codes : ivf->max_codes;

    std::vector<HCounterState<HammingComputer>> cs;
    for (size_t i = 0; i < nx; ++i) {
        cs.push_back(HCounterState<HammingComputer>(
                all_counters.data() + i * nBuckets,
                all_ids_per_dis.get() + i * nBuckets * k,
                x + i * ivf->code_size,
                ivf->d,
                k));
    }

    size_t nlistv = 0, ndis = 0;

#pragma omp parallel for reduction(+ : nlistv, ndis)
    for (int64_t i = 0; i < nx; i++) {
        const idx_t* keysi = keys + i * nprobe;
        HCounterState<HammingComputer>& csi = cs[i];

        size_t nscan = 0;

        for (size_t ik = 0; ik < nprobe; ik++) {
            idx_t key = keysi[ik]; /* select the list  */
            if (key < 0) {
                // not enough centroids for multiprobe
                continue;
            }
            FAISS_THROW_IF_NOT_FMT(
                    key < (idx_t)ivf->nlist,
                    "Invalid key=%" PRId64 " at ik=%zd nlist=%zd\n",
                    key,
                    ik,
                    ivf->nlist);

            nlistv++;
            size_t list_size = ivf->invlists->list_size(key);
            InvertedLists::ScopedCodes scodes(ivf->invlists, key);
            const uint8_t* list_vecs = scodes.get();
            const idx_t* ids =
                    store_pairs ? nullptr : ivf->invlists->get_ids(key);

            for (size_t j = 0; j < list_size; j++) {
                const uint8_t* yj = list_vecs + ivf->code_size * j;

                idx_t id = store_pairs ? (key << 32 | j) : ids[j];
                csi.update_counter(yj, id);
            }
            if (ids) {
                ivf->invlists->release_ids(key, ids);
            }

            nscan += list_size;
            if (max_codes && nscan >= max_codes)
                break;
        }
        ndis += nscan;

        int nres = 0;
        for (int b = 0; b < nBuckets && nres < k; b++) {
            for (int l = 0; l < csi.counters[b] && nres < k; l++) {
                labels[i * k + nres] = csi.ids_per_dis[b * k + l];
                distances[i * k + nres] = b;
                nres++;
            }
        }
        while (nres < k) {
            labels[i * k + nres] = -1;
            distances[i * k + nres] = std::numeric_limits<int32_t>::max();
            ++nres;
        }
    }

    indexIVF_stats.nq += nx;
    indexIVF_stats.nlist += nlistv;
    indexIVF_stats.ndis += ndis;
}

/* Manages NQ queries at a time, stores results */
template <class HammingComputer, int NQ, int K>
struct BlockSearch {
    HammingComputer hcs[NQ];
    // heaps to update for each query
    int32_t* distances[NQ];
    idx_t* labels[NQ];
    // curent top of heap
    int32_t heap_tops[NQ];

    BlockSearch(
            size_t code_size,
            const uint8_t* __restrict x,
            const int32_t* __restrict keys,
            int32_t* __restrict all_distances,
            idx_t* __restrict all_labels) {
        for (idx_t q = 0; q < NQ; q++) {
            idx_t qno = keys[q];
            hcs[q] = HammingComputer(x + qno * code_size, code_size);
            distances[q] = all_distances + qno * K;
            labels[q] = all_labels + qno * K;
            heap_tops[q] = distances[q][0];
        }
    }

    void add_bcode(const uint8_t* bcode, idx_t id) {
        using C = CMax<int32_t, idx_t>;
        for (int q = 0; q < NQ; q++) {
            int dis = hcs[q].hamming(bcode);
            if (dis < heap_tops[q]) {
                heap_replace_top<C>(K, distances[q], labels[q], dis, id);
                heap_tops[q] = distances[q][0];
            }
        }
    }
};

template <class HammingComputer, int NQ>
struct BlockSearchVariableK {
    int k;
    HammingComputer hcs[NQ];
    // heaps to update for each query
    int32_t* distances[NQ];
    idx_t* labels[NQ];
    // curent top of heap
    int32_t heap_tops[NQ];

    BlockSearchVariableK(
            size_t code_size,
            int k,
            const uint8_t* __restrict x,
            const int32_t* __restrict keys,
            int32_t* __restrict all_distances,
            idx_t* __restrict all_labels)
            : k(k) {
        for (idx_t q = 0; q < NQ; q++) {
            idx_t qno = keys[q];
            hcs[q] = HammingComputer(x + qno * code_size, code_size);
            distances[q] = all_distances + qno * k;
            labels[q] = all_labels + qno * k;
            heap_tops[q] = distances[q][0];
        }
    }

    void add_bcode(const uint8_t* bcode, idx_t id) {
        using C = CMax<int32_t, idx_t>;
        for (int q = 0; q < NQ; q++) {
            int dis = hcs[q].hamming(bcode);
            if (dis < heap_tops[q]) {
                heap_replace_top<C>(k, distances[q], labels[q], dis, id);
                heap_tops[q] = distances[q][0];
            }
        }
    }
};

template <class HammingComputer>
void search_knn_hamming_per_invlist(
        const IndexBinaryIVF* ivf,
        size_t n,
        const uint8_t* __restrict x,
        idx_t k,
        const idx_t* __restrict keys_in,
        const int32_t* __restrict coarse_dis,
        int32_t* __restrict distances,
        idx_t* __restrict labels,
        bool store_pairs,
        const IVFSearchParameters* params) {
    idx_t nprobe = params ? params->nprobe : ivf->nprobe;
    nprobe = std::min((idx_t)ivf->nlist, nprobe);
    idx_t max_codes = params ? params->max_codes : ivf->max_codes;
    FAISS_THROW_IF_NOT(max_codes == 0);
    FAISS_THROW_IF_NOT(!store_pairs);

    // reorder buckets
    std::vector<int64_t> lims(n + 1);
    int32_t* keys = new int32_t[n * nprobe];
    std::unique_ptr<int32_t[]> delete_keys(keys);
    for (idx_t i = 0; i < n * nprobe; i++) {
        keys[i] = keys_in[i];
    }
    matrix_bucket_sort_inplace(n, nprobe, keys, ivf->nlist, lims.data(), 0);

    using C = CMax<int32_t, idx_t>;
    heap_heapify<C>(n * k, distances, labels);
    const size_t code_size = ivf->code_size;

    for (idx_t l = 0; l < ivf->nlist; l++) {
        idx_t l0 = lims[l], nq = lims[l + 1] - l0;

        InvertedLists::ScopedCodes scodes(ivf->invlists, l);
        InvertedLists::ScopedIds sidx(ivf->invlists, l);
        idx_t nb = ivf->invlists->list_size(l);
        const uint8_t* bcodes = scodes.get();
        const idx_t* ids = sidx.get();

        idx_t i = 0;

        // process as much as possible by blocks
        constexpr int BS = 4;

        if (k == 1) {
            for (; i + BS <= nq; i += BS) {
                BlockSearch<HammingComputer, BS, 1> bc(
                        code_size, x, keys + l0 + i, distances, labels);
                for (idx_t j = 0; j < nb; j++) {
                    bc.add_bcode(bcodes + j * code_size, ids[j]);
                }
            }
        } else if (k == 2) {
            for (; i + BS <= nq; i += BS) {
                BlockSearch<HammingComputer, BS, 2> bc(
                        code_size, x, keys + l0 + i, distances, labels);
                for (idx_t j = 0; j < nb; j++) {
                    bc.add_bcode(bcodes + j * code_size, ids[j]);
                }
            }
        } else if (k == 4) {
            for (; i + BS <= nq; i += BS) {
                BlockSearch<HammingComputer, BS, 4> bc(
                        code_size, x, keys + l0 + i, distances, labels);
                for (idx_t j = 0; j < nb; j++) {
                    bc.add_bcode(bcodes + j * code_size, ids[j]);
                }
            }
        } else {
            for (; i + BS <= nq; i += BS) {
                BlockSearchVariableK<HammingComputer, BS> bc(
                        code_size, k, x, keys + l0 + i, distances, labels);
                for (idx_t j = 0; j < nb; j++) {
                    bc.add_bcode(bcodes + j * code_size, ids[j]);
                }
            }
        }

        // leftovers
        for (; i < nq; i++) {
            idx_t qno = keys[l0 + i];
            HammingComputer hc(x + qno * code_size, code_size);
            idx_t* __restrict idxi = labels + qno * k;
            int32_t* __restrict simi = distances + qno * k;
            int32_t simi0 = simi[0];
            for (idx_t j = 0; j < nb; j++) {
                int dis = hc.hamming(bcodes + j * code_size);

                if (dis < simi0) {
                    idx_t id = store_pairs ? lo_build(l, j) : ids[j];
                    heap_replace_top<C>(k, simi, idxi, dis, id);
                    simi0 = simi[0];
                }
            }
        }
    }
    for (idx_t i = 0; i < n; i++) {
        heap_reorder<C>(k, distances + i * k, labels + i * k);
    }
}

struct Run_search_knn_hamming_per_invlist {
    using T = void;

    template <class HammingComputer, class... Types>
    void f(Types... args) {
        search_knn_hamming_per_invlist<HammingComputer>(args...);
    }
};

template <bool store_pairs>
struct Run_search_knn_hamming_count {
    using T = void;

    template <class HammingComputer, class... Types>
    void f(Types... args) {
        search_knn_hamming_count<HammingComputer, store_pairs>(args...);
    }
};

struct BuildScanner {
    using T = BinaryInvertedListScanner*;

    template <class HammingComputer>
    T f(size_t code_size, bool store_pairs) {
        return new IVFBinaryScannerL2<HammingComputer>(code_size, store_pairs);
    }
};

} // anonymous namespace

BinaryInvertedListScanner* IndexBinaryIVF::get_InvertedListScanner(
        bool store_pairs) const {
    BuildScanner bs;
    return dispatch_HammingComputer(code_size, bs, code_size, store_pairs);
}

void IndexBinaryIVF::search_preassigned(
        idx_t n,
        const uint8_t* x,
        idx_t k,
        const idx_t* cidx,
        const int32_t* cdis,
        int32_t* dis,
        idx_t* idx,
        bool store_pairs,
        const IVFSearchParameters* params) const {
    if (per_invlist_search) {
        Run_search_knn_hamming_per_invlist r;
        // clang-format off
        dispatch_HammingComputer(
                code_size, r, this, n, x, k,
                cidx, cdis, dis, idx, store_pairs, params);
        // clang-format on
    } else if (use_heap) {
        search_knn_hamming_heap(
                this, n, x, k, cidx, cdis, dis, idx, store_pairs, params);
    } else if (store_pairs) { // !use_heap && store_pairs
        Run_search_knn_hamming_count<true> r;
        dispatch_HammingComputer(
                code_size, r, this, n, x, cidx, k, dis, idx, params);
    } else { // !use_heap && !store_pairs
        Run_search_knn_hamming_count<false> r;
        dispatch_HammingComputer(
                code_size, r, this, n, x, cidx, k, dis, idx, params);
    }
}

void IndexBinaryIVF::range_search(
        idx_t n,
        const uint8_t* __restrict x,
        int radius,
        RangeSearchResult* __restrict res,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(
            !params, "search params not supported for this index");
    const size_t nprobe_2 = std::min(nlist, this->nprobe);
    std::unique_ptr<idx_t[]> idx(new idx_t[n * nprobe_2]);
    std::unique_ptr<int32_t[]> coarse_dis(new int32_t[n * nprobe_2]);

    double t0 = getmillisecs();
    quantizer->search(n, x, nprobe_2, coarse_dis.get(), idx.get());
    indexIVF_stats.quantization_time += getmillisecs() - t0;

    t0 = getmillisecs();
    invlists->prefetch_lists(idx.get(), n * nprobe_2);

    range_search_preassigned(n, x, radius, idx.get(), coarse_dis.get(), res);

    indexIVF_stats.search_time += getmillisecs() - t0;
}

void IndexBinaryIVF::range_search_preassigned(
        idx_t n,
        const uint8_t* __restrict x,
        int radius,
        const idx_t* __restrict assign,
        const int32_t* __restrict centroid_dis,
        RangeSearchResult* __restrict res) const {
    const size_t nprobe_2 = std::min(nlist, this->nprobe);
    bool store_pairs = false;
    size_t nlistv = 0, ndis = 0;

    std::vector<RangeSearchPartialResult*> all_pres(omp_get_max_threads());

#pragma omp parallel reduction(+ : nlistv, ndis)
    {
        RangeSearchPartialResult pres(res);
        std::unique_ptr<BinaryInvertedListScanner> scanner(
                get_InvertedListScanner(store_pairs));
        FAISS_THROW_IF_NOT(scanner.get());

        all_pres[omp_get_thread_num()] = &pres;

        auto scan_list_func = [&](size_t i, size_t ik, RangeQueryResult& qres) {
            idx_t key = assign[i * nprobe_2 + ik]; /* select the list  */
            if (key < 0)
                return;
            FAISS_THROW_IF_NOT_FMT(
                    key < (idx_t)nlist,
                    "Invalid key=%" PRId64 " at ik=%zd nlist=%zd\n",
                    key,
                    ik,
                    nlist);
            const size_t list_size = invlists->list_size(key);

            if (list_size == 0)
                return;

            InvertedLists::ScopedCodes scodes(invlists, key);
            InvertedLists::ScopedIds ids(invlists, key);

            scanner->set_list(key, assign[i * nprobe_2 + ik]);
            nlistv++;
            ndis += list_size;
            scanner->scan_codes_range(
                    list_size, scodes.get(), ids.get(), radius, qres);
        };

#pragma omp for
        for (idx_t i = 0; i < n; i++) {
            scanner->set_query(x + i * code_size);

            RangeQueryResult& qres = pres.new_result(i);

            for (size_t ik = 0; ik < nprobe_2; ik++) {
                scan_list_func(i, ik, qres);
            }
        }

        pres.finalize();
    }
    indexIVF_stats.nq += n;
    indexIVF_stats.nlist += nlistv;
    indexIVF_stats.ndis += ndis;
}

IndexBinaryIVF::~IndexBinaryIVF() {
    if (own_invlists) {
        delete invlists;
    }

    if (own_fields) {
        delete quantizer;
    }
}

} // namespace faiss
