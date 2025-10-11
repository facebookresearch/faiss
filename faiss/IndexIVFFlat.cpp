/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/IndexIVFFlat.h>

#include <omp.h>

#include <cinttypes>
#include <cstdio>
#include <numeric>

#include <faiss/IndexFlat.h>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/IDSelector.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/extra_distances.h>
#include <faiss/utils/utils.h>

namespace faiss {

/*****************************************
 * IndexIVFFlat implementation
 ******************************************/

IndexIVFFlat::IndexIVFFlat(
        Index* quantizer,
        size_t d,
        size_t nlist,
        MetricType metric,
        bool own_invlists)
        : IndexIVF(
                  quantizer,
                  d,
                  nlist,
                  sizeof(float) * d,
                  metric,
                  own_invlists) {
    code_size = sizeof(float) * d;
    by_residual = false;
}

IndexIVFFlat::IndexIVFFlat() {
    by_residual = false;
}

void IndexIVFFlat::add_core(
        idx_t n,
        const float* x,
        const idx_t* xids,
        const idx_t* coarse_idx,
        void* inverted_list_context) {
    FAISS_THROW_IF_NOT(is_trained);
    FAISS_THROW_IF_NOT(coarse_idx);
    FAISS_THROW_IF_NOT(!by_residual);
    assert(invlists);
    direct_map.check_can_add(xids);

    int64_t n_add = 0;

    DirectMapAdd dm_adder(direct_map, n, xids);

#pragma omp parallel reduction(+ : n_add)
    {
        int nt = omp_get_num_threads();
        int rank = omp_get_thread_num();

        // each thread takes care of a subset of lists
        for (size_t i = 0; i < n; i++) {
            idx_t list_no = coarse_idx[i];

            if (list_no >= 0 && list_no % nt == rank) {
                idx_t id = xids ? xids[i] : ntotal + i;
                const float* xi = x + i * d;
                size_t offset = invlists->add_entry(
                        list_no, id, (const uint8_t*)xi, inverted_list_context);
                dm_adder.add(i, list_no, offset);
                n_add++;
            } else if (rank == 0 && list_no == -1) {
                dm_adder.add(i, -1, 0);
            }
        }
    }

    if (verbose) {
        printf("IndexIVFFlat::add_core: added %" PRId64 " / %" PRId64
               " vectors\n",
               n_add,
               n);
    }
    ntotal += n;
}

void IndexIVFFlat::encode_vectors(
        idx_t n,
        const float* x,
        const idx_t* list_nos,
        uint8_t* codes,
        bool include_listnos) const {
    FAISS_THROW_IF_NOT(!by_residual);
    if (!include_listnos) {
        memcpy(codes, x, code_size * n);
    } else {
        size_t coarse_size = coarse_code_size();
        for (size_t i = 0; i < n; i++) {
            int64_t list_no = list_nos[i];
            uint8_t* code = codes + i * (code_size + coarse_size);
            const float* xi = x + i * d;
            if (list_no >= 0) {
                encode_listno(list_no, code);
                memcpy(code + coarse_size, xi, code_size);
            } else {
                memset(code, 0, code_size + coarse_size);
            }
        }
    }
}

void IndexIVFFlat::decode_vectors(
        idx_t n,
        const uint8_t* codes,
        const idx_t* /*listnos*/,
        float* x) const {
    for (size_t i = 0; i < n; i++) {
        const uint8_t* code = codes + i * code_size;
        float* xi = x + i * d;
        memcpy(xi, code, code_size);
    }
}

void IndexIVFFlat::sa_decode(idx_t n, const uint8_t* bytes, float* x) const {
    size_t coarse_size = coarse_code_size();
    for (size_t i = 0; i < n; i++) {
        const uint8_t* code = bytes + i * (code_size + coarse_size);
        float* xi = x + i * d;
        memcpy(xi, code + coarse_size, code_size);
    }
}

namespace {

template <typename VectorDistance, bool use_sel>
struct IVFFlatScanner : InvertedListScanner {
    VectorDistance vd;
    using C = typename VectorDistance::C;

    IVFFlatScanner(
            const VectorDistance& vd,
            bool store_pairs,
            const IDSelector* sel)
            : InvertedListScanner(store_pairs, sel), vd(vd) {
        keep_max = vd.is_similarity;
        code_size = vd.d * sizeof(float);
    }

    const float* xi;
    void set_query(const float* query) override {
        this->xi = query;
    }

    void set_list(idx_t list_no, float /* coarse_dis */) override {
        this->list_no = list_no;
    }

    float distance_to_code(const uint8_t* code) const override {
        const float* yj = (float*)code;
        return vd(xi, yj);
    }

    size_t scan_codes(
            size_t list_size,
            const uint8_t* codes,
            const idx_t* ids,
            float* simi,
            idx_t* idxi,
            size_t k) const override {
        const float* list_vecs = (const float*)codes;
        size_t nup = 0;
        for (size_t j = 0; j < list_size; j++) {
            const float* yj = list_vecs + vd.d * j;
            if (use_sel && !sel->is_member(ids[j])) {
                continue;
            }
            float dis = vd(xi, yj);
            if (C::cmp(simi[0], dis)) {
                int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                heap_replace_top<C>(k, simi, idxi, dis, id);
                nup++;
            }
        }
        return nup;
    }

    void scan_codes_range(
            size_t list_size,
            const uint8_t* codes,
            const idx_t* ids,
            float radius,
            RangeQueryResult& res) const override {
        const float* list_vecs = (const float*)codes;
        for (size_t j = 0; j < list_size; j++) {
            const float* yj = list_vecs + vd.d * j;
            if (use_sel && !sel->is_member(ids[j])) {
                continue;
            }
            float dis = vd(xi, yj);
            if (C::cmp(radius, dis)) {
                int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                res.add(dis, id);
            }
        }
    }
};

struct Run_get_InvertedListScanner {
    using T = InvertedListScanner*;

    template <class VD>
    InvertedListScanner* f(
            VD& vd,
            const IndexIVFFlat* ivf,
            bool store_pairs,
            const IDSelector* sel) {
        if (sel) {
            return new IVFFlatScanner<VD, true>(vd, store_pairs, sel);
        } else {
            return new IVFFlatScanner<VD, false>(vd, store_pairs, sel);
        }
    }
};

} // anonymous namespace

InvertedListScanner* IndexIVFFlat::get_InvertedListScanner(
        bool store_pairs,
        const IDSelector* sel,
        const IVFSearchParameters*) const {
    Run_get_InvertedListScanner run;
    return dispatch_VectorDistance(
            d, metric_type, metric_arg, run, this, store_pairs, sel);
}

void IndexIVFFlat::reconstruct_from_offset(
        int64_t list_no,
        int64_t offset,
        float* recons) const {
    memcpy(recons, invlists->get_single_code(list_no, offset), code_size);
}

/*****************************************
 * IndexIVFFlatDedup implementation
 ******************************************/

IndexIVFFlatDedup::IndexIVFFlatDedup(
        Index* quantizer,
        size_t d,
        size_t nlist_,
        MetricType metric_type,
        bool own_invlists)
        : IndexIVFFlat(quantizer, d, nlist_, metric_type, own_invlists) {}

void IndexIVFFlatDedup::train(idx_t n, const float* x) {
    std::unordered_map<uint64_t, idx_t> map;
    std::unique_ptr<float[]> x2(new float[n * d]);

    int64_t n2 = 0;
    for (int64_t i = 0; i < n; i++) {
        uint64_t hash = hash_bytes((uint8_t*)(x + i * d), code_size);
        if (map.count(hash) &&
            !memcmp(x2.get() + map[hash] * d, x + i * d, code_size)) {
            // is duplicate, skip
        } else {
            map[hash] = n2;
            memcpy(x2.get() + n2 * d, x + i * d, code_size);
            n2++;
        }
    }
    if (verbose) {
        printf("IndexIVFFlatDedup::train: train on %" PRId64
               " points after dedup "
               "(was %" PRId64 " points)\n",
               n2,
               n);
    }
    IndexIVFFlat::train(n2, x2.get());
}

void IndexIVFFlatDedup::add_with_ids(
        idx_t na,
        const float* x,
        const idx_t* xids) {
    FAISS_THROW_IF_NOT(is_trained);
    assert(invlists);
    FAISS_THROW_IF_NOT_MSG(
            direct_map.no(), "IVFFlatDedup not implemented with direct_map");
    std::unique_ptr<int64_t[]> idx(new int64_t[na]);
    quantizer->assign(na, x, idx.get());

    int64_t n_add = 0, n_dup = 0;

#pragma omp parallel reduction(+ : n_add, n_dup)
    {
        int nt = omp_get_num_threads();
        int rank = omp_get_thread_num();

        // each thread takes care of a subset of lists
        for (size_t i = 0; i < na; i++) {
            int64_t list_no = idx[i];

            if (list_no < 0 || list_no % nt != rank) {
                continue;
            }

            idx_t id = xids ? xids[i] : ntotal + i;
            const float* xi = x + i * d;

            // search if there is already an entry with that id
            InvertedLists::ScopedCodes codes(invlists, list_no);

            int64_t n = invlists->list_size(list_no);
            int64_t offset = -1;
            for (int64_t o = 0; o < n; o++) {
                if (!memcmp(codes.get() + o * code_size, xi, code_size)) {
                    offset = o;
                    break;
                }
            }

            if (offset == -1) { // not found
                invlists->add_entry(list_no, id, (const uint8_t*)xi);
            } else {
                // mark equivalence
                idx_t id2 = invlists->get_single_id(list_no, offset);
                std::pair<idx_t, idx_t> pair(id2, id);

#pragma omp critical
                // executed by one thread at a time
                instances.insert(pair);

                n_dup++;
            }
            n_add++;
        }
    }
    if (verbose) {
        printf("IndexIVFFlat::add_with_ids: added %" PRId64 " / %" PRId64
               " vectors"
               " (out of which %" PRId64 " are duplicates)\n",
               n_add,
               na,
               n_dup);
    }
    ntotal += n_add;
}

void IndexIVFFlatDedup::search_preassigned(
        idx_t n,
        const float* x,
        idx_t k,
        const idx_t* assign,
        const float* centroid_dis,
        float* distances,
        idx_t* labels,
        bool store_pairs,
        const IVFSearchParameters* params,
        IndexIVFStats* stats) const {
    FAISS_THROW_IF_NOT_MSG(
            !store_pairs, "store_pairs not supported in IVFDedup");

    IndexIVFFlat::search_preassigned(
            n, x, k, assign, centroid_dis, distances, labels, false, params);

    std::vector<idx_t> labels2(k);
    std::vector<float> dis2(k);

    for (int64_t i = 0; i < n; i++) {
        idx_t* labels1 = labels + i * k;
        float* dis1 = distances + i * k;
        int64_t j = 0;
        for (; j < k; j++) {
            if (instances.find(labels1[j]) != instances.end()) {
                // a duplicate: special handling
                break;
            }
        }
        if (j < k) {
            // there are duplicates, special handling
            int64_t j0 = j;
            int64_t rp = j;
            while (j < k) {
                auto range = instances.equal_range(labels1[rp]);
                float dis = dis1[rp];
                labels2[j] = labels1[rp];
                dis2[j] = dis;
                j++;
                for (auto it = range.first; j < k && it != range.second; ++it) {
                    labels2[j] = it->second;
                    dis2[j] = dis;
                    j++;
                }
                rp++;
            }
            memcpy(labels1 + j0,
                   labels2.data() + j0,
                   sizeof(labels1[0]) * (k - j0));
            memcpy(dis1 + j0, dis2.data() + j0, sizeof(dis2[0]) * (k - j0));
        }
    }
}

size_t IndexIVFFlatDedup::remove_ids(const IDSelector& sel) {
    std::unordered_map<idx_t, idx_t> replace;
    std::vector<std::pair<idx_t, idx_t>> toadd;
    for (auto it = instances.begin(); it != instances.end();) {
        if (sel.is_member(it->first)) {
            // then we erase this entry
            if (!sel.is_member(it->second)) {
                // if the second is not erased
                if (replace.count(it->first) == 0) {
                    replace[it->first] = it->second;
                } else { // remember we should add an element
                    std::pair<idx_t, idx_t> new_entry(
                            replace[it->first], it->second);
                    toadd.push_back(new_entry);
                }
            }
            it = instances.erase(it);
        } else {
            if (sel.is_member(it->second)) {
                it = instances.erase(it);
            } else {
                ++it;
            }
        }
    }

    instances.insert(toadd.begin(), toadd.end());

    // mostly copied from IndexIVF.cpp

    FAISS_THROW_IF_NOT_MSG(
            direct_map.no(), "direct map remove not implemented");

    std::vector<int64_t> toremove(nlist);

#pragma omp parallel for
    for (int64_t i = 0; i < nlist; i++) {
        int64_t l0 = invlists->list_size(i), l = l0, j = 0;
        InvertedLists::ScopedIds idsi(invlists, i);
        while (j < l) {
            if (sel.is_member(idsi[j])) {
                if (replace.count(idsi[j]) == 0) {
                    l--;
                    invlists->update_entry(
                            i,
                            j,
                            invlists->get_single_id(i, l),
                            InvertedLists::ScopedCodes(invlists, i, l).get());
                } else {
                    invlists->update_entry(
                            i,
                            j,
                            replace[idsi[j]],
                            InvertedLists::ScopedCodes(invlists, i, j).get());
                    j++;
                }
            } else {
                j++;
            }
        }
        toremove[i] = l0 - l;
    }
    // this will not run well in parallel on ondisk because of possible shrinks
    int64_t nremove = 0;
    for (int64_t i = 0; i < nlist; i++) {
        if (toremove[i] > 0) {
            nremove += toremove[i];
            invlists->resize(i, invlists->list_size(i) - toremove[i]);
        }
    }
    ntotal -= nremove;
    return nremove;
}

void IndexIVFFlatDedup::range_search(
        idx_t,
        const float*,
        float,
        RangeSearchResult*,
        const SearchParameters*) const {
    FAISS_THROW_MSG("not implemented");
}

void IndexIVFFlatDedup::update_vectors(int, const idx_t*, const float*) {
    FAISS_THROW_MSG("not implemented");
}

void IndexIVFFlatDedup::reconstruct_from_offset(int64_t, int64_t, float*)
        const {
    FAISS_THROW_MSG("not implemented");
}

// TODO(Alexis): Keeping old impl for reference. Delete this later.
// namespace {

// static uint64_t total_active = 0;
// static uint64_t total_points = 0;

// template <MetricType metric, class C, bool use_sel>
// struct IVFFlatScannerPanorama : InvertedListScanner {
//     size_t d;

//     IVFFlatScannerPanorama(size_t d, bool store_pairs, const IDSelector* sel)
//             : InvertedListScanner(store_pairs, sel), d(d) {
//         keep_max = is_similarity_metric(metric);
//         code_size = d * sizeof(float);
//     }

//     const float* xi;
//     void set_query(const float* query) override {
//         this->xi = query;
//     }

//     void set_list(idx_t list_no, float /* coarse_dis */) override {
//         this->list_no = list_no;
//     }

//     float distance_to_code(const uint8_t* code) const override {
//         const float* yj = (float*)code;
//         float dis = metric == METRIC_INNER_PRODUCT
//                 ? fvec_inner_product(xi, yj, d)
//                 : fvec_L2sqr(xi, yj, d);
//         return dis;
//     }

//     /// add one result for query i
//     inline bool add_result(
//             float dis,
//             idx_t idx,
//             float threshold,
//             idx_t* heap_ids,
//             float* heap_dis,
//             size_t k) const {
//         if (C::cmp(threshold, dis)) {
//             heap_replace_top<C>(k, heap_dis, heap_ids, dis, idx);
//             threshold = heap_dis[0];
//             return true;
//         }
//         return false;
//     }
// };

// template <bool use_sel>
// InvertedListScanner* get_InvertedListScanner1(
//         const IndexIVFFlatPanorama* ivf,
//         bool store_pairs,
//         const IDSelector* sel) {
//     // TODO: Implement inner product
//     if (ivf->metric_type == METRIC_L2) {
//         return new IVFFlatScannerPanorama<
//                 METRIC_L2,
//                 CMax<float, int64_t>,
//                 use_sel>(ivf->d, store_pairs, sel);
//     } else {
//         FAISS_THROW_MSG("metric type not supported");
//     }
// }

// } // anonymous namespace

// void IndexIVFFlatPanorama::add(idx_t n, const float* x) {
//     FAISS_ASSERT(!added);
//     FAISS_ASSERT(d % n_levels == 0);
//     added = true;

//     IndexIVFFlat::add(n, x);

//     size_t new_n = 0;
//     size_t total_batches = 0;
//     n_batches = new size_t[nlist];

//     column_offsets = new size_t[nlist];
//     for (size_t i = 0; i < nlist; i++) {
//         column_offsets[i] = new_n;
//         new_n += invlists->list_size(i) * d;
//     }

//     column_storage = new float[d * n];
//     column_storage_offsets = new float*[nlist];

//     for (size_t list_no = 0; list_no < nlist; list_no++) {
//         size_t list_size = invlists->list_size(list_no);
//         n_batches[list_no] = (list_size + batch_size - 1) / batch_size;
//         size_t col_offset = column_offsets[list_no];
//         column_storage_offsets[list_no] = column_storage + col_offset;

//         for (size_t batch_no = 0; batch_no < n_batches[list_no]; batch_no++)
//         {
//             size_t curr_batch_size =
//                     std::min(list_size - batch_no * batch_size, batch_size);

//             size_t batch_offset = batch_no * batch_size * d;
//             size_t idx_offset = batch_no * batch_size;

//             for (size_t level = 0; level < n_levels; level++) {
//                 size_t level_offset = level * level_width * curr_batch_size;

//                 for (size_t point_idx = 0; point_idx < curr_batch_size;
//                      point_idx++) {
//                     float* dest = column_storage + batch_offset + col_offset
//                     +
//                             level_offset + point_idx * level_width;
//                     const float* point = (float*)invlists->get_single_code(
//                             list_no, idx_offset + point_idx);
//                     size_t start_idx = level * level_width;
//                     size_t end_idx =
//                             std::min(start_idx + level_width, (size_t)d);
//                     size_t copy_size = end_idx - start_idx;
//                     memcpy(dest, point + start_idx, copy_size *
//                     sizeof(float));
//                 }
//             }
//         }
//     }

//     cum_sums = new float[(n_levels + 1) * n];
//     cum_sum_offsets = new size_t[nlist];

//     size_t cum_size = 0;
//     for (size_t list_no = 0; list_no < nlist; list_no++) {
//         cum_sum_offsets[list_no] = cum_size;
//         cum_size += invlists->list_size(list_no) * (n_levels + 1);
//     }

//     std::vector<float> vector(d);

//     for (size_t list_no = 0; list_no < nlist; list_no++) {
//         const idx_t* idx = invlists->get_ids(list_no);
//         size_t list_size = invlists->list_size(list_no);

//         for (size_t batch_no = 0; batch_no < n_batches[list_no]; batch_no++)
//         {
//             size_t curr_batch_size =
//                     std::min(list_size - batch_no * batch_size, batch_size);
//             size_t batch_offset = batch_no * batch_size * (n_levels + 1);
//             size_t index_offset = batch_no * batch_size;

//             for (size_t point = 0; point < curr_batch_size; point++) {
//                 float init_exact_distance = 0.0f;

//                 reconstruct_from_offset(
//                         list_no, point + index_offset, vector.data());

//                 std::vector<float> suffix_sums(d + 1);
//                 suffix_sums[d] = 0.0f;

//                 for (int j = d - 1; j >= 0; j--) {
//                     float squared_val = vector[j] * vector[j];
//                     suffix_sums[j] = suffix_sums[j + 1] + squared_val;
//                 }

//                 // Extract level sums and take square root
//                 for (int level = 0; level < n_levels; level++) {
//                     int start_idx = level * level_width;
//                     size_t offset = cum_sum_offsets[list_no] + batch_offset +
//                             level * curr_batch_size + point;
//                     if (start_idx < d) {
//                         cum_sums[offset] = sqrt(suffix_sums[start_idx]);
//                     } else {
//                         cum_sums[offset] = 0.0f;
//                     }
//                 }

//                 // Last level sum
//                 size_t offset = cum_sum_offsets[list_no] + batch_offset +
//                         n_levels * curr_batch_size + point;
//                 cum_sums[offset] = 0.0f;
//             }
//         }
//     }
// }

// void IndexIVFFlatPanorama::search_preassigned(
//         idx_t n,
//         const float* x,
//         idx_t k,
//         const idx_t* keys, // which clusters
//         const float* coarse_dis,
//         float* distances,
//         idx_t* labels,
//         bool store_pairs,
//         const IVFSearchParameters* params,
//         IndexIVFStats* ivf_stats) const {
//     FAISS_THROW_IF_NOT(k > 0);

//     idx_t nprobe = params ? params->nprobe : this->nprobe;
//     nprobe = std::min((idx_t)nlist, nprobe);
//     FAISS_THROW_IF_NOT(nprobe > 0);

//     const idx_t unlimited_list_size = std::numeric_limits<idx_t>::max();
//     idx_t max_codes = params ? params->max_codes : this->max_codes;
//     IDSelector* sel = params ? params->sel : nullptr;
//     const IDSelectorRange* selr = dynamic_cast<const IDSelectorRange*>(sel);
//     if (selr) {
//         if (selr->assume_sorted) {
//             sel = nullptr; // use special IDSelectorRange processing
//         } else {
//             selr = nullptr; // use generic processing
//         }
//     }

//     FAISS_THROW_IF_NOT_MSG(
//             !(sel && store_pairs),
//             "selector and store_pairs cannot be combined");

//     FAISS_THROW_IF_NOT_MSG(
//             !invlists->use_iterator || (max_codes == 0 && store_pairs ==
//             false), "iterable inverted lists don't support max_codes and
//             store_pairs");

//     size_t nlistv = 0, ndis = 0;

//     using HeapForIP = CMin<float, idx_t>;
//     using HeapForL2 = CMax<float, idx_t>;

//     bool interrupt = false;
//     std::mutex exception_mutex;
//     std::string exception_string;

//     int pmode = this->parallel_mode & ~PARALLEL_MODE_NO_HEAP_INIT;
//     bool do_heap_init = !(this->parallel_mode & PARALLEL_MODE_NO_HEAP_INIT);

//     FAISS_THROW_IF_NOT_MSG(
//             max_codes == 0 || pmode == 0 || pmode == 3,
//             "max_codes supported only for parallel_mode = 0 or 3");

//     if (max_codes == 0) {
//         max_codes = unlimited_list_size;
//     }

//     [[maybe_unused]] bool do_parallel = omp_get_max_threads() >= 2 &&
//             (pmode == 0           ? false
//                      : pmode == 3 ? n > 1
//                      : pmode == 1 ? nprobe > 1
//                                   : nprobe * n > 1);

//     void* inverted_list_context =
//             params ? params->inverted_list_context : nullptr;

//     size_t max_num_codes = 0;
//     for (size_t i = 0; i < nlist; i++) {
//         max_num_codes = std::max(max_num_codes, invlists->list_size(i));
//     }

//     std::vector<float> suffix_sums(d + 1);
//     std::vector<float> query_cum_norms(n_levels + 1);
//     std::vector<float> query(d);
//     std::vector<float> exact_distances(std::min(max_num_codes, batch_size));
//     std::vector<uint32_t> indices(std::min(max_num_codes, batch_size));

// #pragma omp parallel if (do_parallel) reduction(+ : nlistv, ndis)
//     {
//         std::unique_ptr<InvertedListScanner> scanner(
//                 get_InvertedListScanner(store_pairs, sel, params));

//         /*****************************************************
//          * Depending on parallel_mode, there are two possible ways
//          * to organize the search. Here we define local functions
//          * that are in common between the two
//          ******************************************************/

//         // initialize + reorder a result heap

//         auto init_result = [&](float* simi, idx_t* idxi) {
//             if (!do_heap_init)
//                 return;
//             if (metric_type == METRIC_INNER_PRODUCT) {
//                 heap_heapify<HeapForIP>(k, simi, idxi);
//             } else {
//                 heap_heapify<HeapForL2>(k, simi, idxi);
//             }
//         };

//         auto add_local_results = [&](const float* local_dis,
//                                      const idx_t* local_idx,
//                                      float* simi,
//                                      idx_t* idxi) {
//             if (metric_type == METRIC_INNER_PRODUCT) {
//                 heap_addn<HeapForIP>(k, simi, idxi, local_dis, local_idx, k);
//             } else {
//                 heap_addn<HeapForL2>(k, simi, idxi, local_dis, local_idx, k);
//             }
//         };

//         auto reorder_result = [&](float* simi, idx_t* idxi) {
//             if (!do_heap_init)
//                 return;
//             if (metric_type == METRIC_INNER_PRODUCT) {
//                 heap_reorder<HeapForIP>(k, simi, idxi);
//             } else {
//                 heap_reorder<HeapForL2>(k, simi, idxi);
//             }
//         };

//         // single list scan using the current scanner (with query
//         // set porperly) and storing results in simi and idxi
//         auto scan_one_list = [&](const float* query,
//                                  size_t list_no,
//                                  const float* cum_sums,
//                                  const float* query_cum_norms,
//                                  float* exact_distances,
//                                  idx_t cluster_id,
//                                  float* simi,
//                                  idx_t* idxi,
//                                  idx_t list_size_max,
//                                  std::vector<uint32_t>& indices) {
//             if (cluster_id < 0) {
//                 return (size_t)0;
//             }
//             FAISS_THROW_IF_NOT_FMT(
//                     cluster_id < (idx_t)nlist,
//                     "Invalid key=%" PRId64 " nlist=%zd\n",
//                     cluster_id,
//                     nlist);

//             if (invlists->is_empty(cluster_id, inverted_list_context)) {
//                 return (size_t)0;
//             }

//             scanner->set_list(cluster_id, 0);

//             nlistv++;

//             try {
//                 FAISS_ASSERT(!invlists->use_iterator);
//                 size_t list_size = invlists->list_size(cluster_id);
//                 if (list_size > list_size_max) {
//                     list_size = list_size_max;
//                 }

//                 std::unique_ptr<InvertedLists::ScopedIds> sids;
//                 const idx_t* ids = nullptr;

//                 if (!store_pairs) {
//                     sids = std::make_unique<InvertedLists::ScopedIds>(
//                             invlists, cluster_id);
//                     ids = sids->get();
//                 }

//                 if (selr) { // IDSelectorRange
//                     // restrict search to a section of the inverted list
//                     size_t jmin, jmax;
//                     selr->find_sorted_ids_bounds(list_size, ids, &jmin,
//                     &jmax); list_size = jmax - jmin; if (list_size == 0) {
//                         return (size_t)0;
//                     }
//                     ids += jmin;
//                 }

//                 idx_t index_offset = 0;
//                 size_t batch_offset = 0;
//                 size_t batch_incr = batch_size * d;

//                 for (size_t batch_no = 0; batch_no < n_batches[cluster_id];
//                      batch_no++) {
//                     size_t curr_batch_size =
//                             std::min(list_size - index_offset, batch_size);

//                     total_points += curr_batch_size;
//                     std::iota(
//                             indices.begin(),
//                             indices.begin() + curr_batch_size,
//                             0);

//                     // Initialize with the first cum sums of each point.
//                     for (size_t idx = 0; idx < curr_batch_size; idx++) {
//                         float squared_root = cum_sums[idx];
//                         exact_distances[idx] = squared_root * squared_root;
//                     }

//                     // offset by +1
//                     cum_sums += curr_batch_size;

//                     const float* storage =
//                             column_storage_offsets[cluster_id] +
//                             batch_offset;

//                     size_t start_dim = 0;
//                     size_t num_active = curr_batch_size;

//                     for (size_t level = 0; level < n_levels; level++) {
//                         float query_cum_norm = query_cum_norms[level + 1];

//                         size_t next_active = 0;
//                         total_active += num_active;
//                         for (size_t j = 0; j < num_active; j++) {
//                             int64_t idx = indices[j];
//                             const float* yj = storage + level_width * idx;

//                             float dot_product = fvec_inner_product(
//                                     query + start_dim, yj, level_width);

//                             float cum_sum = cum_sums[idx];
//                             float cauchy_schwarz_bound =
//                                     2.0f * cum_sum * query_cum_norm;

//                             exact_distances[idx] -= 2.0f * dot_product;
//                             float new_exact = exact_distances[idx];

//                             float lower_bound =
//                                     new_exact - cauchy_schwarz_bound *
//                                     epsilon;

//                             indices[next_active] = idx;
//                             next_active += simi[0] > lower_bound;
//                         }

//                         cum_sums += curr_batch_size;
//                         start_dim += level_width;
//                         num_active = next_active;
//                         storage += level_width * curr_batch_size;
//                     }

//                     for (size_t j = 0; j < num_active; j++) {
//                         int64_t idx = indices[j];
//                         scanner->add_result(
//                                 exact_distances[idx],
//                                 ids[idx + index_offset],
//                                 simi[0],
//                                 idxi,
//                                 simi,
//                                 k);
//                     }

//                     index_offset += curr_batch_size;
//                     batch_offset += batch_incr;
//                 }

//                 return list_size;
//             } catch (const std::exception& e) {
//                 std::lock_guard<std::mutex> lock(exception_mutex);
//                 exception_string =
//                         demangle_cpp_symbol(typeid(e).name()) + "  " +
//                         e.what();
//                 interrupt = true;
//                 return size_t(0);
//             }
//         };

//         /****************************************************
//          * Actual loops
//          ****************************************************/

//         FAISS_ASSERT(pmode == 0);
//         if (pmode == 0) {
// #pragma omp for
//             for (idx_t i = 0; i < n; i++) {
//                 if (interrupt) {
//                     continue;
//                 }

//                 suffix_sums[d] = 0.0f;

//                 const float* query = x + i * d;

//                 for (int j = d - 1; j >= 0; --j) {
//                     float squared_val = query[j] * query[j];
//                     suffix_sums[j] = suffix_sums[j + 1] + squared_val;
//                 }

//                 // Extract level sums and take square root
//                 for (int level_idx = 0; level_idx < n_levels; level_idx++) {
//                     int start_idx = level_idx * level_width;
//                     if (start_idx < d) {
//                         query_cum_norms[level_idx] =
//                         sqrt(suffix_sums[start_idx]);
//                     } else {
//                         query_cum_norms[level_idx] = 0.0f;
//                     }
//                 }
//                 query_cum_norms[n_levels] = 0.0f;

//                 scanner->set_query(x + i * d);

//                 float* simi = distances + i * k;
//                 idx_t* idxi = labels + i * k;

//                 init_result(simi, idxi);

//                 idx_t nscan = 0;

//                 for (size_t list_no = 0; list_no < nprobe; list_no++) {
//                     idx_t cluster_id = keys[i * nprobe + list_no];

//                     size_t cum_sum_offset = cum_sum_offsets[cluster_id];
//                     const float* cum_sumss = cum_sums + cum_sum_offset;

//                     nscan += scan_one_list(
//                             x + i * d,
//                             list_no,
//                             cum_sumss,
//                             query_cum_norms.data(),
//                             exact_distances.data(),
//                             cluster_id,
//                             simi,
//                             idxi,
//                             max_codes - nscan,
//                             indices);
//                 }

//                 ndis += nscan;
//                 reorder_result(simi, idxi);

//                 if (InterruptCallback::is_interrupted()) {
//                     interrupt = true;
//                 }
//             }
//         }
//     }
//     if (interrupt) {
//         if (!exception_string.empty()) {
//             FAISS_THROW_FMT(
//                     "search interrupted with: %s", exception_string.c_str());
//         } else {
//             FAISS_THROW_MSG("computation interrupted");
//         }
//     }

//     if (ivf_stats == nullptr) {
//         ivf_stats = &indexIVF_stats;
//     }
//     ivf_stats->nq += n;
//     ivf_stats->nlist += nlistv;
//     ivf_stats->ndis += ndis;

//     printf("avg_level: %f\n", (float)total_active / (total_points *
//     n_levels));
// }

} // namespace faiss
