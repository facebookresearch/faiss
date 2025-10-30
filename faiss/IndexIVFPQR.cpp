/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/IndexIVFPQR.h>

#include <cinttypes>

#include <faiss/utils/Heap.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/utils.h>

#include <faiss/impl/FaissAssert.h>

namespace faiss {

/*****************************************
 * IndexIVFPQR implementation
 ******************************************/

IndexIVFPQR::IndexIVFPQR(
        Index* quantizer,
        size_t d,
        size_t nlist,
        size_t M,
        size_t nbits_per_idx,
        size_t M_refine,
        size_t nbits_per_idx_refine,
        bool own_invlists)
        : IndexIVFPQ(
                  quantizer,
                  d,
                  nlist,
                  M,
                  nbits_per_idx,
                  METRIC_L2,
                  own_invlists),
          refine_pq(d, M_refine, nbits_per_idx_refine),
          k_factor(4) {
    by_residual = true;
    refine_pq.cp.max_points_per_centroid = 1000;
}

IndexIVFPQR::IndexIVFPQR() : k_factor(1) {
    by_residual = true;
    refine_pq.cp.max_points_per_centroid = 1000;
}

void IndexIVFPQR::reset() {
    IndexIVFPQ::reset();
    refine_codes.clear();
}

void IndexIVFPQR::train_encoder(idx_t n, const float* x, const idx_t* assign) {
    IndexIVFPQ::train_encoder(n, x, assign);
    if (verbose) {
        printf("training %zdx%zd 2nd level PQ quantizer on %" PRId64
               " %dD-vectors\n",
               refine_pq.M,
               refine_pq.ksub,
               n,
               d);
    }
    refine_pq.cp.verbose = verbose;

    // 2nd level residual
    std::vector<float> residual_2(n * d);
    std::vector<uint8_t> train_codes(pq.code_size * n);
    pq.compute_codes(x, train_codes.data(), n);

    for (idx_t i = 0; i < n; i++) {
        const float* xx = x + i * d;
        float* res = residual_2.data() + i * d;
        pq.decode(train_codes.data() + i * pq.code_size, res);
        for (int j = 0; j < d; j++) {
            res[j] = xx[j] - res[j];
        }
    }

    refine_pq.train(n, residual_2.data());
}

idx_t IndexIVFPQR::train_encoder_num_vectors() const {
    return std::max(
            pq.cp.max_points_per_centroid * pq.ksub,
            refine_pq.cp.max_points_per_centroid * refine_pq.ksub);
}

void IndexIVFPQR::add_with_ids(idx_t n, const float* x, const idx_t* xids) {
    add_core(n, x, xids, nullptr);
}

void IndexIVFPQR::add_core(
        idx_t n,
        const float* x,
        const idx_t* xids,
        const idx_t* precomputed_idx,
        void* /*inverted_list_context*/) {
    std::unique_ptr<float[]> residual_2(new float[n * d]);

    idx_t n0 = ntotal;

    add_core_o(n, x, xids, residual_2.get(), precomputed_idx);

    refine_codes.resize(ntotal * refine_pq.code_size);

    refine_pq.compute_codes(
            residual_2.get(), &refine_codes[n0 * refine_pq.code_size], n);
}
#define TIC t0 = get_cycles()
#define TOC get_cycles() - t0

void IndexIVFPQR::search_preassigned(
        idx_t n,
        const float* x,
        idx_t k,
        const idx_t* idx,
        const float* L1_dis,
        float* distances,
        idx_t* labels,
        bool store_pairs,
        const IVFSearchParameters* params,
        IndexIVFStats* stats) const {
    uint64_t t0;
    TIC;
    size_t k_coarse = long(k * k_factor);
    std::unique_ptr<idx_t[]> coarse_labels(new idx_t[k_coarse * n]);
    {
        // query with quantizer levels 1 and 2.
        std::unique_ptr<float[]> coarse_distances(new float[k_coarse * n]);

        IndexIVFPQ::search_preassigned(
                n,
                x,
                k_coarse,
                idx,
                L1_dis,
                coarse_distances.get(),
                coarse_labels.get(),
                true,
                params);
    }

    indexIVFPQ_stats.search_cycles += TOC;

    TIC;

    // 3rd level refinement
    size_t n_refine = 0;
#pragma omp parallel reduction(+ : n_refine)
    {
        // tmp buffers
        std::unique_ptr<float[]> residual_1(new float[2 * d]);
        float* residual_2 = residual_1.get() + d;
#pragma omp for
        for (idx_t i = 0; i < n; i++) {
            const float* xq = x + i * d;
            const idx_t* shortlist = coarse_labels.get() + k_coarse * i;
            float* heap_sim = distances + k * i;
            idx_t* heap_ids = labels + k * i;
            maxheap_heapify(k, heap_sim, heap_ids);

            for (int j = 0; j < k_coarse; j++) {
                idx_t sl = shortlist[j];

                if (sl == -1) {
                    continue;
                }

                int list_no = lo_listno(sl);
                int ofs = lo_offset(sl);

                assert(list_no >= 0 && list_no < nlist);
                assert(ofs >= 0 && ofs < invlists->list_size(list_no));

                // 1st level residual
                quantizer->compute_residual(xq, residual_1.get(), list_no);

                // 2nd level residual
                const uint8_t* l2code = invlists->get_single_code(list_no, ofs);

                pq.decode(l2code, residual_2);
                for (int l = 0; l < d; l++) {
                    residual_2[l] = residual_1[l] - residual_2[l];
                }

                // 3rd level residual's approximation
                idx_t id = invlists->get_single_id(list_no, ofs);
                assert(0 <= id && id < ntotal);
                refine_pq.decode(
                        &refine_codes[id * refine_pq.code_size],
                        residual_1.get());

                float dis = fvec_L2sqr(residual_1.get(), residual_2, d);

                if (dis < heap_sim[0]) {
                    idx_t id_or_pair = store_pairs ? sl : id;
                    maxheap_replace_top(k, heap_sim, heap_ids, dis, id_or_pair);
                }
                n_refine++;
            }
            maxheap_reorder(k, heap_sim, heap_ids);
        }
    }
    indexIVFPQ_stats.nrefine += n_refine;
    indexIVFPQ_stats.refine_cycles += TOC;
}

void IndexIVFPQR::reconstruct_from_offset(
        int64_t list_no,
        int64_t offset,
        float* recons) const {
    IndexIVFPQ::reconstruct_from_offset(list_no, offset, recons);

    idx_t id = invlists->get_single_id(list_no, offset);
    assert(0 <= id && id < ntotal);

    std::vector<float> r3(d);
    refine_pq.decode(&refine_codes[id * refine_pq.code_size], r3.data());
    for (int i = 0; i < d; ++i) {
        recons[i] += r3[i];
    }
}

void IndexIVFPQR::merge_from(Index& otherIndex, idx_t add_id) {
    IndexIVFPQR* other = dynamic_cast<IndexIVFPQR*>(&otherIndex);
    FAISS_THROW_IF_NOT(other);

    IndexIVF::merge_from(otherIndex, add_id);

    refine_codes.insert(
            refine_codes.end(),
            other->refine_codes.begin(),
            other->refine_codes.end());
    other->refine_codes.clear();
}

size_t IndexIVFPQR::remove_ids(const IDSelector& /*sel*/) {
    FAISS_THROW_MSG("not implemented");
    return 0;
}

} // namespace faiss
