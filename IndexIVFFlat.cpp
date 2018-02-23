/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

/* Copyright 2004-present Facebook. All Rights Reserved.
   Inverted list structure.
*/

#include "IndexIVFFlat.h"

#include <cstdio>

#include "utils.h"

#include "FaissAssert.h"
#include "IndexFlat.h"
#include "AuxIndexStructures.h"

namespace faiss {


/*****************************************
 * IndexIVFFlat implementation
 ******************************************/

IndexIVFFlat::IndexIVFFlat (Index * quantizer,
                            size_t d, size_t nlist, MetricType metric):
    IndexIVF (quantizer, d, nlist, sizeof(float) * d, metric)
{
    code_size = sizeof(float) * d;
}




void IndexIVFFlat::add_with_ids (idx_t n, const float * x, const long *xids)
{
    add_core (n, x, xids, nullptr);
}

void IndexIVFFlat::add_core (idx_t n, const float * x, const long *xids,
                             const long *precomputed_idx)

{
    FAISS_THROW_IF_NOT (is_trained);
    assert (invlists);
    FAISS_THROW_IF_NOT_MSG (!(maintain_direct_map && xids),
                            "cannot have direct map and add with ids");
    const long * idx;
    ScopeDeleter<long> del;

    if (precomputed_idx) {
        idx = precomputed_idx;
    } else {
        long * idx0 = new long [n];
        del.set (idx0);
        quantizer->assign (n, x, idx0);
        idx = idx0;
    }
    long n_add = 0;
    for (size_t i = 0; i < n; i++) {
        long id = xids ? xids[i] : ntotal + i;
        long list_no = idx [i];

        if (list_no < 0)
            continue;
        const float *xi = x + i * d;
        size_t offset = invlists->add_entry (
              list_no, id, (const uint8_t*) xi);

        if (maintain_direct_map)
            direct_map.push_back (list_no << 32 | offset);
        n_add++;
    }
    if (verbose) {
        printf("IndexIVFFlat::add_core: added %ld / %ld vectors\n",
               n_add, n);
    }
    ntotal += n_add;
}


namespace {

void search_knn_inner_product (const IndexIVFFlat & ivf,
                               size_t nx,
                               const float * x,
                               const long * keys,
                               float_minheap_array_t * res,
                               bool store_pairs)
{

    const size_t k = res->k;
    size_t nlistv = 0, ndis = 0;
    size_t d = ivf.d;

#pragma omp parallel for reduction(+: nlistv, ndis)
    for (size_t i = 0; i < nx; i++) {
        const float * xi = x + i * d;
        const long * keysi = keys + i * ivf.nprobe;
        float * __restrict simi = res->get_val (i);
        long * __restrict idxi = res->get_ids (i);
        minheap_heapify (k, simi, idxi);
        size_t nscan = 0;

        for (size_t ik = 0; ik < ivf.nprobe; ik++) {
            long key = keysi[ik];  /* select the list  */
            if (key < 0) {
                // not enough centroids for multiprobe
                continue;
            }
            FAISS_THROW_IF_NOT_FMT (
                key < (long) ivf.nlist,
                "Invalid key=%ld  at ik=%ld nlist=%ld\n",
                key, ik, ivf.nlist);

            nlistv++;
            size_t list_size = ivf.invlists->list_size(key);
            const float * list_vecs =
                (const float*)ivf.invlists->get_codes (key);
            const Index::idx_t * ids = store_pairs ? nullptr :
                ivf.invlists->get_ids (key);

            for (size_t j = 0; j < list_size; j++) {
                const float * yj = list_vecs + d * j;
                float ip = fvec_inner_product (xi, yj, d);
                if (ip > simi[0]) {
                    minheap_pop (k, simi, idxi);
                    long id = store_pairs ? (key << 32 | j) : ids[j];
                    minheap_push (k, simi, idxi, ip, id);
                }
            }
            nscan += list_size;
            if (ivf.max_codes && nscan >= ivf.max_codes)
                break;
        }
        ndis += nscan;
        minheap_reorder (k, simi, idxi);
    }
    indexIVF_stats.nq += nx;
    indexIVF_stats.nlist += nlistv;
    indexIVF_stats.ndis += ndis;
}


void search_knn_L2sqr (const IndexIVFFlat &ivf,
                       size_t nx,
                       const float * x,
                       const long * keys,
                       float_maxheap_array_t * res,
                       bool store_pairs)
{
    const size_t k = res->k;
    size_t nlistv = 0, ndis = 0;
    size_t d = ivf.d;
#pragma omp parallel for reduction(+: nlistv, ndis)
    for (size_t i = 0; i < nx; i++) {
        const float * xi = x + i * d;
        const long * keysi = keys + i * ivf.nprobe;
        float * __restrict disi = res->get_val (i);
        long * __restrict idxi = res->get_ids (i);
        maxheap_heapify (k, disi, idxi);

        size_t nscan = 0;

        for (size_t ik = 0; ik < ivf.nprobe; ik++) {
            long key = keysi[ik];  /* select the list  */
            if (key < 0) {
                // not enough centroids for multiprobe
                continue;
            }
            FAISS_THROW_IF_NOT_FMT (
                key < (long) ivf.nlist,
                "Invalid key=%ld  at ik=%ld nlist=%ld\n",
                key, ik, ivf.nlist);

            nlistv++;
            size_t list_size = ivf.invlists->list_size(key);
            const float * list_vecs =
                (const float*)ivf.invlists->get_codes (key);
            const Index::idx_t * ids = store_pairs ? nullptr :
                ivf.invlists->get_ids (key);

            for (size_t j = 0; j < list_size; j++) {
                const float * yj = list_vecs + d * j;
                float disij = fvec_L2sqr (xi, yj, d);
                if (disij < disi[0]) {
                    maxheap_pop (k, disi, idxi);
                    long id = store_pairs ? (key << 32 | j) : ids[j];
                    maxheap_push (k, disi, idxi, disij, id);
                }
            }
            nscan += list_size;
            if (ivf.max_codes && nscan >= ivf.max_codes)
                break;
        }
        ndis += nscan;
        maxheap_reorder (k, disi, idxi);
    }
    indexIVF_stats.nq += nx;
    indexIVF_stats.nlist += nlistv;
    indexIVF_stats.ndis += ndis;
}


} // anonymous namespace

void IndexIVFFlat::search_preassigned (idx_t n, const float *x, idx_t k,
                                     const idx_t *idx,
                                      const float * /* coarse_dis */,
                                      float *distances, idx_t *labels,
                                      bool store_pairs) const
{
   if (metric_type == METRIC_INNER_PRODUCT) {
        float_minheap_array_t res = {
            size_t(n), size_t(k), labels, distances};
        search_knn_inner_product (*this, n, x, idx, &res, store_pairs);

    } else if (metric_type == METRIC_L2) {
        float_maxheap_array_t res = {
            size_t(n), size_t(k), labels, distances};
        search_knn_L2sqr (*this, n, x, idx, &res, store_pairs);
    }
}


void IndexIVFFlat::range_search (idx_t nx, const float *x, float radius,
                                 RangeSearchResult *result) const
{
    idx_t * keys = new idx_t [nx * nprobe];
    ScopeDeleter<idx_t> del (keys);
    quantizer->assign (nx, x, keys, nprobe);

#pragma omp parallel
    {
        RangeSearchPartialResult pres(result);

        for (size_t i = 0; i < nx; i++) {
            const float * xi = x + i * d;
            const long * keysi = keys + i * nprobe;

            RangeSearchPartialResult::QueryResult & qres =
                pres.new_result (i);

            for (size_t ik = 0; ik < nprobe; ik++) {
                long key = keysi[ik];  /* select the list  */
                if (key < 0 || key >= (long) nlist) {
                    fprintf (stderr, "Invalid key=%ld  at ik=%ld nlist=%ld\n",
                             key, ik, nlist);
                    throw;
                }

                const size_t list_size = invlists->list_size(key);
                const float * list_vecs =
                    (const float*)invlists->get_codes (key);
                const Index::idx_t * ids = invlists->get_ids (key);

                for (size_t j = 0; j < list_size; j++) {
                    const float * yj = list_vecs + d * j;
                    if (metric_type == METRIC_L2) {
                        float disij = fvec_L2sqr (xi, yj, d);
                        if (disij < radius) {
                            qres.add (disij, ids[j]);
                        }
                    } else if (metric_type == METRIC_INNER_PRODUCT) {
                        float disij = fvec_inner_product(xi, yj, d);
                        if (disij > radius) {
                            qres.add (disij, ids[j]);
                        }
                    }
                }
            }
        }

        pres.finalize ();
    }
}

void IndexIVFFlat::update_vectors (int n, idx_t *new_ids, const float *x)
{

    FAISS_THROW_IF_NOT (maintain_direct_map);
    FAISS_THROW_IF_NOT (is_trained);
    std::vector<idx_t> assign (n);
    quantizer->assign (n, x, assign.data());

    for (size_t i = 0; i < n; i++) {
        idx_t id = new_ids[i];
        FAISS_THROW_IF_NOT_MSG (0 <= id && id < ntotal,
                                "id to update out of range");
        { // remove old one
            long dm = direct_map[id];
            long ofs = dm & 0xffffffff;
            long il = dm >> 32;
            size_t l = invlists->list_size (il);
            if (ofs != l - 1) { // move l - 1 to ofs
                long id2 = invlists->get_single_id (il, l - 1);
                direct_map[id2] = (il << 32) | ofs;
                invlists->update_entry (il, ofs, id2,
                                        invlists->get_single_code (il, l - 1));
            }
            invlists->resize (il, l - 1);
        }
        { // insert new one
            long il = assign[i];
            size_t l = invlists->list_size (il);
            long dm = (il << 32) | l;
            direct_map[id] = dm;
            invlists->add_entry (il, id, (const uint8_t*)(x + i * d));
        }
    }

}

void IndexIVFFlat::reconstruct_from_offset (long list_no, long offset,
                                            float* recons) const
{
    memcpy (recons, invlists->get_single_code (list_no, offset), code_size);
}



} // namespace faiss
