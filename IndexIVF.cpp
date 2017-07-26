/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the CC-by-NC license found in the
 * LICENSE file in the root directory of this source tree.
 */

/* Copyright 2004-present Facebook. All Rights Reserved.
   Inverted list structure.
*/

#include "IndexIVF.h"

#include <cstdio>

#include "utils.h"
#include "hamming.h"

#include "FaissAssert.h"
#include "IndexFlat.h"
#include "AuxIndexStructures.h"

namespace faiss {

/*****************************************
 * IndexIVF implementation
 ******************************************/


IndexIVF::IndexIVF (Index * quantizer, size_t d, size_t nlist,
                    MetricType metric):
    Index (d, metric),
    nlist (nlist),
    nprobe (1),
    quantizer (quantizer),
    quantizer_trains_alone (false),
    own_fields (false),
    ids (nlist),
    maintain_direct_map (false)
{
    FAISS_THROW_IF_NOT (d == quantizer->d);
    is_trained = quantizer->is_trained && (quantizer->ntotal == nlist);
    // Spherical by default if the metric is inner_product
    if (metric_type == METRIC_INNER_PRODUCT) {
        cp.spherical = true;
    }
    // here we set a low # iterations because this is typically used
    // for large clusterings (nb this is not used for the MultiIndex,
    // for which quantizer_trains_alone = true)
    cp.niter = 10;
    cp.verbose = verbose;

}

IndexIVF::IndexIVF ():
    nlist (0), nprobe (1), quantizer (nullptr),
    quantizer_trains_alone (false), own_fields (false),
    maintain_direct_map (false)
{}


void IndexIVF::add (idx_t n, const float * x)
{
    add_with_ids (n, x, nullptr);
}

void IndexIVF::make_direct_map (bool new_maintain_direct_map)
{
    // nothing to do
    if (new_maintain_direct_map == maintain_direct_map)
        return;

    if (new_maintain_direct_map) {
        direct_map.resize (ntotal, -1);
        for (size_t key = 0; key < nlist; key++) {
            const std::vector<long> & idlist = ids[key];

            for (long ofs = 0; ofs < idlist.size(); ofs++) {
                FAISS_THROW_IF_NOT_MSG (
                       0 <= idlist [ofs] && idlist[ofs] < ntotal,
                       "direct map supported only for seuquential ids");
                direct_map [idlist [ofs]] = key << 32 | ofs;
            }
        }
    } else {
        direct_map.clear ();
    }
    maintain_direct_map = new_maintain_direct_map;
}


void IndexIVF::reset ()
{
    ntotal = 0;
    direct_map.clear();
    for (size_t i = 0; i < ids.size(); i++)
        ids[i].clear();
}


void IndexIVF::train (idx_t n, const float *x)
{
    if (quantizer->is_trained && (quantizer->ntotal == nlist)) {
        if (verbose)
            printf ("IVF quantizer does not need training.\n");
    } else if (quantizer_trains_alone) {
        if (verbose)
            printf ("IVF quantizer trains alone...\n");
        quantizer->train (n, x);
        FAISS_THROW_IF_NOT_MSG (quantizer->ntotal == nlist,
                          "nlist not consistent with quantizer size");
    } else {
        if (verbose)
            printf ("Training IVF quantizer on %ld vectors in %dD\n",
                    n, d);

        Clustering clus (d, nlist, cp);

        quantizer->reset();
        clus.train (n, x, *quantizer);
        quantizer->is_trained = true;
    }
    if (verbose)
        printf ("Training IVF residual\n");

    train_residual (n, x);
    is_trained = true;
}

void IndexIVF::train_residual(idx_t /*n*/, const float* /*x*/) {
  if (verbose)
    printf("IndexIVF: no residual training\n");
  // does nothing by default
}



double IndexIVF::imbalance_factor () const
{
    std::vector<int> hist (nlist);
    for (int i = 0; i < nlist; i++) {
        hist[i] = ids[i].size();
    }
    return faiss::imbalance_factor (nlist, hist.data());
}

void IndexIVF::print_stats () const
{
    std::vector<int> sizes(40);
    for (int i = 0; i < nlist; i++) {
        for (int j = 0; j < sizes.size(); j++) {
            if ((ids[i].size() >> j) == 0) {
                sizes[j]++;
                break;
            }
        }
    }
    for (int i = 0; i < sizes.size(); i++) {
        if (sizes[i]) {
            printf ("list size in < %d: %d instances\n",
                    1 << i, sizes[i]);
        }
    }

}

void IndexIVF::merge_from (IndexIVF &other, idx_t add_id)
{
    // minimal sanity checks
    FAISS_THROW_IF_NOT (other.d == d);
    FAISS_THROW_IF_NOT (other.nlist == nlist);
    FAISS_THROW_IF_NOT_MSG ((!maintain_direct_map &&
                             !other.maintain_direct_map),
                  "direct map copy not implemented");
    FAISS_THROW_IF_NOT_MSG (typeid (*this) == typeid (other),
                  "can only merge indexes of the same type");
    for (long i = 0; i < nlist; i++) {
        std::vector<idx_t> & src = other.ids[i];
        std::vector<idx_t> & dest = ids[i];
        for (long j = 0; j < src.size(); j++)
            dest.push_back (src[j] + add_id);
        src.clear();
    }
    merge_from_residuals (other);
    ntotal += other.ntotal;
    other.ntotal = 0;
}



IndexIVF::~IndexIVF()
{
    if (own_fields) delete quantizer;
}



/*****************************************
 * IndexIVFFlat implementation
 ******************************************/

IndexIVFFlat::IndexIVFFlat (Index * quantizer,
                            size_t d, size_t nlist, MetricType metric):
    IndexIVF (quantizer, d, nlist, metric)
{
    vecs.resize (nlist);
}






void IndexIVFFlat::add_with_ids (idx_t n, const float * x, const long *xids)
{
    add_core (n, x, xids, nullptr);
}

void IndexIVFFlat::add_core (idx_t n, const float * x, const long *xids,
                             const long *precomputed_idx)

{
    FAISS_THROW_IF_NOT (is_trained);
    FAISS_THROW_IF_NOT_MSG (!(maintain_direct_map && xids),
                            "cannot have direct map and add with ids");
    const long * idx;
    ScopeDeleter<long> del;

    if (precomputed_idx) {
        idx = precomputed_idx;
    } else {
        long * idx0 = new long [n];
        quantizer->assign (n, x, idx0);
        idx = idx0;
        del.set (idx);
    }
    long n_add = 0;
    for (size_t i = 0; i < n; i++) {
        long id = xids ? xids[i] : ntotal + i;
        long list_no = idx [i];
        if (list_no < 0)
            continue;
        assert (list_no < nlist);

        ids[list_no].push_back (id);
        const float *xi = x + i * d;
        /* store the vectors */
        for (size_t j = 0 ; j < d ; j++)
            vecs[list_no].push_back (xi [j]);

        if (maintain_direct_map)
            direct_map.push_back (list_no << 32 | (ids[list_no].size() - 1));
        n_add++;
    }
    if (verbose) {
        printf("IndexIVFFlat::add_core: added %ld / %ld vectors\n",
               n_add, n);
    }
    ntotal += n_add;
}

void IndexIVFFlatStats::reset()
{
    memset ((void*)this, 0, sizeof (*this));
}


IndexIVFFlatStats indexIVFFlat_stats;

void IndexIVFFlat::search_knn_inner_product (
    size_t nx,
    const float * x,
    const long * __restrict keys,
    float_minheap_array_t * res) const
{

    const size_t k = res->k;
    size_t nlistv = 0, ndis = 0;

#pragma omp parallel for reduction(+: nlistv, ndis)
    for (size_t i = 0; i < nx; i++) {
        const float * xi = x + i * d;
        const long * keysi = keys + i * nprobe;
        float * __restrict simi = res->get_val (i);
        long * __restrict idxi = res->get_ids (i);
        minheap_heapify (k, simi, idxi);

        for (size_t ik = 0; ik < nprobe; ik++) {
            long key = keysi[ik];  /* select the list  */
            if (key < 0) {
                // not enough centroids for multiprobe
                continue;
            }
            if (key >= (long) nlist) {
                fprintf (stderr, "Invalid key=%ld  at ik=%ld nlist=%ld\n",
                                  key, ik, nlist);
                throw;
            }
            nlistv++;
            const size_t list_size = ids[key].size();
            const float * list_vecs = vecs[key].data();

            for (size_t j = 0; j < list_size; j++) {
                const float * yj = list_vecs + d * j;
                float ip = fvec_inner_product (xi, yj, d);
                if (ip > simi[0]) {
                    minheap_pop (k, simi, idxi);
                    minheap_push (k, simi, idxi, ip, ids[key][j]);
                }
            }
            ndis += list_size;
        }
        minheap_reorder (k, simi, idxi);
    }
    indexIVFFlat_stats.nq += nx;
    indexIVFFlat_stats.nlist += nlistv;
    indexIVFFlat_stats.ndis += ndis;
}


void IndexIVFFlat::search_knn_L2sqr (
    size_t nx,
    const float * x,
    const long * __restrict keys,
    float_maxheap_array_t * res) const
{
    const size_t k = res->k;
    size_t nlistv = 0, ndis = 0;

#pragma omp parallel for reduction(+: nlistv, ndis)
    for (size_t i = 0; i < nx; i++) {
        const float * xi = x + i * d;
        const long * keysi = keys + i * nprobe;
        float * __restrict disi = res->get_val (i);
        long * __restrict idxi = res->get_ids (i);
        maxheap_heapify (k, disi, idxi);

        for (size_t ik = 0; ik < nprobe; ik++) {
            long key = keysi[ik];  /* select the list  */
            if (key < 0) {
                // not enough centroids for multiprobe
                continue;
            }
            if (key >= (long) nlist) {
                fprintf (stderr, "Invalid key=%ld  at ik=%ld nlist=%ld\n",
                                  key, ik, nlist);
                throw;
            }
            nlistv++;
            const size_t list_size = ids[key].size();
            const float * list_vecs = vecs[key].data();

            for (size_t j = 0; j < list_size; j++) {
                const float * yj = list_vecs + d * j;
                float disij = fvec_L2sqr (xi, yj, d);
                if (disij < disi[0]) {
                    maxheap_pop (k, disi, idxi);
                    maxheap_push (k, disi, idxi, disij, ids[key][j]);
                }
            }
            ndis += list_size;
        }
        maxheap_reorder (k, disi, idxi);
    }
    indexIVFFlat_stats.nq += nx;
    indexIVFFlat_stats.nlist += nlistv;
    indexIVFFlat_stats.ndis += ndis;
}


void IndexIVFFlat::search (idx_t n, const float *x, idx_t k,
                                float *distances, idx_t *labels) const
{
    idx_t * idx = new idx_t [n * nprobe];
    ScopeDeleter <idx_t> del (idx);
    quantizer->assign (n, x, idx, nprobe);
    search_preassigned (n, x, k, idx, distances, labels);
}


void IndexIVFFlat::search_preassigned (idx_t n, const float *x, idx_t k,
                                       const idx_t *idx,
                                       float *distances, idx_t *labels) const
{
   if (metric_type == METRIC_INNER_PRODUCT) {
        float_minheap_array_t res = {
            size_t(n), size_t(k), labels, distances};
        search_knn_inner_product (n, x, idx, &res);

    } else if (metric_type == METRIC_L2) {
        float_maxheap_array_t res = {
            size_t(n), size_t(k), labels, distances};
        search_knn_L2sqr (n, x, idx, &res);
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

                const size_t list_size = ids[key].size();
                const float * list_vecs = vecs[key].data();

                for (size_t j = 0; j < list_size; j++) {
                    const float * yj = list_vecs + d * j;
                    if (metric_type == METRIC_L2) {
                        float disij = fvec_L2sqr (xi, yj, d);
                        if (disij < radius) {
                            qres.add (disij, ids[key][j]);
                        }
                    } else if (metric_type == METRIC_INNER_PRODUCT) {
                        float disij = fvec_inner_product(xi, yj, d);
                        if (disij > radius) {
                            qres.add (disij, ids[key][j]);
                        }
                    }
                }
            }
        }

        pres.finalize ();
    }
}

void IndexIVFFlat::merge_from_residuals (IndexIVF &other_in)
{
    IndexIVFFlat &other = dynamic_cast<IndexIVFFlat &> (other_in);
    for (int i = 0; i < nlist; i++) {
        std::vector<float> & src = other.vecs[i];
        std::vector<float> & dest = vecs[i];
        for (int j = 0; j < src.size(); j++)
            dest.push_back (src[j]);
        src.clear();
    }
}

void IndexIVFFlat::copy_subset_to (IndexIVFFlat & other, int subset_type,
                     long a1, long a2) const
{
    FAISS_THROW_IF_NOT (nlist == other.nlist);
    FAISS_THROW_IF_NOT (!other.maintain_direct_map);

    for (long list_no = 0; list_no < nlist; list_no++) {
        const std::vector<idx_t> & ids_in = ids[list_no];
        std::vector<idx_t> & ids_out = other.ids[list_no];
        const std::vector<float> & vecs_in = vecs[list_no];
        std::vector<float> & vecs_out = other.vecs[list_no];

        for (long i = 0; i < ids_in.size(); i++) {
            idx_t id = ids_in[i];
            if (subset_type == 0 && a1 <= id && id < a2) {
                ids_out.push_back (id);
                vecs_out.insert (vecs_out.end(),
                                  vecs_in.begin() + i * d,
                                  vecs_in.begin() + (i + 1) * d);
                other.ntotal++;
            }
        }
    }
}

void IndexIVFFlat::update_vectors (int n, idx_t *new_ids, const float *x)
{
    FAISS_THROW_IF_NOT (maintain_direct_map);
    FAISS_THROW_IF_NOT (is_trained);
    std::vector<idx_t> assign (n);
    quantizer->assign (n, x, assign.data());

    for (int i = 0; i < n; i++) {
        idx_t id = new_ids[i];
        FAISS_THROW_IF_NOT_MSG (0 <= id && id < ntotal,
                                "id to update out of range");
        { // remove old one
            long dm = direct_map[id];
            long ofs = dm & 0xffffffff;
            long il = dm >> 32;
            size_t l = ids[il].size();
            if (ofs != l - 1) {
                long id2 = ids[il].back();
                ids[il][ofs] = id2;
                direct_map[id2] = (il << 32) | ofs;
                memcpy (vecs[il].data() + ofs * d,
                        vecs[il].data() + (l - 1) * d,
                        d * sizeof(vecs[il][0]));
            }
            ids[il].pop_back();
            vecs[il].resize((l - 1) * d);
        }
        { // insert new one
            long il = assign[i];
            size_t l = ids[il].size();
            long dm = (il << 32) | l;
            direct_map[id] = dm;
            ids[il].push_back (id);
            vecs[il].resize((l + 1) * d);
            memcpy (vecs[il].data() + l * d,
                    x + i * d,
                    d * sizeof(vecs[il][0]));
        }
    }

}




void IndexIVFFlat::reset()
{
    IndexIVF::reset();
    for (size_t key = 0; key < nlist; key++) {
        vecs[key].clear();
    }
}

long IndexIVFFlat::remove_ids (const IDSelector & sel)
{
    FAISS_THROW_IF_NOT_MSG (!maintain_direct_map,
                      "direct map remove not implemented");
    long nremove = 0;
#pragma omp parallel for reduction(+: nremove)
    for (long i = 0; i < nlist; i++) {
        std::vector<idx_t> & idsi = ids[i];
        float *vecsi = vecs[i].data();

        long l = idsi.size(), j = 0;
        while (j < l) {
            if (sel.is_member (idsi[j])) {
                l--;
                idsi [j] = idsi [l];
                memmove (vecsi + j * d,
                         vecsi + l * d, d * sizeof (float));
            } else {
                j++;
            }
        }
        if (l < idsi.size()) {
            nremove += idsi.size() - l;
            idsi.resize (l);
            vecs[i].resize (l * d);
        }
    }
    ntotal -= nremove;
    return nremove;
}


void IndexIVFFlat::reconstruct (idx_t key, float * recons) const
{
    FAISS_THROW_IF_NOT_MSG (direct_map.size() == ntotal,
                      "direct map is not initialized");
    int list_no = direct_map[key] >> 32;
    int ofs = direct_map[key] & 0xffffffff;
    memcpy (recons, &vecs[list_no][ofs * d], d * sizeof(recons[0]));
}




/*****************************************
 * IndexIVFFlatIPBounds implementation
 ******************************************/

IndexIVFFlatIPBounds::IndexIVFFlatIPBounds (
           Index * quantizer, size_t d, size_t nlist,
           size_t fsize):
    IndexIVFFlat(quantizer, d, nlist, METRIC_INNER_PRODUCT), fsize(fsize)
{
    part_norms.resize(nlist);
}



void IndexIVFFlatIPBounds::add_core (idx_t n, const float * x, const long *xids,
               const long *precomputed_idx) {

    FAISS_THROW_IF_NOT (is_trained);
    const long * idx;
    ScopeDeleter<long> del;

    if (precomputed_idx) {
        idx = precomputed_idx;
    } else {
        long * idx0 = new long [n];
        quantizer->assign (n, x, idx0);
        idx = idx0;
        del.set (idx);
    }
    IndexIVFFlat::add_core(n, x, xids, idx);

    // compute
    const float * xi = x + fsize;
    for (size_t i = 0; i < n; i++) {
        float norm = std::sqrt (fvec_norm_L2sqr (xi, d - fsize));
        part_norms[idx[i]].push_back(norm);
        xi += d;
    }


}

namespace {

void search_bounds_knn_inner_product (
    const IndexIVFFlatIPBounds & ivf,
    const float *x,
    const long *keys,
    float_minheap_array_t *res,
    const float *qnorms)
{

    size_t k = res->k, nx = res->nh, nprobe = ivf.nprobe;
    size_t d = ivf.d;
    int fsize = ivf.fsize;

    size_t nlistv = 0, ndis = 0, npartial = 0;

#pragma omp parallel for reduction(+: nlistv, ndis, npartial)
    for (size_t i = 0; i < nx; i++) {
        const float * xi = x + i * d;
        const long * keysi = keys + i * nprobe;
        float qnorm = qnorms[i];
        float * __restrict simi = res->get_val (i);
        long * __restrict idxi = res->get_ids (i);
        minheap_heapify (k, simi, idxi);

        for (size_t ik = 0; ik < nprobe; ik++) {
            long key = keysi[ik];  /* select the list  */
            if (key < 0) {
                // not enough centroids for multiprobe
                continue;
            }
            assert (key < (long) ivf.nlist);
            nlistv++;

            const size_t list_size = ivf.ids[key].size();
            const float * yj = ivf.vecs[key].data();
            const float * bnorms = ivf.part_norms[key].data();

            for (size_t j = 0; j < list_size; j++) {
                float ip_part = fvec_inner_product (xi, yj, fsize);
                float bound = ip_part + bnorms[j] * qnorm;

                if (bound > simi[0]) {
                    float ip = ip_part + fvec_inner_product (
                           xi + fsize, yj + fsize, d - fsize);
                    if (ip > simi[0]) {
                        minheap_pop (k, simi, idxi);
                        minheap_push (k, simi, idxi, ip, ivf.ids[key][j]);
                    }
                    ndis ++;
                }
                yj += d;
            }
            npartial += list_size;
        }
        minheap_reorder (k, simi, idxi);
    }
    indexIVFFlat_stats.nq += nx;
    indexIVFFlat_stats.nlist += nlistv;
    indexIVFFlat_stats.ndis += ndis;
    indexIVFFlat_stats.npartial += npartial;
}


}


void IndexIVFFlatIPBounds::search (
            idx_t n, const float *x, idx_t k,
            float *distances, idx_t *labels) const
{
    // compute query remainder norms and distances
    idx_t * idx = new idx_t [n * nprobe];
    ScopeDeleter<idx_t> del (idx);
    quantizer->assign (n, x, idx, nprobe);

    float * qnorms = new float [n];
    ScopeDeleter <float> del2 (qnorms);

#pragma omp parallel for
    for (size_t i = 0; i < n; i++) {
        qnorms[i] = std::sqrt (fvec_norm_L2sqr (
                x + i * d + fsize, d - fsize));
    }

    float_minheap_array_t res = {
        size_t(n), size_t(k), labels, distances};

    search_bounds_knn_inner_product (*this, x, idx, &res, qnorms);

}

} // namespace faiss
