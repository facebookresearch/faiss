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
    quantizer_trains_alone (0),
    own_fields (false),
    clustering_index (nullptr),
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
    code_size = 0; // let sub-classes set this
    codes.resize(nlist);
}

IndexIVF::IndexIVF ():
    nlist (0), nprobe (1), quantizer (nullptr),
    quantizer_trains_alone (0), own_fields (false),
    clustering_index (nullptr),
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


void IndexIVF::search (idx_t n, const float *x, idx_t k,
                         float *distances, idx_t *labels) const
{
    long * idx = new long [n * nprobe];
    ScopeDeleter<long> del (idx);
    float * coarse_dis = new float [n * nprobe];
    ScopeDeleter<float> del2 (coarse_dis);

    quantizer->search (n, x, nprobe, coarse_dis, idx);

    search_preassigned (n, x, k, idx, coarse_dis,
                        distances, labels, false);

}


void IndexIVF::reset ()
{
    ntotal = 0;
    direct_map.clear();
    for (size_t i = 0; i < ids.size(); i++) {
        ids[i].clear();
        codes[i].clear();
    }
}


long IndexIVF::remove_ids (const IDSelector & sel)
{
    FAISS_THROW_IF_NOT_MSG (!maintain_direct_map,
                    "direct map remove not implemented");
    long nremove = 0;
#pragma omp parallel for reduction(+: nremove)
    for (long i = 0; i < nlist; i++) {
        std::vector<idx_t> & idsi = ids[i];
        uint8_t * codesi = codes[i].data();

        long l = idsi.size(), j = 0;
        while (j < l) {
            if (sel.is_member (idsi[j])) {
                l--;
                idsi [j] = idsi [l];
                memmove (codesi + j * code_size,
                         codesi + l * code_size, code_size);
            } else {
                j++;
            }
        }
        if (l < idsi.size()) {
            nremove += idsi.size() - l;
            idsi.resize (l);
            codes[i].resize (l * code_size);
        }
    }
    ntotal -= nremove;
    return nremove;
}




void IndexIVF::train (idx_t n, const float *x)
{
    if (quantizer->is_trained && (quantizer->ntotal == nlist)) {
        if (verbose)
            printf ("IVF quantizer does not need training.\n");
    } else if (quantizer_trains_alone == 1) {
        if (verbose)
            printf ("IVF quantizer trains alone...\n");
        quantizer->train (n, x);
        quantizer->verbose = verbose;
        FAISS_THROW_IF_NOT_MSG (quantizer->ntotal == nlist,
                          "nlist not consistent with quantizer size");
    } else if (quantizer_trains_alone == 0) {
        if (verbose)
            printf ("Training IVF quantizer on %ld vectors in %dD\n",
                    n, d);

        Clustering clus (d, nlist, cp);
        quantizer->reset();
        if (clustering_index) {
            clus.train (n, x, *clustering_index);
            quantizer->add (nlist, clus.centroids.data());
        } else {
            clus.train (n, x, *quantizer);
        }
        quantizer->is_trained = true;
    } else if (quantizer_trains_alone == 2) {
        if (verbose)
            printf (
                "Training L2 quantizer on %ld vectors in %dD%s\n",
                n, d,
                clustering_index ? "(user provided index)" : "");
        FAISS_THROW_IF_NOT (metric_type == METRIC_L2);
        Clustering clus (d, nlist, cp);
        if (!clustering_index) {
            IndexFlatL2 assigner (d);
            clus.train(n, x, assigner);
        } else {
            clus.train(n, x, *clustering_index);
        }
        if (verbose)
            printf ("Adding centroids to quantizer\n");
        quantizer->add (nlist, clus.centroids.data());
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
        codes[i].insert (codes[i].end(),
                         other.codes[i].begin(),
                         other.codes[i].end());
        other.codes[i].clear();
    }

    ntotal += other.ntotal;
    other.ntotal = 0;
}


void IndexIVF::copy_subset_to (IndexIVF & other, int subset_type,
                                 long a1, long a2) const
{
    FAISS_THROW_IF_NOT (nlist == other.nlist);
    FAISS_THROW_IF_NOT (!other.maintain_direct_map);
    FAISS_THROW_IF_NOT_FMT (
          subset_type == 0 || subset_type == 1 || subset_type == 2,
          "subset type %d not implemented", subset_type);

    size_t accu_n = 0;
    size_t accu_a1 = 0;
    size_t accu_a2 = 0;

    for (long list_no = 0; list_no < nlist; list_no++) {
        const std::vector<idx_t> & ids_in = ids[list_no];
        std::vector<idx_t> & ids_out = other.ids[list_no];
        const std::vector<uint8_t> & codes_in = codes[list_no];
        std::vector<uint8_t> & codes_out = other.codes[list_no];
        size_t n = ids_in.size();

        if (subset_type == 0) {
            for (long i = 0; i < n; i++) {
                idx_t id = ids_in[i];
                if (a1 <= id && id < a2) {
                    ids_out.push_back (id);
                    codes_out.insert (codes_out.end(),
                                      codes_in.begin() + i * code_size,
                                  codes_in.begin() + (i + 1) * code_size);
                    other.ntotal++;
                }
            }
        } else if (subset_type == 1) {
            for (long i = 0; i < n; i++) {
                idx_t id = ids_in[i];
                if (id % a1 == a2) {
                    ids_out.push_back (id);
                    codes_out.insert (codes_out.end(),
                                      codes_in.begin() + i * code_size,
                                  codes_in.begin() + (i + 1) * code_size);
                    other.ntotal++;
                }
            }
        } else if (subset_type == 2) {
            // see what is allocated to a1 and to a2
            size_t next_accu_n = accu_n + n;
            size_t next_accu_a1 = next_accu_n * a1 / ntotal;
            size_t i1 = next_accu_a1 - accu_a1;
            size_t next_accu_a2 = next_accu_n * a2 / ntotal;
            size_t i2 = next_accu_a2 - accu_a2;
            ids_out.insert(ids_out.end(),
                           ids_in.begin() + i1,
                           ids_in.begin() + i2);
            codes_out.insert (codes_out.end(),
                              codes_in.begin() + i1 * code_size,
                              codes_in.begin() + i2 * code_size);
            other.ntotal += i2 - i1;
            accu_a1 = next_accu_a1;
            accu_a2 = next_accu_a2;
        }
        accu_n += n;
    }
    FAISS_ASSERT(accu_n == ntotal);
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
        size_t ofs = codes[list_no].size();
        codes[list_no].resize(ofs + code_size);
        memcpy(codes[list_no].data() + ofs,
               xi, code_size);

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
            const size_t list_size = ivf.ids[key].size();
            const float * list_vecs = (const float*)(ivf.codes[key].data());

            for (size_t j = 0; j < list_size; j++) {
                const float * yj = list_vecs + d * j;
                float ip = fvec_inner_product (xi, yj, d);
                if (ip > simi[0]) {
                    minheap_pop (k, simi, idxi);
                    long id = store_pairs ? (key << 32 | j) : ivf.ids[key][j];
                    minheap_push (k, simi, idxi, ip, id);
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
            const size_t list_size = ivf.ids[key].size();
            const float * list_vecs = (const float*)(ivf.codes[key].data());

            for (size_t j = 0; j < list_size; j++) {
                const float * yj = list_vecs + d * j;
                float disij = fvec_L2sqr (xi, yj, d);
                if (disij < disi[0]) {
                    maxheap_pop (k, disi, idxi);
                    long id = store_pairs ? (key << 32 | j) : ivf.ids[key][j];
                    maxheap_push (k, disi, idxi, disij, id);
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

                const size_t list_size = ids[key].size();
                const float * list_vecs = (const float *)(codes[key].data());

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
                float * vecs = (float*)codes[il].data();
                memcpy (vecs + ofs * d,
                        vecs + (l - 1) * d,
                        d * sizeof(float));
            }
            ids[il].pop_back();
            codes[il].resize((l - 1) * code_size);
        }
        { // insert new one
            long il = assign[i];
            size_t l = ids[il].size();
            long dm = (il << 32) | l;
            direct_map[id] = dm;
            ids[il].push_back (id);
            codes[il].resize((l + 1) * code_size);
            float * vecs = (float*)codes[il].data();
            memcpy (vecs + l * d,
                    x + i * d,
                    d * sizeof(float));
        }
    }

}





void IndexIVFFlat::reconstruct (idx_t key, float * recons) const
{
    FAISS_THROW_IF_NOT_MSG (direct_map.size() == ntotal,
                      "direct map is not initialized");
    int list_no = direct_map[key] >> 32;
    int ofs = direct_map[key] & 0xffffffff;
    memcpy (recons, &codes[list_no][ofs * code_size], d * sizeof(recons[0]));
}




} // namespace faiss
