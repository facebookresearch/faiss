
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
    FAISS_ASSERT (d == quantizer->d);
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

void IndexIVF::make_direct_map ()
{
    if (maintain_direct_map) return;

    direct_map.resize (ntotal, -1);
    for (size_t key = 0; key < nlist; key++) {
        const std::vector<long> & idlist = ids[key];

        for (long ofs = 0; ofs < idlist.size(); ofs++) {
            direct_map [idlist [ofs]] =
                key << 32 | ofs;
        }
    }

    maintain_direct_map = true;
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
        FAISS_ASSERT (quantizer->ntotal == nlist ||
                      !"nlist not consistent with quantizer size");
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

void IndexIVF::train_residual (idx_t n, const float *x)
{
    if (verbose)
        printf ("IndexIVF: no residual training\n");
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
    FAISS_ASSERT (other.d == d);
    FAISS_ASSERT (other.nlist == nlist);
    FAISS_ASSERT ((!maintain_direct_map && !other.maintain_direct_map) ||
                  !"direct map copy not implemented");
    FAISS_ASSERT (typeid (*this) == typeid (other) ||
                  !"can only merge indexes of the same type");
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
    set_typename();
}


void IndexIVFFlat::set_typename ()
{
    std::stringstream s;
    if (metric_type == METRIC_INNER_PRODUCT)
        s << "IvfIP";
    else if (metric_type == METRIC_L2)
        s << "IvfL2";
    else s << "??";
    s << "[" << nlist << ":" << quantizer->index_typename << "]";
    index_typename = s.str();
}






void IndexIVFFlat::add_with_ids (idx_t n, const float * x, const long *xids)
{
    add_core (n, x, xids, nullptr);
}

void IndexIVFFlat::add_core (idx_t n, const float * x, const long *xids,
                             const long *precomputed_idx)

{
    FAISS_ASSERT (is_trained);
    const long * idx;

    if (precomputed_idx) {
        idx = precomputed_idx;
    } else {
        long * idx0 = new long [n];
        quantizer->assign (n, x, idx0);
        idx = idx0;
    }
    long n_add = 0;
    for (size_t i = 0; i < n; i++) {
        long id = xids ? xids[i] : ntotal + i;
        long list_no = idx [i];
        if (list_no < 0)
            continue;
        FAISS_ASSERT (list_no < nlist);

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
    if (!precomputed_idx)
        delete [] idx;
    ntotal += n_add;
}




void IndexIVFFlat::search_knn_inner_product (
    size_t nx,
    const float * x,
    const long * __restrict keys,
    float_minheap_array_t * res) const
{

    const size_t k = res->k;

#pragma omp parallel for
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
        }
        minheap_reorder (k, simi, idxi);
    }
}


void IndexIVFFlat::search_knn_L2sqr (
    size_t nx,
    const float * x,
    const long * __restrict keys,
    float_maxheap_array_t * res) const
{
    const size_t k = res->k;

#pragma omp parallel for
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
        }
        maxheap_reorder (k, disi, idxi);
    }
}


void IndexIVFFlat::search (idx_t n, const float *x, idx_t k,
                                float *distances, idx_t *labels) const
{
    idx_t * idx = new idx_t [n * nprobe];
    quantizer->assign (n, x, idx, nprobe);

   if (metric_type == METRIC_INNER_PRODUCT) {
        float_minheap_array_t res = {
            size_t(n), size_t(k), labels, distances};
        search_knn_inner_product (n, x, idx, &res);

    } else if (metric_type == METRIC_L2) {
        float_maxheap_array_t res = {
            size_t(n), size_t(k), labels, distances};
        search_knn_L2sqr (n, x, idx, &res);
    }

    delete [] idx;
}


void IndexIVFFlat::range_search (idx_t nx, const float *x, float radius,
                                 RangeSearchResult *result) const
{
    idx_t * keys = new idx_t [nx * nprobe];
    quantizer->assign (nx, x, keys, nprobe);

    assert (metric_type == METRIC_L2 || !"Only L2 implemented");
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
                    float disij = fvec_L2sqr (xi, yj, d);
                    if (disij < radius) {
                        qres.add (disij, ids[key][j]);
                    }
                }
            }
        }

        pres.finalize ();
    }
    delete[] keys;
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
    FAISS_ASSERT (nlist == other.nlist);
    FAISS_ASSERT (!other.maintain_direct_map);

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



void IndexIVFFlat::reset()
{
    IndexIVF::reset();
    for (size_t key = 0; key < nlist; key++) {
        vecs[key].clear();
    }
}

long IndexIVFFlat::remove_ids (const IDSelector & sel)
{
    FAISS_ASSERT (!maintain_direct_map ||
                  !"direct map remove not implemented");
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
    assert (direct_map.size() == ntotal);
    int list_no = direct_map[key] >> 32;
    int ofs = direct_map[key] & 0xffffffff;
    memcpy (recons, &vecs[list_no][ofs * d], d * sizeof(recons[0]));
}


} // namespace faiss
