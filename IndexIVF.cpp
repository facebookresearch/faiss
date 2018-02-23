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
 * Level1Quantizer implementation
 ******************************************/


Level1Quantizer::Level1Quantizer (Index * quantizer, size_t nlist):
    quantizer (quantizer),
    nlist (nlist),
    quantizer_trains_alone (0),
    own_fields (false),
    clustering_index (nullptr)
{
    // here we set a low # iterations because this is typically used
    // for large clusterings (nb this is not used for the MultiIndex,
    // for which quantizer_trains_alone = true)
    cp.niter = 10;
}

Level1Quantizer::Level1Quantizer ():
    quantizer (nullptr),
    nlist (0),
    quantizer_trains_alone (0), own_fields (false),
    clustering_index (nullptr)
{}

Level1Quantizer::~Level1Quantizer ()
{
    if (own_fields) delete quantizer;
}

void Level1Quantizer::train_q1 (size_t n, const float *x, bool verbose, MetricType metric_type)
{
    size_t d = quantizer->d;
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
            printf ("Training level-1 quantizer on %ld vectors in %ldD\n",
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
                "Training L2 quantizer on %ld vectors in %ldD%s\n",
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
}

/*****************************************
 * InvertedLists implementation
 ******************************************/


InvertedLists::InvertedLists (size_t nlist, size_t code_size):
    nlist (nlist), code_size (code_size)
{
}


InvertedLists::~InvertedLists ()
{}


InvertedLists::idx_t InvertedLists::get_single_id (
     size_t list_no, size_t offset) const
{
    assert (offset < list_size (list_no));
    return get_ids(list_no)[offset];
}


void InvertedLists::prefetch_lists (const long *, int) const
{}

const uint8_t * InvertedLists::get_single_code (
                   size_t list_no, size_t offset) const
{
    assert (offset < list_size (list_no));
    return get_codes(list_no) + offset * code_size;
}

size_t InvertedLists::add_entry (size_t list_no, idx_t theid,
                                 const uint8_t *code)
{
    return add_entries (list_no, 1, &theid, code);
}

void InvertedLists::update_entry (size_t list_no, size_t offset,
                                        idx_t id, const uint8_t *code)
{
    update_entries (list_no, offset, 1, &id, code);
}

void InvertedLists::reset () {
    for (size_t i = 0; i < nlist; i++) {
        resize (i, 0);
    }
}

/*****************************************
 * ArrayInvertedLists implementation
 ******************************************/

ArrayInvertedLists::ArrayInvertedLists (size_t nlist, size_t code_size):
    InvertedLists (nlist, code_size)
{
    ids.resize (nlist);
    codes.resize (nlist);
}

size_t ArrayInvertedLists::add_entries (
           size_t list_no, size_t n_entry,
           const idx_t* ids_in, const uint8_t *code)
{
    if (n_entry == 0) return 0;
    assert (list_no < nlist);
    size_t o = ids [list_no].size();
    ids [list_no].resize (o + n_entry);
    memcpy (&ids[list_no][o], ids_in, sizeof (ids_in[0]) * n_entry);
    codes [list_no].resize ((o + n_entry) * code_size);
    memcpy (&codes[list_no][o * code_size], code, code_size * n_entry);
    return o;
}

size_t ArrayInvertedLists::list_size(size_t list_no) const
{
    assert (list_no < nlist);
    return ids[list_no].size();
}

const uint8_t * ArrayInvertedLists::get_codes (size_t list_no) const
{
    assert (list_no < nlist);
    return codes[list_no].data();
}

const InvertedLists::idx_t * ArrayInvertedLists::get_ids (size_t list_no) const
{
    assert (list_no < nlist);
    return ids[list_no].data();
}

void ArrayInvertedLists::resize (size_t list_no, size_t new_size)
{
    ids[list_no].resize (new_size);
    codes[list_no].resize (new_size * code_size);
}

void ArrayInvertedLists::update_entries (
      size_t list_no, size_t offset, size_t n_entry,
      const idx_t *ids_in, const uint8_t *codes_in)
{
    assert (list_no < nlist);
    assert (n_entry + offset <= ids[list_no].size());
    memcpy (&ids[list_no][offset], ids_in, sizeof(ids_in[0]) * n_entry);
    memcpy (&codes[list_no][offset * code_size], codes_in, code_size * n_entry);
}


ArrayInvertedLists::~ArrayInvertedLists ()
{}



/*****************************************
 * IndexIVF implementation
 ******************************************/


IndexIVF::IndexIVF (Index * quantizer, size_t d,
                    size_t nlist, size_t code_size,
                    MetricType metric):
    Index (d, metric),
    Level1Quantizer (quantizer, nlist),
    invlists (new ArrayInvertedLists (nlist, code_size)),
    own_invlists (true),
    code_size (code_size),
    nprobe (1),
    max_codes (0),
    maintain_direct_map (false)
{
    FAISS_THROW_IF_NOT (d == quantizer->d);
    is_trained = quantizer->is_trained && (quantizer->ntotal == nlist);
    // Spherical by default if the metric is inner_product
    if (metric_type == METRIC_INNER_PRODUCT) {
        cp.spherical = true;
    }

}

IndexIVF::IndexIVF ():
    invlists (nullptr), own_invlists (false),
    code_size (0),
    nprobe (1), max_codes (0),
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
            size_t list_size = invlists->list_size (key);
            const idx_t *idlist = invlists->get_ids (key);

            for (long ofs = 0; ofs < list_size; ofs++) {
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

    invlists->prefetch_lists (idx, n * nprobe);

    search_preassigned (n, x, k, idx, coarse_dis,
                        distances, labels, false);

}


void IndexIVF::reconstruct (idx_t key, float* recons) const
{
    FAISS_THROW_IF_NOT_MSG (direct_map.size() == ntotal,
                            "direct map is not initialized");
    long list_no = direct_map[key] >> 32;
    long offset = direct_map[key] & 0xffffffff;
    reconstruct_from_offset (list_no, offset, recons);
}


void IndexIVF::reconstruct_n (idx_t i0, idx_t ni, float* recons) const
{
    FAISS_THROW_IF_NOT (ni == 0 || (i0 >= 0 && i0 + ni <= ntotal));

    for (long list_no = 0; list_no < nlist; list_no++) {
        size_t list_size = invlists->list_size (list_no);
        const Index::idx_t * idlist = invlists->get_ids (list_no);

        for (long offset = 0; offset < list_size; offset++) {
            long id = idlist[offset];
            if (!(id >= i0 && id < i0 + ni)) {
              continue;
            }

            float* reconstructed = recons + (id - i0) * d;
            reconstruct_from_offset (list_no, offset, reconstructed);
        }
    }
}


void IndexIVF::search_and_reconstruct (idx_t n, const float *x, idx_t k,
                                       float *distances, idx_t *labels,
                                       float *recons) const
{
    long * idx = new long [n * nprobe];
    ScopeDeleter<long> del (idx);
    float * coarse_dis = new float [n * nprobe];
    ScopeDeleter<float> del2 (coarse_dis);

    quantizer->search (n, x, nprobe, coarse_dis, idx);

    invlists->prefetch_lists (idx, n * nprobe);

    // search_preassigned() with `store_pairs` enabled to obtain the list_no
    // and offset into `codes` for reconstruction
    search_preassigned (n, x, k, idx, coarse_dis,
                        distances, labels, true /* store_pairs */);
    for (idx_t i = 0; i < n; ++i) {
      for (idx_t j = 0; j < k; ++j) {
        idx_t ij = i * k + j;
        idx_t key = labels[ij];
        float* reconstructed = recons + ij * d;
        if (key < 0) {
          // Fill with NaNs
          memset(reconstructed, -1, sizeof(*reconstructed) * d);
        } else {
          int list_no = key >> 32;
          int offset = key & 0xffffffff;

          // Update label to the actual id
          labels[ij] = invlists->get_single_id (list_no, offset);

          reconstruct_from_offset (list_no, offset, reconstructed);
        }
      }
    }
}

void IndexIVF::reconstruct_from_offset (long list_no, long offset,
                                        float* recons) const
{
  FAISS_THROW_MSG ("reconstruct_from_offset not implemented");
}

void IndexIVF::reset ()
{
    direct_map.clear ();
    invlists->reset ();
    ntotal = 0;
}


long IndexIVF::remove_ids (const IDSelector & sel)
{
    FAISS_THROW_IF_NOT_MSG (!maintain_direct_map,
                    "direct map remove not implemented");

    std::vector<long> toremove(nlist);

#pragma omp parallel for
    for (long i = 0; i < nlist; i++) {
        long l0 = invlists->list_size (i), l = l0, j = 0;
        const idx_t *idsi = invlists->get_ids (i);
        while (j < l) {
            if (sel.is_member (idsi[j])) {
                l--;
                invlists->update_entry (
                     i, j,
                     invlists->get_single_id (i, l),
                     invlists->get_single_code (i, l));
            } else {
                j++;
            }
        }
        toremove[i] = l0 - l;
    }
    // this will not run well in parallel on ondisk because of possible shrinks
    long nremove = 0;
    for (long i = 0; i < nlist; i++) {
        if (toremove[i] > 0) {
            nremove += toremove[i];
            invlists->resize(
                i, invlists->list_size(i) - toremove[i]);
        }
    }
    ntotal -= nremove;
    return nremove;
}




void IndexIVF::train (idx_t n, const float *x)
{
    if (verbose)
        printf ("Training level-1 quantizer\n");

    train_q1 (n, x, verbose, metric_type);

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
        hist[i] = invlists->list_size(i);
    }
    return faiss::imbalance_factor (nlist, hist.data());
}

void IndexIVF::print_stats () const
{
    std::vector<int> sizes(40);
    for (int i = 0; i < nlist; i++) {
        for (int j = 0; j < sizes.size(); j++) {
            if ((invlists->list_size(i) >> j) == 0) {
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
    FAISS_THROW_IF_NOT (other.code_size == code_size);
    FAISS_THROW_IF_NOT_MSG ((!maintain_direct_map &&
                             !other.maintain_direct_map),
                  "direct map copy not implemented");
    FAISS_THROW_IF_NOT_MSG (typeid (*this) == typeid (other),
                  "can only merge indexes of the same type");

    InvertedLists *oivf = other.invlists;
#pragma omp parallel for
    for (long i = 0; i < nlist; i++) {
        size_t list_size = oivf->list_size (i);
        const idx_t * ids = oivf->get_ids (i);
        if (add_id == 0) {
            invlists->add_entries (i, list_size, ids,
                                   oivf->get_codes (i));
        } else {
            std::vector <idx_t> new_ids (list_size);

            for (size_t j = 0; j < list_size; j++) {
                new_ids [j] = ids[j] + add_id;
            }

            invlists->add_entries (i, list_size, new_ids.data(),
                                   oivf->get_codes (i));
        }
        oivf->resize (i, 0);
    }

    ntotal += other.ntotal;
    other.ntotal = 0;
}


void IndexIVF::replace_invlists (InvertedLists *il, bool own)
{
    //FAISS_THROW_IF_NOT (ntotal == 0);
    FAISS_THROW_IF_NOT (il->nlist == nlist &&
                        il->code_size == code_size);
    if (own_invlists) {
        delete invlists;
    }
    invlists = il;
    own_invlists = own;
}


void IndexIVF::copy_subset_to (IndexIVF & other, int subset_type,
                                 long a1, long a2) const
{

    FAISS_THROW_IF_NOT (nlist == other.nlist);
    FAISS_THROW_IF_NOT (code_size == other.code_size);
    FAISS_THROW_IF_NOT (!other.maintain_direct_map);
    FAISS_THROW_IF_NOT_FMT (
          subset_type == 0 || subset_type == 1 || subset_type == 2,
          "subset type %d not implemented", subset_type);

    size_t accu_n = 0;
    size_t accu_a1 = 0;
    size_t accu_a2 = 0;

    InvertedLists *oivf = other.invlists;

    for (long list_no = 0; list_no < nlist; list_no++) {
        size_t n = invlists->list_size (list_no);
        const idx_t *ids_in = invlists->get_ids (list_no);

        if (subset_type == 0) {
            for (long i = 0; i < n; i++) {
                idx_t id = ids_in[i];
                if (a1 <= id && id < a2) {
                    oivf->add_entry (list_no,
                                     invlists->get_single_id (list_no, i),
                                     invlists->get_single_code (list_no, i));
                    other.ntotal++;
                }
            }
        } else if (subset_type == 1) {
            for (long i = 0; i < n; i++) {
                idx_t id = ids_in[i];
                if (id % a1 == a2) {
                    oivf->add_entry (list_no,
                                     invlists->get_single_id (list_no, i),
                                     invlists->get_single_code (list_no, i));
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

            for (long i = i1; i < i2; i++) {
                oivf->add_entry (list_no,
                                 invlists->get_single_id (list_no, i),
                                 invlists->get_single_code (list_no, i));
            }

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
    if (own_invlists) {
        delete invlists;
    }
}


void IndexIVFStats::reset()
{
    memset ((void*)this, 0, sizeof (*this));
}


IndexIVFStats indexIVF_stats;



} // namespace faiss
