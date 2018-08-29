/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved
// -*- c++ -*-

#include "IndexBinaryIVF.h"

#include <cstdio>
#include <memory>

#include "hamming.h"
#include "utils.h"

#include "AuxIndexStructures.h"
#include "FaissAssert.h"
#include "IndexFlat.h"


namespace faiss {

IndexBinaryIVF::IndexBinaryIVF(IndexBinary *quantizer, size_t d, size_t nlist)
    : IndexBinary(d),
      invlists(new ArrayInvertedLists(nlist, code_size)),
      own_invlists(true),
      nprobe(1),
      max_codes(0),
      maintain_direct_map(false),
      quantizer(quantizer),
      nlist(nlist),
      own_fields(false) {
  FAISS_THROW_IF_NOT (d == quantizer->d);
  is_trained = quantizer->is_trained && (quantizer->ntotal == nlist);

  cp.niter = 10;
}

IndexBinaryIVF::IndexBinaryIVF()
    : invlists(nullptr),
      own_invlists(false),
      nprobe(1),
      max_codes(0),
      maintain_direct_map(false),
      quantizer(nullptr),
      nlist(0),
      own_fields(false) {}

void IndexBinaryIVF::add(idx_t n, const uint8_t *x) {
  add_with_ids(n, x, nullptr);
}

void IndexBinaryIVF::add_with_ids(idx_t n, const uint8_t *x, const long *xids) {
  add_core(n, x, xids, nullptr);
}

void IndexBinaryIVF::add_core(idx_t n, const uint8_t *x, const long *xids,
                              const long *precomputed_idx) {
  FAISS_THROW_IF_NOT(is_trained);
  assert(invlists);
  FAISS_THROW_IF_NOT_MSG(!(maintain_direct_map && xids),
                         "cannot have direct map and add with ids");

  const long * idx;

  std::unique_ptr<long[]> scoped_idx;

  if (precomputed_idx) {
    idx = precomputed_idx;
  } else {
    scoped_idx.reset(new long[n]);
    quantizer->assign(n, x, scoped_idx.get());
    idx = scoped_idx.get();
  }

  long n_add = 0;
  for (size_t i = 0; i < n; i++) {
    long id = xids ? xids[i] : ntotal + i;
    long list_no = idx[i];

    if (list_no < 0)
      continue;
    const uint8_t *xi = x + i * code_size;
    size_t offset = invlists->add_entry(list_no, id, xi);

    if (maintain_direct_map)
      direct_map.push_back(list_no << 32 | offset);
    n_add++;
  }
  if (verbose) {
    printf("IndexBinaryIVF::add_with_ids: added %ld / %ld vectors\n",
           n_add, n);
  }
  ntotal += n_add;
}

void IndexBinaryIVF::make_direct_map(bool new_maintain_direct_map) {
  // nothing to do
  if (new_maintain_direct_map == maintain_direct_map)
    return;

  if (new_maintain_direct_map) {
    direct_map.resize(ntotal, -1);
    for (size_t key = 0; key < nlist; key++) {
      size_t list_size = invlists->list_size(key);
      const idx_t *idlist = invlists->get_ids(key);

      for (long ofs = 0; ofs < list_size; ofs++) {
        FAISS_THROW_IF_NOT_MSG(0 <= idlist[ofs] && idlist[ofs] < ntotal,
                               "direct map supported only for seuquential ids");
        direct_map[idlist[ofs]] = key << 32 | ofs;
      }
    }
  } else {
    direct_map.clear();
  }
  maintain_direct_map = new_maintain_direct_map;
}

void IndexBinaryIVF::search(idx_t n, const uint8_t *x, idx_t k,
                            int32_t *distances, idx_t *labels) const {
  std::unique_ptr<idx_t[]> idx(new idx_t[n * nprobe]);
  std::unique_ptr<int32_t[]> coarse_dis(new int32_t[n * nprobe]);

  quantizer->search(n, x, nprobe, coarse_dis.get(), idx.get());

  invlists->prefetch_lists(idx.get(), n * nprobe);

  search_preassigned(n, x, k, idx.get(), coarse_dis.get(),
                     distances, labels, false);
}

void IndexBinaryIVF::reconstruct(idx_t key, uint8_t *recons) const {
  FAISS_THROW_IF_NOT_MSG(direct_map.size() == ntotal,
                         "direct map is not initialized");
  long list_no = direct_map[key] >> 32;
  long offset = direct_map[key] & 0xffffffff;
  reconstruct_from_offset(list_no, offset, recons);
}

void IndexBinaryIVF::reconstruct_n(idx_t i0, idx_t ni, uint8_t *recons) const {
  FAISS_THROW_IF_NOT(ni == 0 || (i0 >= 0 && i0 + ni <= ntotal));

  for (long list_no = 0; list_no < nlist; list_no++) {
    size_t list_size = invlists->list_size(list_no);
    const Index::idx_t *idlist = invlists->get_ids(list_no);

    for (long offset = 0; offset < list_size; offset++) {
      long id = idlist[offset];
      if (!(id >= i0 && id < i0 + ni)) {
        continue;
      }

      uint8_t *reconstructed = recons + (id - i0) * d;
      reconstruct_from_offset(list_no, offset, reconstructed);
    }
  }
}

void IndexBinaryIVF::search_and_reconstruct(idx_t n, const uint8_t *x, idx_t k,
                                            int32_t *distances, idx_t *labels,
                                            uint8_t *recons) const {
  std::unique_ptr<idx_t[]> idx(new long[n * nprobe]);
  std::unique_ptr<int32_t[]> coarse_dis(new int32_t[n * nprobe]);

  quantizer->search(n, x, nprobe, coarse_dis.get(), idx.get());

  invlists->prefetch_lists(idx.get(), n * nprobe);

  // search_preassigned() with `store_pairs` enabled to obtain the list_no
  // and offset into `codes` for reconstruction
  search_preassigned(n, x, k, idx.get(), coarse_dis.get(),
                     distances, labels, /* store_pairs */true);
  for (idx_t i = 0; i < n; ++i) {
    for (idx_t j = 0; j < k; ++j) {
      idx_t ij = i * k + j;
      idx_t key = labels[ij];
      uint8_t *reconstructed = recons + ij * d;
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

void IndexBinaryIVF::reconstruct_from_offset(long list_no, long offset,
                                             uint8_t *recons) const {
  memcpy(recons, invlists->get_single_code(list_no, offset), code_size);
}

void IndexBinaryIVF::reset() {
  direct_map.clear();
  invlists->reset();
  ntotal = 0;
}

long IndexBinaryIVF::remove_ids(const IDSelector& sel) {
  FAISS_THROW_IF_NOT_MSG(!maintain_direct_map,
                         "direct map remove not implemented");

  std::vector<long> toremove(nlist);

#pragma omp parallel for
  for (long i = 0; i < nlist; i++) {
    long l0 = invlists->list_size (i), l = l0, j = 0;
    const idx_t *idsi = invlists->get_ids(i);
    while (j < l) {
      if (sel.is_member(idsi[j])) {
        l--;
        invlists->update_entry(
          i, j,
          invlists->get_single_id(i, l),
          invlists->get_single_code(i, l));
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

void IndexBinaryIVF::train(idx_t n, const uint8_t *x) {
  if (verbose)
    printf("Training level-1 quantizer\n");

  train_q1(n, x, verbose);

  is_trained = true;
}

double IndexBinaryIVF::imbalance_factor () const {
  std::vector<int> hist(nlist);

  for (int i = 0; i < nlist; i++) {
    hist[i] = invlists->list_size(i);
  }

  return faiss::imbalance_factor(nlist, hist.data());
}

void IndexBinaryIVF::print_stats() const {
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
      printf("list size in < %d: %d instances\n", 1 << i, sizes[i]);
    }
  }
}

void IndexBinaryIVF::merge_from(IndexBinaryIVF &other, idx_t add_id) {
  // minimal sanity checks
  FAISS_THROW_IF_NOT(other.d == d);
  FAISS_THROW_IF_NOT(other.nlist == nlist);
  FAISS_THROW_IF_NOT(other.code_size == code_size);
  FAISS_THROW_IF_NOT_MSG((!maintain_direct_map &&
                          !other.maintain_direct_map),
                         "direct map copy not implemented");
  FAISS_THROW_IF_NOT_MSG(typeid (*this) == typeid (other),
                         "can only merge indexes of the same type");

  invlists->merge_from (other.invlists, add_id);

  ntotal += other.ntotal;
  other.ntotal = 0;
}

void IndexBinaryIVF::replace_invlists(InvertedLists *il, bool own) {
  FAISS_THROW_IF_NOT(il->nlist == nlist &&
                     il->code_size == code_size);
  if (own_invlists) {
    delete invlists;
  }
  invlists = il;
  own_invlists = own;
}


void IndexBinaryIVF::train_q1(size_t n, const uint8_t *x, bool verbose) {
  if (quantizer->is_trained && (quantizer->ntotal == nlist)) {
    if (verbose)
      printf("IVF quantizer does not need training.\n");
  } else {
    if (verbose)
      printf("Training level-1 quantizer on %ld vectors in %dD\n", n, d);

    Clustering clus(d, nlist, cp);
    quantizer->reset();

    std::unique_ptr<float[]> x_f(new float[n * d]);
    binary_to_real(n * d, x, x_f.get());

    IndexFlatL2 index_tmp(d);

    clus.train(n, x_f.get(), index_tmp);

    std::unique_ptr<uint8_t[]> x_b(new uint8_t[clus.k * code_size]);
    real_to_binary(d * clus.k, clus.centroids.data(), x_b.get());

    quantizer->add(clus.k, x_b.get());
    quantizer->is_trained = true;
  }
}


namespace {


template<class HammingComputer, bool store_pairs>
void search_knn_hamming_heap(const IndexBinaryIVF& ivf,
                             size_t nx,
                             const uint8_t *x,
                             const long *keys,
                             int_maxheap_array_t *res,
                             const IVFSearchParameters *params) {
  const size_t k = res->k;
  size_t nlistv = 0, ndis = 0;
  long nprobe = params ? params->nprobe : ivf.nprobe;
  long max_codes = params ? params->max_codes : ivf.max_codes;

#pragma omp parallel for reduction(+: nlistv, ndis)
  for (size_t i = 0; i < nx; i++) {
    const uint8_t *xi = x + i * ivf.code_size;
    const long * keysi = keys + i * nprobe;
    int32_t * __restrict disi = res->get_val(i);
    long * __restrict idxi = res->get_ids(i);
    maxheap_heapify(k, disi, idxi);

    size_t nscan = 0;

    for (size_t ik = 0; ik < nprobe; ik++) {
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

      InvertedLists::ScopedCodes scodes (ivf.invlists, key);
      const uint8_t *list_vecs = scodes.get();
      const Index::idx_t *ids = store_pairs
        ? nullptr
        : ivf.invlists->get_ids(key);
      HammingComputer hc(xi, ivf.code_size);

      for (size_t j = 0; j < list_size; j++) {
        const uint8_t * yj = list_vecs + ivf.code_size * j;

        int32_t disij = hc.hamming(yj);

        if (disij < disi[0]) {
          maxheap_pop(k, disi, idxi);
          long id = store_pairs ? (key << 32 | j) : ids[j];
          maxheap_push(k, disi, idxi, disij, id);
        }
      }
      if (ids)
        ivf.invlists->release_ids (ids);

      nscan += list_size;
      if (max_codes && nscan >= max_codes)
        break;
    }
    ndis += nscan;
    maxheap_reorder(k, disi, idxi);
  }
  indexIVF_stats.nq += nx;
  indexIVF_stats.nlist += nlistv;
  indexIVF_stats.ndis += ndis;
}

template<class HammingComputer, bool store_pairs>
void search_knn_hamming_count(const IndexBinaryIVF& ivf,
                              size_t nx,
                              const uint8_t *x,
                              const long *keys,
                              int k,
                              int32_t *distances,
                              long *labels,
                              const IVFSearchParameters *params) {
  const int nBuckets = ivf.d + 1;
  std::vector<int> all_counters(nx * nBuckets, 0);
  std::unique_ptr<long[]> all_ids_per_dis(new long[nx * nBuckets * k]);

  long nprobe = params ? params->nprobe : ivf.nprobe;
  long max_codes = params ? params->max_codes : ivf.max_codes;

  std::vector<HCounterState<HammingComputer>> cs;
  for (size_t i = 0; i < nx; ++i) {
    cs.push_back(HCounterState<HammingComputer>(
                   all_counters.data() + i * nBuckets,
                   all_ids_per_dis.get() + i * nBuckets * k,
                   x + i * ivf.code_size,
                   ivf.d,
                   k
                 ));
  }

  size_t nlistv = 0, ndis = 0;

#pragma omp parallel for reduction(+: nlistv, ndis)
  for (size_t i = 0; i < nx; i++) {
    const long * keysi = keys + i * nprobe;
    HCounterState<HammingComputer>& csi = cs[i];

    size_t nscan = 0;

    for (size_t ik = 0; ik < nprobe; ik++) {
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
      InvertedLists::ScopedCodes scodes (ivf.invlists, key);
      const uint8_t *list_vecs = scodes.get();
      const Index::idx_t *ids = store_pairs
        ? nullptr
        : ivf.invlists->get_ids(key);

      for (size_t j = 0; j < list_size; j++) {
        const uint8_t * yj = list_vecs + ivf.code_size * j;

        long id = store_pairs ? (key << 32 | j) : ids[j];
        csi.update_counter(yj, id);
      }
      if (ids)
        ivf.invlists->release_ids (ids);

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
  }
  indexIVF_stats.nq += nx;
  indexIVF_stats.nlist += nlistv;
  indexIVF_stats.ndis += ndis;
}


template<class HammingComputer>
void search_knn_hamming(const IndexBinaryIVF& ivf,
                        size_t nx,
                        const uint8_t *x,
                        const long *keys,
                        int k,
                        int32_t *distances,
                        long *labels,
                        bool store_pairs,
                        bool use_heap,
                        const IVFSearchParameters *params) {
  if (use_heap) {
    int_maxheap_array_t res = {
      size_t(nx), size_t(k), labels, distances};
    if (store_pairs) {
        search_knn_hamming_heap<HammingComputer, true>(
            ivf, nx, x, keys, &res, params);
    } else {
        search_knn_hamming_heap<HammingComputer, false>(
            ivf, nx, x, keys, &res, params);
    }
  } else {
    if (store_pairs) {
      search_knn_hamming_count<HammingComputer, true>(
           ivf, nx, x, keys, k, distances, labels, params);
    } else {
      search_knn_hamming_count<HammingComputer, false>(
           ivf, nx, x, keys, k, distances, labels, params);
    }
  }
}


}  // namespace


void IndexBinaryIVF::search_preassigned(idx_t n, const uint8_t *x, idx_t k,
                                        const idx_t *idx,
                                        const int32_t * /* coarse_dis */,
                                        int32_t *distances, idx_t *labels,
                                        bool store_pairs,
                                        const IVFSearchParameters *params
                                        ) const {
  switch (code_size) {
#define HANNDLE_CS(cs)                                  \
    case cs:                                            \
      search_knn_hamming<HammingComputer ## cs>(        \
            *this, n, x, idx, k, distances,             \
            labels, store_pairs, use_heap, params);     \
      break;
      HANNDLE_CS(4);
      HANNDLE_CS(8);
      HANNDLE_CS(16);
      HANNDLE_CS(20);
      HANNDLE_CS(32);
      HANNDLE_CS(64);
#undef HANNDLE_CS

    default:
      if (code_size % 8 == 0) {
        search_knn_hamming<HammingComputerM8>(
             *this, n, x, idx, k, distances,
             labels, store_pairs, use_heap, params);
      } else if (code_size % 4 == 0) {
          search_knn_hamming<HammingComputerM4>(
             *this, n, x, idx, k, distances,
             labels, store_pairs, use_heap, params);
      } else {
          search_knn_hamming<HammingComputerDefault>(
             *this, n, x, idx, k, distances,
             labels, store_pairs, use_heap, params);
      }
      break;
  }
}

IndexBinaryIVF::~IndexBinaryIVF() {
  if (own_invlists) {
    delete invlists;
  }

  if (own_fields) {
    delete quantizer;
  }
}


}  // namespace faiss
