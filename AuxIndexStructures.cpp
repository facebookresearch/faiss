/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved
// -*- c++ -*-

#include "AuxIndexStructures.h"

#include <cstring>

namespace faiss {


/***********************************************************************
 * RangeSearchResult
 ***********************************************************************/

RangeSearchResult::RangeSearchResult (idx_t nq, bool alloc_lims): nq (nq) {
    if (alloc_lims) {
        lims = new size_t [nq + 1];
        memset (lims, 0, sizeof(*lims) * (nq + 1));
    } else {
        lims = nullptr;
    }
    labels = nullptr;
    distances = nullptr;
    buffer_size = 1024 * 256;
}

/// called when lims contains the nb of elements result entries
/// for each query
void RangeSearchResult::do_allocation () {
    size_t ofs = 0;
    for (int i = 0; i < nq; i++) {
        size_t n = lims[i];
        lims [i] = ofs;
        ofs += n;
    }
    lims [nq] = ofs;
    labels = new idx_t [ofs];
    distances = new float [ofs];
}

RangeSearchResult::~RangeSearchResult () {
    delete [] labels;
    delete [] distances;
    delete [] lims;
}

/***********************************************************************
 * BufferList
 ***********************************************************************/


BufferList::BufferList (size_t buffer_size):
    buffer_size (buffer_size)
{
    wp = buffer_size;
}

BufferList::~BufferList ()
{
    for (int i = 0; i < buffers.size(); i++) {
        delete [] buffers[i].ids;
        delete [] buffers[i].dis;
    }
}



void BufferList::append_buffer ()
{
        Buffer buf = {new idx_t [buffer_size], new float [buffer_size]};
        buffers.push_back (buf);
        wp = 0;
}

/// copy elemnts ofs:ofs+n-1 seen as linear data in the buffers to
/// tables dest_ids, dest_dis
void BufferList::copy_range (size_t ofs, size_t n,
                             idx_t * dest_ids, float *dest_dis)
{
    size_t bno = ofs / buffer_size;
    ofs -= bno * buffer_size;
    while (n > 0) {
        size_t ncopy = ofs + n < buffer_size ? n : buffer_size - ofs;
        Buffer buf = buffers [bno];
        memcpy (dest_ids, buf.ids + ofs, ncopy * sizeof(*dest_ids));
        memcpy (dest_dis, buf.dis + ofs, ncopy * sizeof(*dest_dis));
        dest_ids += ncopy;
        dest_dis += ncopy;
        ofs = 0;
        bno ++;
            n -= ncopy;
    }
}


/***********************************************************************
 * RangeSearchPartialResult
 ***********************************************************************/


RangeSearchPartialResult::RangeSearchPartialResult (RangeSearchResult * res_in):
    BufferList(res_in->buffer_size),
    res(res_in)
{}


/// begin a new result
RangeSearchPartialResult::QueryResult &
    RangeSearchPartialResult::new_result (idx_t qno)
{
    QueryResult qres = {qno, 0, this};
    queries.push_back (qres);
    return queries.back();
}


void RangeSearchPartialResult::finalize ()
{
    set_lims ();
#pragma omp barrier

#pragma omp single
    res->do_allocation ();

#pragma omp barrier
    set_result ();
}


/// called by range_search before do_allocation
void RangeSearchPartialResult::set_lims ()
{
    for (int i = 0; i < queries.size(); i++) {
        QueryResult & qres = queries[i];
        res->lims[qres.qno] = qres.nres;
    }
}

/// called by range_search after do_allocation
void RangeSearchPartialResult::set_result (bool incremental)
{
    size_t ofs = 0;
    for (int i = 0; i < queries.size(); i++) {
        QueryResult & qres = queries[i];

        copy_range (ofs, qres.nres,
                    res->labels + res->lims[qres.qno],
                    res->distances + res->lims[qres.qno]);
        if (incremental) {
            res->lims[qres.qno] += qres.nres;
        }
        ofs += qres.nres;
    }
}


/***********************************************************************
 * IDSelectorRange
 ***********************************************************************/

IDSelectorRange::IDSelectorRange (idx_t imin, idx_t imax):
    imin (imin), imax (imax)
{
}

bool IDSelectorRange::is_member (idx_t id) const
{
    return id >= imin && id < imax;
}


/***********************************************************************
 * IDSelectorBatch
 ***********************************************************************/

IDSelectorBatch::IDSelectorBatch (long n, const idx_t *indices)
{
    nbits = 0;
    while (n > (1L << nbits)) nbits++;
    nbits += 5;
    // for n = 1M, nbits = 25 is optimal, see P56659518

    mask = (1L << nbits) - 1;
    bloom.resize (1UL << (nbits - 3), 0);
    for (long i = 0; i < n; i++) {
        long id = indices[i];
        set.insert(id);
        id &= mask;
        bloom[id >> 3] |= 1 << (id & 7);
    }
}

bool IDSelectorBatch::is_member (idx_t i) const
{
    long im = i & mask;
    if(!(bloom[im>>3] & (1 << (im & 7)))) {
        return 0;
    }
    return set.count(i);
}







}
