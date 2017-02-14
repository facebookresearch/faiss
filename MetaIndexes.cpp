
/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the CC-by-NC license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved
// -*- c++ -*-

#include "MetaIndexes.h"

#include <pthread.h>

#include <cstdio>

#include "FaissAssert.h"
#include "Heap.h"

namespace faiss {

/*****************************************************
 * IndexIDMap implementation
 *******************************************************/

IndexIDMap::IndexIDMap (Index *index):
    index (index),
    own_fields (false)
{
    FAISS_ASSERT (index->ntotal == 0 || !"index must be empty on input");
    is_trained = index->is_trained;
    metric_type = index->metric_type;
    verbose = index->verbose;
    d = index->d;
    set_typename ();
}

void IndexIDMap::add (idx_t, const float *)
{
   FAISS_ASSERT (!"add does not make sense with IndexIDMap, "
                 "use add_with_ids");
}


void IndexIDMap::train (idx_t n, const float *x)
{
    index->train (n, x);
    is_trained = index->is_trained;
}

void IndexIDMap::reset ()
{
    index->reset ();
    ntotal = 0;
}


void IndexIDMap::add_with_ids (idx_t n, const float * x, const long *xids)
{
    index->add (n, x);
    for (idx_t i = 0; i < n; i++)
        id_map.push_back (xids[i]);
    ntotal = index->ntotal;
}


void IndexIDMap::search (idx_t n, const float *x, idx_t k,
                              float *distances, idx_t *labels) const
{
    index->search (n, x, k, distances, labels);
    idx_t *li = labels;
    for (idx_t i = 0; i < n * k; i++) {
        li[i] = li[i] < 0 ? li[i] : id_map[li[i]];
    }
}



IndexIDMap::~IndexIDMap ()
{
    if (own_fields) delete index;
}

void IndexIDMap::set_typename ()
{
    index_typename = "IDMap[" + index->index_typename + "]";
}


/*****************************************************
 * IndexShards implementation
 *******************************************************/

// subroutines
namespace {


typedef Index::idx_t idx_t;


template<class Job>
struct Thread {
    Job job;
    pthread_t thread;

    Thread () {}

    explicit Thread (const Job & job): job(job) {}

    void start () {
        pthread_create (&thread, nullptr, run, this);
    }

    void wait () {
        pthread_join (thread, nullptr);
    }

    static void * run (void *arg) {
        static_cast<Thread*> (arg)->job.run();
        return nullptr;
    }

};


/// callback + thread management to train 1 shard
struct TrainJob {
    IndexShards *index;    // the relevant index
    int no;                // shard number
    idx_t n;               // train points
    const float *x;

    void run ()
    {
        if (index->verbose)
            printf ("begin train shard %d on %ld points\n", no, n);
        index->shard_indexes [no]->train(n, x);
        if (index->verbose)
            printf ("end train shard %d\n", no);
    }

};

struct AddJob {
    IndexShards *index;    // the relevant index
    int no;                // shard number

    idx_t n;
    const float *x;
    const idx_t *ids;

    void run ()
    {
        if (index->verbose)
            printf ("begin add shard %d on %ld points\n", no, n);
        if (ids)
            index->shard_indexes[no]->add_with_ids (n, x, ids);
        else
            index->shard_indexes[no]->add (n, x);
        if (index->verbose)
            printf ("end add shard %d on %ld points\n", no, n);
    }
};



/// callback + thread management to query in 1 shard
struct QueryJob {
    const IndexShards *index;    // the relevant index
    int no;                // shard number

    // query params
    idx_t n;
    const float *x;
    idx_t k;
    float *distances;
    idx_t *labels;


    void run ()
    {
        if (index->verbose)
            printf ("begin query shard %d on %ld points\n", no, n);
        index->shard_indexes [no]->search (n, x, k,
                                           distances, labels);
        if (index->verbose)
            printf ("end query shard %d\n", no);
    }


};




// add translation to all valid labels
void translate_labels (long n, idx_t *labels, long translation)
{
    if (translation == 0) return;
    for (long i = 0; i < n; i++) {
        if(labels[i] < 0) return;
        labels[i] += translation;
    }
}


/** merge result tables from several shards.
 * @param all_distances  size nshard * n * k
 * @param all_labels     idem
 * @param translartions  label translations to apply, size nshard
 */
template <class C>
void merge_tables (long n, long k, long nshard,
                   float *distances, idx_t *labels,
                   const float *all_distances,
                   idx_t *all_labels,
                   const long *translations)
{
    long shard_stride = n * k;
#pragma omp parallel for
    for (long i = 0; i < n; i++) {
        float *D = distances + i * k;
        idx_t *I = labels + i * k;
        const float *Ds = all_distances + i * k;
        idx_t *Is = all_labels + i * k;
        translate_labels (k, Is, translations[0]);
        heap_heapify<C>(k, D, I, Ds, Is, k);
        for (int s = 1; s < nshard; s++) {
            Ds += shard_stride;
            Is += shard_stride;
            translate_labels (k, Is, translations[s]);
            heap_addn<C> (k, D, I, Ds, Is, k);
        }
        heap_reorder<C>(k, D, I);
    }
}


};




IndexShards::IndexShards (idx_t d, bool threaded, bool successive_ids):
    Index (d), own_fields (false),
    threaded (threaded), successive_ids (successive_ids)
{

}

void IndexShards::add_shard (Index *idx)
{
    shard_indexes.push_back (idx);
    sync_with_shard_indexes ();
}

void IndexShards::sync_with_shard_indexes ()
{
    if (shard_indexes.empty()) return;
    Index * index0 = shard_indexes[0];
    d = index0->d;
    metric_type = index0->metric_type;
    is_trained = index0->is_trained;
    ntotal = index0->ntotal;
    for (int i = 1; i < shard_indexes.size(); i++) {
        Index * index = shard_indexes[i];
        FAISS_ASSERT (metric_type == index->metric_type);
        FAISS_ASSERT (d == index->d);
        ntotal += index->ntotal;
    }
}


void IndexShards::train (idx_t n, const float *x)
{

    // pre-alloc because we don't want reallocs
    std::vector<Thread<TrainJob > > tss (shard_indexes.size());
    int nt = 0;
    for (int i = 0; i < shard_indexes.size(); i++) {
        if(!shard_indexes[i]->is_trained) {
            TrainJob ts = {this, i, n, x};
            if (threaded) {
                tss[nt] = Thread<TrainJob> (ts);
                tss[nt++].start();
            } else {
                ts.run();
            }
        }
    }
    for (int i = 0; i < nt; i++) {
        tss[i].wait();
    }
    sync_with_shard_indexes ();
}

void IndexShards::add (idx_t n, const float *x)
{
    add_with_ids (n, x, nullptr);
}

 /**
  * Cases (successive_ids, xids):
  * - true, non-NULL       ERROR: it makes no sense to pass in ids and
  *                        request them to be shifted
  * - true, NULL           OK, but should be called only once (calls add()
  *                        on sub-indexes).
  * - false, non-NULL      OK: will call add_with_ids with passed in xids
  *                        distributed evenly over shards
  * - false, NULL          OK: will call add_with_ids on each sub-index,
  *                        starting at ntotal
  */

void IndexShards::add_with_ids (idx_t n, const float * x, const long *xids)
{

    FAISS_ASSERT(!(successive_ids && xids) ||
        !"It makes no sense to pass in ids and request them to be shifted");

    if (successive_ids) {
        FAISS_ASSERT(!xids ||
           !"It makes no sense to pass in ids and request them to be shifted");
        FAISS_ASSERT(ntotal == 0 ||
            !"when adding to IndexShards with sucessive_ids, only add() "
            "in a single pass is supported");
    }

    long nshard = shard_indexes.size();
    const long *ids = xids;
    if (!ids && !successive_ids) {
        long *aids = new long[n];
        for (long i = 0; i < n; i++)
            aids[i] = ntotal + i;
        ids = aids;
    }

    std::vector<Thread<AddJob > > asa (shard_indexes.size());
    int nt = 0;
    for (int i = 0; i < nshard; i++) {
        long i0 = i * n / nshard;
        long i1 = (i + 1) * n / nshard;

        AddJob as = {this, i,
                       i1 - i0, x + i0 * d,
                       ids ? ids + i0 : nullptr};
        if (threaded) {
            asa[nt] = Thread<AddJob>(as);
            asa[nt++].start();
        } else {
            as.run();
        }
    }
    for (int i = 0; i < nt; i++) {
        asa[i].wait();
    }
    if (ids != xids) delete [] ids;
    ntotal += n;
}


void IndexShards::reset ()
{
    for (int i = 0; i < shard_indexes.size(); i++) {
        shard_indexes[i]->reset ();
    }
    sync_with_shard_indexes ();
}

void IndexShards::search (
           idx_t n, const float *x, idx_t k,
           float *distances, idx_t *labels) const
{
    long nshard = shard_indexes.size();
    float *all_distances = new float [nshard * k * n];
    idx_t *all_labels = new idx_t [nshard * k * n];

#if 1

    // pre-alloc because we don't want reallocs
    std::vector<Thread<QueryJob> > qss (nshard);
    for (int i = 0; i < nshard; i++) {
        QueryJob qs = {
            this, i, n, x, k,
            all_distances + i * k * n,
            all_labels + i * k * n
        };
        if (threaded) {
            qss[i] = Thread<QueryJob> (qs);
            qss[i].start();
        } else {
            qs.run();
        }
    }

    if (threaded) {
        for (int i = 0; i < qss.size(); i++) {
            qss[i].wait();
        }
    }
#else

    // pre-alloc because we don't want reallocs
    std::vector<QueryJob> qss (nshard);
    for (int i = 0; i < nshard; i++) {
        QueryJob qs = {
            this, i, n, x, k,
            all_distances + i * k * n,
            all_labels + i * k * n
        };
        if (threaded) {
            qss[i] = qs;
        } else {
            qs.run();
        }
    }

    if (threaded) {
#pragma omp parallel for
        for (int i = 0; i < qss.size(); i++) {
            qss[i].run();
        }
    }

#endif
    std::vector<long> translations (nshard, 0);
    if (successive_ids) {
        translations[0] = 0;
        for (int s = 0; s + 1 < nshard; s++)
            translations [s + 1] = translations [s] +
                shard_indexes [s]->ntotal;
    }

    if (metric_type == METRIC_L2) {
        merge_tables< CMax<float, idx_t> > (
             n, k, nshard, distances, labels,
             all_distances, all_labels, translations.data ());
    } else {
        merge_tables< CMin<float, idx_t> > (
             n, k, nshard, distances, labels,
             all_distances, all_labels, translations.data ());
    }

    delete [] all_distances;
    delete [] all_labels;
}


void IndexShards::set_typename ()
{

}

IndexShards::~IndexShards ()
{
    if (own_fields) {
        for (int s = 0; s < shard_indexes.size(); s++)
            delete shard_indexes [s];
    }
}


/*****************************************************
 * IndexSplitVectors implementation
 *******************************************************/


IndexSplitVectors::IndexSplitVectors (idx_t d, bool threaded):
    Index (d), own_fields (false),
    threaded (threaded), sum_d (0)
{

}

void IndexSplitVectors::add_sub_index (Index *index)
{
    sub_indexes.push_back (index);
    sync_with_sub_indexes ();
}

void IndexSplitVectors::sync_with_sub_indexes ()
{
    if (sub_indexes.empty()) return;
    Index * index0 = sub_indexes[0];
    sum_d = index0->d;
    metric_type = index0->metric_type;
    is_trained = index0->is_trained;
    ntotal = index0->ntotal;
    for (int i = 1; i < sub_indexes.size(); i++) {
        Index * index = sub_indexes[i];
        FAISS_ASSERT (metric_type == index->metric_type);
        FAISS_ASSERT (ntotal == index->ntotal);
        sum_d += index->d;
    }

}

void IndexSplitVectors::add (idx_t n, const float *x)
{
    FAISS_ASSERT (!"not implemented");
}

namespace {

/// callback + thread management to query in 1 shard
struct SplitQueryJob {
    const IndexSplitVectors *index;    // the relevant index
    int no;                // shard number

    // query params
    idx_t n;
    const float *x;
    idx_t k;
    float *distances;
    idx_t *labels;


    void run ()
    {
        if (index->verbose)
            printf ("begin query shard %d on %ld points\n", no, n);
        const Index * sub_index = index->sub_indexes[no];
        long sub_d = sub_index->d, d = index->d;
        idx_t ofs = 0;
        for (int i = 0; i < no; i++) ofs += index->sub_indexes[i]->d;
        float *sub_x = new float [sub_d * n];
        for (idx_t i = 0; i < n; i++)
            memcpy (sub_x + i * sub_d, x + ofs + i * d, sub_d * sizeof (sub_x));
        sub_index->search (n, sub_x, k, distances, labels);
        delete [] sub_x;
        if (index->verbose)
            printf ("end query shard %d\n", no);
    }

};



}



void IndexSplitVectors::search (
           idx_t n, const float *x, idx_t k,
           float *distances, idx_t *labels) const
{
    FAISS_ASSERT (k == 1 || !"search implemented only for k=1");
    FAISS_ASSERT (sum_d == d || !"not enough indexes compared to # dimensions");

    long nshard = sub_indexes.size();
    float *all_distances = new float [nshard * k * n];
    idx_t *all_labels = new idx_t [nshard * k * n];

    // pre-alloc because we don't want reallocs
    std::vector<Thread<SplitQueryJob> > qss (nshard);
    for (int i = 0; i < nshard; i++) {
        SplitQueryJob qs = {
            this, i, n, x, k,
            i == 0 ? distances : all_distances + i * k * n,
            i == 0 ? labels : all_labels + i * k * n
        };
        if (threaded) {
            qss[i] = Thread<SplitQueryJob> (qs);
            qss[i].start();
        } else {
            qs.run();
        }
    }

    if (threaded) {
        for (int i = 0; i < qss.size(); i++) {
            qss[i].wait();
        }
    }

    long factor = 1;
    for (int i = 0; i < nshard; i++) {
        if (i > 0) { // results of 0 are already in the table
            const float *distances_i = all_distances + i * k * n;
            const idx_t *labels_i = all_labels + i * k * n;
            for (long j = 0; j < n; j++) {
                if (labels[j] >= 0 && labels_i[j] >= 0) {
                    labels[j] += labels_i[j] * factor;
                    distances[j] += distances_i[j];
                } else {
                    labels[j] = -1;
                    distances[j] = 0.0 / 0.0;
                }
            }
        }
        factor *= sub_indexes[i]->ntotal;
    }
    delete [] all_labels;
    delete [] all_distances;
}


void IndexSplitVectors::train (idx_t n, const float *x)
{
    FAISS_ASSERT (!"not implemented");
}

void IndexSplitVectors::reset ()
{
    FAISS_ASSERT (!"not implemented");
}

void IndexSplitVectors::set_typename ()
{}

IndexSplitVectors::~IndexSplitVectors ()
{
    if (own_fields) {
        for (int s = 0; s < sub_indexes.size(); s++)
            delete sub_indexes [s];
    }
}






}; // namespace faiss
