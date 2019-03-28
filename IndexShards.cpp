/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include "IndexShards.h"

#include <cstdio>

#include "FaissAssert.h"
#include "Heap.h"
#include "WorkerThread.h"

namespace faiss {

// subroutines
namespace {

typedef Index::idx_t idx_t;

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
    if(k == 0) {
        return;
    }

    long stride = n * k;
#pragma omp parallel
    {
        std::vector<int> buf (2 * nshard);
        int * pointer = buf.data();
        int * shard_ids = pointer + nshard;
        std::vector<float> buf2 (nshard);
        float * heap_vals = buf2.data();
#pragma omp for
        for (long i = 0; i < n; i++) {
            // the heap maps values to the shard where they are
            // produced.
            const float *D_in = all_distances + i * k;
            const idx_t *I_in = all_labels + i * k;
            int heap_size = 0;

            for (long s = 0; s < nshard; s++) {
                pointer[s] = 0;
                if (I_in[stride * s] >= 0)
                    heap_push<C> (++heap_size, heap_vals, shard_ids,
                                 D_in[stride * s], s);
            }

            float *D = distances + i * k;
            idx_t *I = labels + i * k;

            for (int j = 0; j < k; j++) {
                if (heap_size == 0) {
                    I[j] = -1;
                    D[j] = C::neutral();
                } else {
                    // pop best element
                    int s = shard_ids[0];
                    int & p = pointer[s];
                    D[j] = heap_vals[0];
                    I[j] = I_in[stride * s + p] + translations[s];

                    heap_pop<C> (heap_size--, heap_vals, shard_ids);
                    p++;
                    if (p < k && I_in[stride * s + p] >= 0)
                        heap_push<C> (++heap_size, heap_vals, shard_ids,
                                     D_in[stride * s + p], s);
                }
            }
        }
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
        FAISS_THROW_IF_NOT (metric_type == index->metric_type);
        FAISS_THROW_IF_NOT (d == index->d);
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


void IndexShards::add_with_ids (idx_t n, const float * x, const long *xids)
{

    FAISS_THROW_IF_NOT_MSG(!(successive_ids && xids),
                   "It makes no sense to pass in ids and "
                   "request them to be shifted");

    if (successive_ids) {
      FAISS_THROW_IF_NOT_MSG(!xids,
                       "It makes no sense to pass in ids and "
                       "request them to be shifted");
      FAISS_THROW_IF_NOT_MSG(ntotal == 0,
                       "when adding to IndexShards with sucessive_ids, "
                       "only add() in a single pass is supported");
    }

    long nshard = shard_indexes.size();
    const long *ids = xids;
    ScopeDeleter<long> del;
    if (!ids && !successive_ids) {
        long *aids = new long[n];
        for (long i = 0; i < n; i++)
            aids[i] = ntotal + i;
        ids = aids;
        del.set (ids);
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
    ScopeDeleter<float> del (all_distances);
    ScopeDeleter<idx_t> del2 (all_labels);

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
        merge_tables< CMin<float, int> > (
             n, k, nshard, distances, labels,
             all_distances, all_labels, translations.data ());
    } else {
        merge_tables< CMax<float, int> > (
             n, k, nshard, distances, labels,
             all_distances, all_labels, translations.data ());
    }

}



IndexShards::~IndexShards ()
{
    if (own_fields) {
        for (int s = 0; s < shard_indexes.size(); s++)
            delete shard_indexes [s];
    }
}


} // namespace faiss
