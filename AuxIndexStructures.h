/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved
// -*- c++ -*-
// Auxiliary index structures, that are used in indexes but that can
// be forward-declared

#ifndef FAISS_AUX_INDEX_STRUCTURES_H
#define FAISS_AUX_INDEX_STRUCTURES_H

#include <vector>
#include <unordered_set>


#include "Index.h"

namespace faiss {

/** The objective is to have a simple result structure while
 *  minimizing the number of mem copies in the result. The method
 *  do_allocation can be overloaded to allocate the result tables in
 *  the matrix type of a scripting language like Lua or Python. */
struct RangeSearchResult {
    size_t nq;      ///< nb of queries
    size_t *lims;   ///< size (nq + 1)

    typedef Index::idx_t idx_t;

    idx_t *labels;     ///< result for query i is labels[lims[i]:lims[i+1]]
    float *distances;  ///< corresponding distances (not sorted)

    size_t buffer_size; ///< size of the result buffers used

    /// lims must be allocated on input to range_search.
    explicit RangeSearchResult (idx_t nq, bool alloc_lims=true);

    /// called when lims contains the nb of elements result entries
    /// for each query
    virtual void do_allocation ();

    virtual ~RangeSearchResult ();
};


/** Encapsulates a set of ids to remove. */
struct IDSelector {
    typedef Index::idx_t idx_t;
    virtual bool is_member (idx_t id) const = 0;
    virtual ~IDSelector() {}
};



/** remove ids between [imni, imax) */
struct IDSelectorRange: IDSelector {
    idx_t imin, imax;

    IDSelectorRange (idx_t imin, idx_t imax);
    bool is_member(idx_t id) const override;
    ~IDSelectorRange() override {}
};


/** Remove ids from a set. Repetitions of ids in the indices set
 * passed to the constructor does not hurt performance. The hash
 * function used for the bloom filter and GCC's implementation of
 * unordered_set are just the least significant bits of the id. This
 * works fine for random ids or ids in sequences but will produce many
 * hash collisions if lsb's are always the same */
struct IDSelectorBatch: IDSelector {

    std::unordered_set<idx_t> set;

    typedef unsigned char uint8_t;
    std::vector<uint8_t> bloom; // assumes low bits of id are a good hash value
    int nbits;
    idx_t mask;

    IDSelectorBatch (long n, const idx_t *indices);
    bool is_member(idx_t id) const override;
    ~IDSelectorBatch() override {}
};


// Below are structures used only by Index implementations



/** List of temporary buffers used to store results before they are
 *  copied to the RangeSearchResult object. */
struct BufferList {
    typedef Index::idx_t idx_t;

    // buffer sizes in # entries
    size_t buffer_size;

    struct Buffer {
        idx_t *ids;
        float *dis;
    };

    std::vector<Buffer> buffers;
    size_t wp; ///< write pointer in the last buffer.

    explicit BufferList (size_t buffer_size);

    ~BufferList ();

    // create a new buffer
    void append_buffer ();

    inline void add (idx_t id, float dis)
    {
        if (wp == buffer_size) { // need new buffer
            append_buffer();
        }
        Buffer & buf = buffers.back();
        buf.ids [wp] = id;
        buf.dis [wp] = dis;
        wp++;
    }

    /// copy elemnts ofs:ofs+n-1 seen as linear data in the buffers to
    /// tables dest_ids, dest_dis
    void copy_range (size_t ofs, size_t n,
                     idx_t * dest_ids, float *dest_dis);

};



/// the entries in the buffers are split per query
struct RangeSearchPartialResult: BufferList {
    RangeSearchResult * res;

    explicit RangeSearchPartialResult (RangeSearchResult * res_in);

    /// result structure for a single query
    struct QueryResult {
        idx_t qno;
        size_t nres;
        RangeSearchPartialResult * pres;
        inline void add (float dis, idx_t id) {
            nres++;
            pres->add (id, dis);
        }
    };

    std::vector<QueryResult> queries;

    /// begin a new result
    QueryResult & new_result (idx_t qno);


    void finalize ();


    /// called by range_search before do_allocation
    void set_lims ();

    /// called by range_search after do_allocation
    void set_result (bool incremental = false);

};


}; // namespace faiss



#endif
