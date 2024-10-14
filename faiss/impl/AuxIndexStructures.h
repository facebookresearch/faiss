/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Auxiliary index structures, that are used in indexes but that can
// be forward-declared

#ifndef FAISS_AUX_INDEX_STRUCTURES_H
#define FAISS_AUX_INDEX_STRUCTURES_H

#include <stdint.h>

#include <cstring>
#include <memory>
#include <mutex>
#include <vector>

#include <faiss/MetricType.h>
#include <faiss/impl/platform_macros.h>

namespace faiss {

/** The objective is to have a simple result structure while
 *  minimizing the number of mem copies in the result. The method
 *  do_allocation can be overloaded to allocate the result tables in
 *  the matrix type of a scripting language like Lua or Python. */
struct RangeSearchResult {
    size_t nq;    ///< nb of queries
    size_t* lims; ///< size (nq + 1)

    idx_t* labels;    ///< result for query i is labels[lims[i]:lims[i+1]]
    float* distances; ///< corresponding distances (not sorted)

    size_t buffer_size; ///< size of the result buffers used

    /// lims must be allocated on input to range_search.
    explicit RangeSearchResult(size_t nq, bool alloc_lims = true);

    /// called when lims contains the nb of elements result entries
    /// for each query
    virtual void do_allocation();

    virtual ~RangeSearchResult();
};

/****************************************************************
 * Result structures for range search.
 *
 * The main constraint here is that we want to support parallel
 * queries from different threads in various ways: 1 thread per query,
 * several threads per query. We store the actual results in blocks of
 * fixed size rather than exponentially increasing memory. At the end,
 * we copy the block content to a linear result array.
 *****************************************************************/

/** List of temporary buffers used to store results before they are
 *  copied to the RangeSearchResult object. */
struct BufferList {
    // buffer sizes in # entries
    size_t buffer_size;

    struct Buffer {
        idx_t* ids;
        float* dis;
    };

    std::vector<Buffer> buffers;
    size_t wp; ///< write pointer in the last buffer.

    explicit BufferList(size_t buffer_size);

    ~BufferList();

    /// create a new buffer
    void append_buffer();

    /// add one result, possibly appending a new buffer if needed
    void add(idx_t id, float dis);

    /// copy elemnts ofs:ofs+n-1 seen as linear data in the buffers to
    /// tables dest_ids, dest_dis
    void copy_range(size_t ofs, size_t n, idx_t* dest_ids, float* dest_dis);
};

struct RangeSearchPartialResult;

/// result structure for a single query
struct RangeQueryResult {
    idx_t qno;   //< id of the query
    size_t nres; //< nb of results for this query
    RangeSearchPartialResult* pres;

    /// called by search function to report a new result
    void add(float dis, idx_t id);
};

/// the entries in the buffers are split per query
struct RangeSearchPartialResult : BufferList {
    RangeSearchResult* res;

    /// eventually the result will be stored in res_in
    explicit RangeSearchPartialResult(RangeSearchResult* res_in);

    /// query ids + nb of results per query.
    std::vector<RangeQueryResult> queries;

    /// begin a new result
    RangeQueryResult& new_result(idx_t qno);

    /*****************************************
     * functions used at the end of the search to merge the result
     * lists */
    void finalize();

    /// called by range_search before do_allocation
    void set_lims();

    /// called by range_search after do_allocation
    void copy_result(bool incremental = false);

    /// merge a set of PartialResult's into one RangeSearchResult
    /// on output the partialresults are empty!
    static void merge(
            std::vector<RangeSearchPartialResult*>& partial_results,
            bool do_delete = true);
};

/***********************************************************
 * Interrupt callback
 ***********************************************************/

struct FAISS_API InterruptCallback {
    virtual bool want_interrupt() = 0;
    virtual ~InterruptCallback() {}

    // lock that protects concurrent calls to is_interrupted
    static std::mutex lock;

    static std::unique_ptr<InterruptCallback> instance;

    static void clear_instance();

    /** check if:
     * - an interrupt callback is set
     * - the callback returns true
     * if this is the case, then throw an exception. Should not be called
     * from multiple threads.
     */
    static void check();

    /// same as check() but return true if is interrupted instead of
    /// throwing. Can be called from multiple threads.
    static bool is_interrupted();

    /** assuming each iteration takes a certain number of flops, what
     * is a reasonable interval to check for interrupts?
     */
    static size_t get_period_hint(size_t flops);
};

struct TimeoutCallback : InterruptCallback {
    std::chrono::time_point<std::chrono::steady_clock> start;
    double timeout;
    bool want_interrupt() override;
    void set_timeout(double timeout_in_seconds);
    static void reset(double timeout_in_seconds);
};

/// set implementation optimized for fast access.
struct VisitedTable {
    std::vector<uint8_t> visited;
    uint8_t visno;

    explicit VisitedTable(int size) : visited(size), visno(1) {}

    /// set flag #no to true
    void set(int no) {
        visited[no] = visno;
    }

    /// get flag #no
    bool get(int no) const {
        return visited[no] == visno;
    }

    /// reset all flags to false
    void advance() {
        visno++;
        if (visno == 250) {
            // 250 rather than 255 because sometimes we use visno and visno+1
            memset(visited.data(), 0, sizeof(visited[0]) * visited.size());
            visno = 1;
        }
    }
};

} // namespace faiss

#endif
