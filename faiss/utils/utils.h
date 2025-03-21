/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

/*
 *  A few utilitary functions for similarity search:
 * - optimized exhaustive distance and knn search functions
 * - some functions reimplemented from torch for speed
 */

#ifndef FAISS_utils_h
#define FAISS_utils_h

#include <stdint.h>
#include <set>
#include <string>
#include <vector>

#include <faiss/impl/platform_macros.h>
#include <faiss/utils/Heap.h>

namespace faiss {

/****************************************************************************
 * Get compile specific variables
 ***************************************************************************/

/// get compile options
std::string get_compile_options();

/**************************************************
 * Get some stats about the system
 **************************************************/

// Expose Faiss version as a string
std::string get_version();

/// ms elapsed since some arbitrary epoch
double getmillisecs();

/// get current RSS usage in kB
size_t get_mem_usage_kb();

uint64_t get_cycles();

/***************************************************************************
 * Misc  matrix and vector manipulation functions
 ***************************************************************************/

/* perform a reflection (not an efficient implementation, just for test ) */
void reflection(const float* u, float* x, size_t n, size_t d, size_t nu);

/** compute the Q of the QR decomposition for m > n
 * @param a   size n * m: input matrix and output Q
 */
void matrix_qr(int m, int n, float* a);

/** distances are supposed to be sorted. Sorts indices with same distance*/
void ranklist_handle_ties(int k, int64_t* idx, const float* dis);

/** count the number of common elements between v1 and v2
 * algorithm = sorting + bissection to avoid double-counting duplicates
 */
size_t ranklist_intersection_size(
        size_t k1,
        const int64_t* v1,
        size_t k2,
        const int64_t* v2);

/** merge a result table into another one
 *
 * @param I0, D0       first result table, size (n, k)
 * @param I1, D1       second result table, size (n, k)
 * @param keep_min     if true, keep min values, otherwise keep max
 * @param translation  add this value to all I1's indexes
 * @return             nb of values that were taken from the second table
 */
size_t merge_result_table_with(
        size_t n,
        size_t k,
        int64_t* I0,
        float* D0,
        const int64_t* I1,
        const float* D1,
        bool keep_min = true,
        int64_t translation = 0);

/// a balanced assignment has a IF of 1, a completely unbalanced assignment has
/// an IF = k.
double imbalance_factor(int64_t n, int k, const int64_t* assign);

/// same, takes a histogram as input
double imbalance_factor(int k, const int64_t* hist);

/// compute histogram on v
int ivec_hist(size_t n, const int* v, int vmax, int* hist);

/** Compute histogram of bits on a code array
 *
 * @param codes   size(n, nbits / 8)
 * @param hist    size(nbits): nb of 1s in the array of codes
 */
void bincode_hist(size_t n, size_t nbits, const uint8_t* codes, int* hist);

/// compute a checksum on a table.
uint64_t ivec_checksum(size_t n, const int32_t* a);

/// compute a checksum on a table.
uint64_t bvec_checksum(size_t n, const uint8_t* a);

/** compute checksums for the rows of a matrix
 *
 * @param n   number of rows
 * @param d   size per row
 * @param a   matrix to handle, size n * d
 * @param cs  output checksums, size n
 */
void bvecs_checksum(size_t n, size_t d, const uint8_t* a, uint64_t* cs);

/** random subsamples a set of vectors if there are too many of them
 *
 * @param d      dimension of the vectors
 * @param n      on input: nb of input vectors, output: nb of output vectors
 * @param nmax   max nb of vectors to keep
 * @param x      input array, size *n-by-d
 * @param seed   random seed to use for sampling
 * @return       x or an array allocated with new [] with *n vectors
 */
const float* fvecs_maybe_subsample(
        size_t d,
        size_t* n,
        size_t nmax,
        const float* x,
        bool verbose = false,
        int64_t seed = 1234);

/** Convert binary vector to +1/-1 valued float vector.
 *
 * @param d      dimension of the vector (multiple of 8)
 * @param x_in   input binary vector (uint8_t table of size d / 8)
 * @param x_out  output float vector (float table of size d)
 */
void binary_to_real(size_t d, const uint8_t* x_in, float* x_out);

/** Convert float vector to binary vector. Components > 0 are converted to 1,
 * others to 0.
 *
 * @param d      dimension of the vector (multiple of 8)
 * @param x_in   input float vector (float table of size d)
 * @param x_out  output binary vector (uint8_t table of size d / 8)
 */
void real_to_binary(size_t d, const float* x_in, uint8_t* x_out);

/** A reasonable hashing function */
uint64_t hash_bytes(const uint8_t* bytes, int64_t n);

/** Whether OpenMP annotations were respected. */
bool check_openmp();

/** This class is used to combine range and knn search results
 * in contrib.exhaustive_search.range_search_gpu */

template <typename T>
struct CombinerRangeKNN {
    int64_t nq;    /// nb of queries
    size_t k;      /// number of neighbors for the knn search part
    T r2;          /// range search radius
    bool keep_max; /// whether to keep max values instead of min.

    CombinerRangeKNN(int64_t nq, size_t k, T r2, bool keep_max)
            : nq(nq), k(k), r2(r2), keep_max(keep_max) {}

    /// Knn search results
    const int64_t* I = nullptr; /// size nq * k
    const T* D = nullptr;       /// size nq * k

    /// optional: range search results (ignored if mask is NULL)
    const bool* mask =
            nullptr; /// mask for where knn results are valid, size nq
    // range search results for remaining entries nrange = sum(mask)
    const int64_t* lim_remain = nullptr; /// size nrange + 1
    const T* D_remain = nullptr;         /// size lim_remain[nrange]
    const int64_t* I_remain = nullptr;   /// size lim_remain[nrange]

    const int64_t* L_res = nullptr; /// size nq + 1
    // Phase 1: compute sizes into limits array (of size nq + 1)
    void compute_sizes(int64_t* L_res);

    /// Phase 2: caller allocates D_res and I_res (size L_res[nq])
    /// Phase 3: fill in D_res and I_res
    void write_result(T* D_res, int64_t* I_res);
};

struct CodeSet {
    size_t d;
    std::set<std::vector<uint8_t>> s;

    explicit CodeSet(size_t d) : d(d) {}
    void insert(size_t n, const uint8_t* codes, bool* inserted);
};

} // namespace faiss

#endif /* FAISS_utils_h */
