/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/impl/platform_macros.h>

namespace faiss {

/** Indirect sort of a floating-point array
 *
 * @param n     size of the array
 * @param vals  array to sort, size n
 * @param perm  output: permutation of [0..n-1], st.
 *              vals[perm[i + 1]] >= vals[perm[i]]
 */
void fvec_argsort(size_t n, const float* vals, size_t* perm);

/** Same as fvec_argsort, parallelized */
void fvec_argsort_parallel(size_t n, const float* vals, size_t* perm);

/// increase verbosity of the bucket_sort functions
FAISS_API extern int bucket_sort_verbose;

/** Bucket sort of a list of values
 *
 * @param vals     values to sort, size nval, max value nbucket - 1
 * @param lims     output limits of buckets, size nbucket + 1
 * @param perm     output buckets, the elements of bucket
 *                 i are in perm[lims[i]:lims[i + 1]]
 * @param nt       number of threads (0 = pure sequential code)
 */
void bucket_sort(
        size_t nval,
        const uint64_t* vals,
        uint64_t nbucket,
        int64_t* lims,
        int64_t* perm,
        int nt = 0);

/** in-place bucket sort (with attention to memory=>int32)
 * on input the values are in a nrow * col matrix
 * we want to store the row numbers in the output.
 *
 * @param vals     positive values to sort, size nrow * ncol,
 *                 max value nbucket - 1
 * @param lims     output limits of buckets, size nbucket + 1
 * @param nt       number of threads (0 = pure sequential code)
 */
void matrix_bucket_sort_inplace(
        size_t nrow,
        size_t ncol,
        int32_t* vals,
        int32_t nbucket,
        int64_t* lims,
        int nt = 0);

/// same with int64 elements
void matrix_bucket_sort_inplace(
        size_t nrow,
        size_t ncol,
        int64_t* vals,
        int64_t nbucket,
        int64_t* lims,
        int nt = 0);

/** Hashtable implementation for int64 -> int64 with external storage
 * implemented for fast batch add and lookup.
 *
 * tab is of size  2 * (1 << log2_capacity)
 * n is the number of elements to add or search
 *
 * adding several values in a same batch: an arbitrary one gets added
 * in different batches: the newer batch overwrites.
 * raises an exception if capacity is exhausted.
 */

void hashtable_int64_to_int64_init(int log2_capacity, int64_t* tab);

void hashtable_int64_to_int64_add(
        int log2_capacity,
        int64_t* tab,
        size_t n,
        const int64_t* keys,
        const int64_t* vals);

void hashtable_int64_to_int64_lookup(
        int log2_capacity,
        const int64_t* tab,
        size_t n,
        const int64_t* keys,
        int64_t* vals);

} // namespace faiss
