
/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the CC-by-NC license found in the
 * LICENSE file in the root directory of this source tree.
 */

/* Copyright 2004-present Facebook. All Rights Reserved.
   Index based on product quantiztion.
*/

#include "IndexPQ.h"


#include <cstddef>
#include <cstring>
#include <cstdio>

#include <algorithm>

#include "FaissAssert.h"
#include "hamming.h"

namespace faiss {

/*********************************************************
 * IndexPQ implementation
 ********************************************************/


IndexPQ::IndexPQ (int d, size_t M, size_t nbits, MetricType metric):
    Index(d, metric), pq(d, M, nbits)
{
    is_trained = false;
    do_polysemous_training = false;
    set_typename();
    polysemous_ht = nbits * M + 1;
    search_type = ST_PQ;
    encode_signs = false;
}

IndexPQ::IndexPQ ()
{
    metric_type = METRIC_L2;
    is_trained = false;
    do_polysemous_training = false;
    set_typename();
    polysemous_ht = pq.nbits * pq.M + 1;
    search_type = ST_PQ;
    encode_signs = false;
}


void IndexPQ::set_typename ()
{
    std::stringstream s;
    s << "PQ_" << pq.M << "x" << pq.nbits;
    index_typename = s.str();
}


void IndexPQ::train (idx_t n, const float *x)
{
    if (!do_polysemous_training) {        // standard training
        pq.train(n, x);
    } else {
        idx_t ntrain_perm = polysemous_training.ntrain_permutation;

        if (ntrain_perm > n / 4)
            ntrain_perm = n / 4;
        if (verbose) {
            printf ("PQ training on %ld points, remains %ld points: "
                    "training polysemous on %s\n",
                    n - ntrain_perm, ntrain_perm,
                    ntrain_perm == 0 ? "centroids" : "these");
        }
        pq.train(n - ntrain_perm, x);

        polysemous_training.optimize_pq_for_hamming (
            pq, ntrain_perm, x + (n - ntrain_perm) * d);
    }
    is_trained = true;
}


void IndexPQ::add (idx_t n, const float *x)
{
    FAISS_ASSERT (is_trained);
    codes.resize ((n + ntotal) * pq.code_size);
    pq.compute_codes (x, &codes[ntotal * pq.code_size], n);
    ntotal += n;
}



void IndexPQ::reset()
{
    codes.clear();
    ntotal = 0;
}

void IndexPQ::reconstruct_n (idx_t i0, idx_t ni, float *recons) const
{
    FAISS_ASSERT (ni == 0 || (i0 >= 0 && i0 + ni <= ntotal));
    for (idx_t i = 0; i < ni; i++) {
        const uint8_t * code = &codes[(i0 + i) * pq.code_size];
        pq.decode (code, recons + i * d);
    }
}


void IndexPQ::reconstruct (idx_t key, float * recons) const
{
    FAISS_ASSERT (key >= 0 && key < ntotal);
    pq.decode (&codes[key * pq.code_size], recons);
}







/*****************************************
 * IndexPQ polysemous search routines
 ******************************************/





void IndexPQ::search (idx_t n, const float *x, idx_t k,
                           float *distances, idx_t *labels) const
{
    FAISS_ASSERT (is_trained);
    if (search_type == ST_PQ) {  // Simple PQ search

        if (metric_type == METRIC_L2) {
            float_maxheap_array_t res = {
                size_t(n), size_t(k), labels, distances };
            pq.search (x, n, codes.data(), ntotal, &res, true);
        } else {
            float_minheap_array_t res = {
                size_t(n), size_t(k), labels, distances };
            pq.search_ip (x, n, codes.data(), ntotal, &res, true);
        }
        indexPQ_stats.nq += n;
        indexPQ_stats.ncode += n * ntotal;

    } else if (search_type == ST_polysemous ||
               search_type == ST_polysemous_generalize) {

        FAISS_ASSERT (metric_type == METRIC_L2);

        search_core_polysemous (n, x, k, distances, labels);

    } else { // code-to-code distances

        uint8_t * q_codes = new uint8_t [n * pq.code_size];

        if (!encode_signs) {

            pq.compute_codes (x, q_codes, n);



        } else {
            FAISS_ASSERT (d == pq.nbits * pq.M);
            memset (q_codes, 0, n * pq.code_size);
            for (size_t i = 0; i < n; i++) {
                const float *xi = x + i * d;
                uint8_t *code = q_codes + i * pq.code_size;
                for (int j = 0; j < d; j++)
                    if (xi[j] > 0) code [j>>3] |= 1 << (j & 7);
            }
        }

        if (search_type == ST_SDC)  {

            float_maxheap_array_t res = {
                size_t(n),  size_t(k), labels, distances};

            pq.search_sdc (q_codes, n, codes.data(), ntotal, &res, true);

        } else {
            int * idistances = new int [n * k];

            int_maxheap_array_t res = {
                size_t (n), size_t (k), labels, idistances};

            if (search_type == ST_HE) {

                hammings_knn (&res, q_codes, codes.data(),
                              ntotal, pq.code_size, true);

            } else if (search_type == ST_generalized_HE) {

                generalized_hammings_knn (&res, q_codes, codes.data(),
                                          ntotal, pq.code_size, true);
            }

            // convert distances to floats
            for (int i = 0; i < k * n; i++)
                distances[i] = idistances[i];
            delete [] idistances;
        }

        delete [] q_codes;
        indexPQ_stats.nq += n;
        indexPQ_stats.ncode += n * ntotal;
    }
}





void IndexPQStats::reset()
{
    nq = ncode = n_hamming_pass = 0;
}

IndexPQStats indexPQ_stats;


template <class HammingComputer>
static size_t polysemous_inner_loop (
        const IndexPQ & index,
        const float *dis_table_qi, const uint8_t *q_code,
        size_t k, float *heap_dis, long *heap_ids)
{

    int M = index.pq.M;
    int code_size = index.pq.code_size;
    int ksub = index.pq.ksub;
    size_t ntotal = index.ntotal;
    int ht = index.polysemous_ht;

    const uint8_t *b_code = index.codes.data();

    size_t n_pass_i = 0;

    HammingComputer hc (q_code, code_size);

    for (long bi = 0; bi < ntotal; bi++) {
        int hd = hc.hamming (b_code);

        if (hd < ht) {
            n_pass_i ++;

            float dis = 0;
            const float * dis_table = dis_table_qi;
            for (int m = 0; m < M; m++) {
                dis += dis_table [b_code[m]];
                dis_table += ksub;
            }

            if (dis < heap_dis[0]) {
                maxheap_pop (k, heap_dis, heap_ids);
                maxheap_push (k, heap_dis, heap_ids, dis, bi);
            }
        }
        b_code += code_size;
    }
    return n_pass_i;
}


void IndexPQ::search_core_polysemous (idx_t n, const float *x, idx_t k,
                                          float *distances, idx_t *labels) const
{
    FAISS_ASSERT (pq.code_size % 8 == 0);
    FAISS_ASSERT (pq.byte_per_idx == 1);

    // PQ distance tables
    float * dis_tables = new float [n * pq.ksub * pq.M];
    pq.compute_distance_tables (n, x, dis_tables);

    // Hamming embedding queries
    uint8_t * q_codes = new uint8_t [n * pq.code_size];

    if (false) {
        pq.compute_codes (x, q_codes, n);
    } else {
#pragma omp parallel for
        for (idx_t qi = 0; qi < n; qi++) {
            pq.compute_code_from_distance_table
                (dis_tables + qi * pq.M * pq.ksub,
                 q_codes + qi * pq.code_size);
        }
    }

    size_t n_pass = 0;

#pragma omp parallel for reduction (+: n_pass)
    for (idx_t qi = 0; qi < n; qi++) {
        const uint8_t * q_code = q_codes + qi * pq.code_size;

        const float * dis_table_qi = dis_tables + qi * pq.M * pq.ksub;

        long * heap_ids = labels + qi * k;
        float *heap_dis = distances + qi * k;
        maxheap_heapify (k, heap_dis, heap_ids);

        if (search_type == ST_polysemous) {

            switch (pq.code_size) {
            case 4:
                n_pass += polysemous_inner_loop<HammingComputer4>
                    (*this, dis_table_qi, q_code, k, heap_dis, heap_ids);
                break;
            case 8:
                n_pass += polysemous_inner_loop<HammingComputer8>
                    (*this, dis_table_qi, q_code, k, heap_dis, heap_ids);
                break;
            case 16:
                n_pass += polysemous_inner_loop<HammingComputer16>
                    (*this, dis_table_qi, q_code, k, heap_dis, heap_ids);
                break;
            case 32:
                n_pass += polysemous_inner_loop<HammingComputer32>
                    (*this, dis_table_qi, q_code, k, heap_dis, heap_ids);
                break;
            case 20:
                n_pass += polysemous_inner_loop<HammingComputer20>
                    (*this, dis_table_qi, q_code, k, heap_dis, heap_ids);
                break;
            default:
                if (pq.code_size % 8 == 0)
                    n_pass += polysemous_inner_loop<HammingComputerM8>
                        (*this, dis_table_qi, q_code, k, heap_dis, heap_ids);
                else
                    n_pass += polysemous_inner_loop<HammingComputerM4>
                        (*this, dis_table_qi, q_code, k, heap_dis, heap_ids);
                break;
            }
        } else {
            switch (pq.code_size) {
            case 8:
                n_pass += polysemous_inner_loop<GenHammingComputer8>
                    (*this, dis_table_qi, q_code, k, heap_dis, heap_ids);
                break;
            case 16:
                n_pass += polysemous_inner_loop<GenHammingComputer16>
                    (*this, dis_table_qi, q_code, k, heap_dis, heap_ids);
                break;
            case 32:
                n_pass += polysemous_inner_loop<GenHammingComputer32>
                    (*this, dis_table_qi, q_code, k, heap_dis, heap_ids);
                break;
            default:
                n_pass += polysemous_inner_loop<GenHammingComputerM8>
                    (*this, dis_table_qi, q_code, k, heap_dis, heap_ids);
                break;
            }
        }
        maxheap_reorder (k, heap_dis, heap_ids);
    }

    indexPQ_stats.nq += n;
    indexPQ_stats.ncode += n * ntotal;
    indexPQ_stats.n_hamming_pass += n_pass;

    delete [] q_codes;
    delete [] dis_tables;

}




/*****************************************
 * Stats of IndexPQ codes
 ******************************************/




void IndexPQ::hamming_distance_table (idx_t n, const float *x,
                                      int32_t *dis) const
{
    uint8_t * q_codes = new uint8_t [n * pq.code_size];

    pq.compute_codes (x, q_codes, n);

    hammings (q_codes, codes.data(), n, ntotal, pq.code_size, dis);

    delete [] q_codes;
}


void IndexPQ::hamming_distance_histogram (idx_t n, const float *x,
                                          idx_t nb, const float *xb,
                                          long *hist)
{
    FAISS_ASSERT (metric_type == METRIC_L2);
    FAISS_ASSERT (pq.code_size % 8 == 0);
    FAISS_ASSERT (pq.byte_per_idx == 1);

    // Hamming embedding queries
    uint8_t * q_codes = new uint8_t [n * pq.code_size];
    pq.compute_codes (x, q_codes, n);

    uint8_t * b_codes ;

    if (xb) {
        b_codes = new uint8_t [nb * pq.code_size];
        pq.compute_codes (xb, b_codes, nb);
    } else {
        nb = ntotal;
        b_codes = codes.data();
    }
    int nbits = pq.M * pq.nbits;
    memset (hist, 0, sizeof(*hist) * (nbits + 1));
    size_t bs = 256;

#pragma omp parallel
    {
        std::vector<long> histi (nbits + 1);
        hamdis_t *distances = new hamdis_t [nb * bs];
#pragma omp for
        for (size_t q0 = 0; q0 < n; q0 += bs) {
            // printf ("dis stats: %ld/%ld\n", q0, n);
            size_t q1 = q0 + bs;
            if (q1 > n) q1 = n;

            hammings (q_codes + q0 * pq.code_size, b_codes,
                      q1 - q0, nb,
                      pq.code_size, distances);

            for (size_t i = 0; i < nb * (q1 - q0); i++)
                histi [distances [i]]++;
        }
#pragma omp critical
        {
            for (int i = 0; i <= nbits; i++)
                hist[i] += histi[i];
        }
        delete [] distances;
    }

    delete [] q_codes;
    if (xb)
        delete [] b_codes;

}




















/*****************************************
 * MultiIndexQuantizer
 ******************************************/



template <typename T>
struct ArgSort {
    const T * x;
    bool operator() (size_t i, size_t j) {
        return x[i] < x[j];
    }
};


/** Array that maintains a permutation of its elements so that the
 *  array's elements are sorted
 */
template <typename T>
struct SortedArray {
    const T * x;
    int N;
    std::vector<int> perm;

    explicit SortedArray (int N) {
        this->N = N;
        perm.resize (N);
    }

    void init (const T*x) {
        this->x = x;
        for (int n = 0; n < N; n++)
            perm[n] = n;
        ArgSort<T> cmp = {x };
        std::sort (perm.begin(), perm.end(), cmp);
    }

    // get smallest value
    T get_0 () {
        return x[perm[0]];
    }

    // get delta between n-smallest and n-1 -smallest
    T get_diff (int n) {
        return x[perm[n]] - x[perm[n - 1]];
    }

    // remap orders counted from smallest to indices in array
    int get_ord (int n) {
        return perm[n];
    }
};



/** Array has n values. Sort the k first ones and copy the other ones
 *  into elements k..n-1
 */
template <class C>
void partial_sort (int k, int n,
                   const typename C::T * vals, typename C::TI * perm) {
    // insert first k elts in heap
    for (int i = 1; i < k; i++) {
        indirect_heap_push<C> (i + 1, vals, perm, perm[i]);
    }

    // insert next n - k elts in heap
    for (int i = k; i < n; i++) {
        typename C::TI id = perm[i];
        typename C::TI top = perm[0];

        if (C::cmp(vals[top], vals[id])) {
            indirect_heap_pop<C> (k, vals, perm);
            indirect_heap_push<C> (k, vals, perm, id);
            perm[i] = top;
        } else {
            // nothing, elt at i is good where it is.
        }
    }

    // order the k first elements in heap
    for (int i = k - 1; i > 0; i--) {
        typename C::TI top = perm[0];
        indirect_heap_pop<C> (i + 1, vals, perm);
        perm[i] = top;
    }
}

/** same as SortedArray, but only the k first elements are sorted */
template <typename T>
struct SemiSortedArray {
    const T * x;
    int N;

    // type of the heap: CMax = sort ascending
    typedef CMax<T, int> HC;
    std::vector<int> perm;

    int k;  // k elements are sorted

    int initial_k, k_factor;

    explicit SemiSortedArray (int N) {
        this->N = N;
        perm.resize (N);
        perm.resize (N);
        initial_k = 3;
        k_factor = 4;
    }

    void init (const T*x) {
        this->x = x;
        for (int n = 0; n < N; n++)
            perm[n] = n;
        k = 0;
        grow (initial_k);
    }

    /// grow the sorted part of the array to size next_k
    void grow (int next_k) {
        if (next_k < N) {
            partial_sort<HC> (next_k - k, N - k, x, &perm[k]);
            k = next_k;
        } else { // full sort of remainder of array
            ArgSort<T> cmp = {x };
            std::sort (perm.begin() + k, perm.end(), cmp);
            k = N;
        }
    }

    // get smallest value
    T get_0 () {
        return x[perm[0]];
    }

    // get delta between n-smallest and n-1 -smallest
    T get_diff (int n) {
        if (n >= k) {
            // want to keep powers of 2 - 1
            int next_k = (k + 1) * k_factor - 1;
            grow (next_k);
        }
        return x[perm[n]] - x[perm[n - 1]];
    }

    // remap orders counted from smallest to indices in array
    int get_ord (int n) {
        assert (n < k);
        return perm[n];
    }
};



/*****************************************
 * Find the k smallest sums of M terms, where each term is taken in a
 *  table x of n values.
 *
 * A combination of terms is encoded as a scalar 0 <= t < n^M. The
 * combination t0 ... t(M-1) that correspond to the sum
 *
 *           sum = x[0, t0] + x[1, t1] + .... + x[M-1, t(M-1)]
 *
 * is encoded as
 *
 *           t = t0 + t1 * n + t2 * n^2 + ... + t(M-1) * n^(M-1)
 *
 * MinSumK is an object rather than a function, so that storage can be
 * re-used over several computations with the same sizes. use_seen is
 * good when there may be ties in the x array and it is a concern if
 * occasionally several t's are returned.
 *
 * @param x       size M * n, values to add up
 * @parms k       nb of results to retrieve
 * @param M       nb of terms
 * @param n       nb of distinct values
 * @param sums    output, size k, sorted
 * @prarm terms   output, size k, with encoding as above
 *
 ******************************************/
template <typename T, class SSA, bool use_seen>
struct MinSumK {
    int K;  ///< nb of sums to return
    int M;  ///< nb of elements to sum up
    int N;  ///< nb of possible elements for each of the M terms

    /** the heap.
     * We use a heap to maintain a queue of sums, with the associated
     * terms involved in the sum.
     */
    typedef CMin<T, long> HC;
    size_t heap_capacity, heap_size;
    T *bh_val;
    long *bh_ids;

    std::vector <SSA> ssx;
    std::vector <long> weights;

    // all results get pushed several times. When there are ties, they
    // are popped interleaved with others, so it is not easy to
    // identify them. Therefore, this bit array just marks elements
    // that were seen before.
    std::vector <uint8_t> seen;

    MinSumK (int K, int M, int N): K(K), M(M), N(N) {
        heap_capacity = K * M;
        // we'll do k steps, each step pushes at most M vals
        bh_val = new T[heap_capacity];
        bh_ids = new long[heap_capacity];

        weights.push_back (1);
        for (int m = 1; m < M; m++)
            weights.push_back(weights[m - 1] * N);

        if (use_seen) {
            long n_ids = weights.back() * N;
            seen.resize ((n_ids + 7) / 8);
        }

        for (int m = 0; m < M; m++)
            ssx.push_back (SSA(N));

    }

    bool is_seen (long i) {
        return (seen[i >> 3] >> (i & 7)) & 1;
    }

    void mark_seen (long i) {
        if (use_seen)
            seen [i >> 3] |= 1 << (i & 7);
    }

    void run (const T *x, T * sums, long * terms) {
        heap_size = 0;

        for (int m = 0; m < M; m++)
            ssx[m].init(x + N * m);

        { // intial result: take min for all elements
            T sum = 0;
            terms[0] = 0;
            mark_seen (0);
            for (int m = 0; m < M; m++) {
                sum += ssx[m].get_0();
            }
            sums[0] = sum;
            for (int m = 0; m < M; m++) {
                heap_push<HC> (++heap_size, bh_val, bh_ids,
                                      sum + ssx[m].get_diff(1),
                                      weights[m]);
            }
        }

        for (int k = 1; k < K; k++) {
            // pop smallest value from heap
            if (use_seen) {// skip already seen elements
                while (is_seen (bh_ids[0])) {
                    assert (heap_size > 0);
                    heap_pop<HC> (heap_size--, bh_val, bh_ids);
                }
            }
            assert (heap_size > 0);

            T sum = sums[k] = bh_val[0];
            long ti = terms[k] = bh_ids[0];

            if (use_seen) {
                mark_seen (ti);
                heap_pop<HC> (heap_size--, bh_val, bh_ids);
            } else {
                do {
                    heap_pop<HC> (heap_size--, bh_val, bh_ids);
                }  while (heap_size > 0 && bh_ids[0] == ti);
            }

            // enqueue followers
            long ii = ti;
            for (int m = 0; m < M; m++) {
                long n = ii % N;
                ii /= N;
                if (n + 1 >= N) continue;

                enqueue_follower (ti, m, n, sum);
            }
        }

        /*
        for (int k = 0; k < K; k++)
            for (int l = k + 1; l < K; l++)
                assert (terms[k] != terms[l]);
        */

        // convert indices by applying permutation
        for (int k = 0; k < K; k++) {
            long ii = terms[k];
            if (use_seen) {
                // clear seen for reuse at next loop
                seen[ii >> 3] = 0;
            }
            long ti = 0;
            for (int m = 0; m < M; m++) {
                long n = ii % N;
                ti += weights[m] * ssx[m].get_ord(n);
                ii /= N;
            }
            terms[k] = ti;
        }
    }


    void enqueue_follower (long ti, int m, int n, T sum) {
        T next_sum = sum + ssx[m].get_diff(n + 1);
        long next_ti   = ti + weights[m];
        heap_push<HC> (++heap_size, bh_val, bh_ids, next_sum, next_ti);
    }


    ~MinSumK () {
        delete [] bh_ids;
        delete [] bh_val;
    }
};




MultiIndexQuantizer::MultiIndexQuantizer (int d,
                     size_t M,
                     size_t nbits):
    Index(d, METRIC_L2), pq(d, M, nbits)
{
    is_trained = false;
    set_typename();
}


void MultiIndexQuantizer::set_typename()
{
    std::stringstream s;
    s << "MI_" << pq.M << "x" << pq.nbits;
    index_typename = s.str();
}


void MultiIndexQuantizer::train(idx_t n, const float *x)
{
    pq.train (n, x);
    is_trained = true;
    // count virtual elements in index
    ntotal = 1;
    for (int m = 0; m < pq.M; m++)
        ntotal *= pq.ksub;
}


void MultiIndexQuantizer::search (idx_t n, const float *x, idx_t k,
                                       float *distances, idx_t *labels) const {

    float * dis_tables = new float [n * pq.ksub * pq.M];

    pq.compute_distance_tables (n, x, dis_tables);

    /// TODO: special version for k==1

#pragma omp parallel
    {
        MinSumK <float, SemiSortedArray<float>, false> msk(k, pq.M, pq.ksub);
#pragma omp for
        for (int i = 0; i < n; i++) {
            msk.run (dis_tables + i * pq.ksub * pq.M,
                     distances + i * k, labels + i * k);

        }
    }
    delete [] dis_tables;
}


void MultiIndexQuantizer::reconstruct (idx_t key, float * recons) const
{
    if (pq.byte_per_idx == 1) {
        uint8_t code[pq.M];
        long jj = key;
        for (int m = 0; m < pq.M; m++) {
            long n = jj % pq.ksub;
            jj /= pq.ksub;
            code[m] = n;
        }
        pq.decode (code, recons);
    } else if (pq.byte_per_idx == 2) {
        uint16_t code[pq.M];
        long jj = key;
        for (int m = 0; m < pq.M; m++) {
            long n = jj % pq.ksub;
            jj /= pq.ksub;
            code[m] = n;
        }
        pq.decode ((uint8_t*)code, recons);
    } else FAISS_ASSERT(!"only 1 or 2 bytes per index supported");
}


void MultiIndexQuantizer::add (idx_t n, const float *x)
{
    FAISS_ASSERT (!"This index has virtual elements, it does not support add");
}

void MultiIndexQuantizer::reset ()
{
    FAISS_ASSERT (!"This index has virtual elements, "
                  "it does not support reset");
}




} // END namespace faiss
