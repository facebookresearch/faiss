/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexPQ.h>

#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstring>

#include <algorithm>
#include <memory>

#include <faiss/impl/DistanceComputer.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/hamming.h>

#include <faiss/impl/code_distance/code_distance.h>

namespace faiss {

/*********************************************************
 * IndexPQ implementation
 ********************************************************/

IndexPQ::IndexPQ(int d, size_t M, size_t nbits, MetricType metric)
        : IndexFlatCodes(0, d, metric), pq(d, M, nbits) {
    is_trained = false;
    do_polysemous_training = false;
    polysemous_ht = nbits * M + 1;
    search_type = ST_PQ;
    encode_signs = false;
    code_size = pq.code_size;
}

IndexPQ::IndexPQ() {
    metric_type = METRIC_L2;
    is_trained = false;
    do_polysemous_training = false;
    polysemous_ht = pq.nbits * pq.M + 1;
    search_type = ST_PQ;
    encode_signs = false;
}

void IndexPQ::train(idx_t n, const float* x) {
    if (!do_polysemous_training) { // standard training
        pq.train(n, x);
    } else {
        idx_t ntrain_perm = polysemous_training.ntrain_permutation;

        if (ntrain_perm > n / 4)
            ntrain_perm = n / 4;
        if (verbose) {
            printf("PQ training on %" PRId64 " points, remains %" PRId64
                   " points: "
                   "training polysemous on %s\n",
                   n - ntrain_perm,
                   ntrain_perm,
                   ntrain_perm == 0 ? "centroids" : "these");
        }
        pq.train(n - ntrain_perm, x);

        polysemous_training.optimize_pq_for_hamming(
                pq, ntrain_perm, x + (n - ntrain_perm) * d);
    }
    is_trained = true;
}

namespace {

template <class PQDecoder>
struct PQDistanceComputer : FlatCodesDistanceComputer {
    size_t d;
    MetricType metric;
    idx_t nb;
    const ProductQuantizer& pq;
    const float* sdc;
    std::vector<float> precomputed_table;
    size_t ndis;

    float distance_to_code(const uint8_t* code) final {
        ndis++;

        float dis = distance_single_code<PQDecoder>(
                pq.M, pq.nbits, precomputed_table.data(), code);
        return dis;
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        FAISS_THROW_IF_NOT(sdc);
        const float* sdci = sdc;
        float accu = 0;
        PQDecoder codei(codes + i * code_size, pq.nbits);
        PQDecoder codej(codes + j * code_size, pq.nbits);

        for (int l = 0; l < pq.M; l++) {
            accu += sdci[codei.decode() + (codej.decode() << codei.nbits)];
            sdci += uint64_t(1) << (2 * codei.nbits);
        }
        ndis++;
        return accu;
    }

    explicit PQDistanceComputer(const IndexPQ& storage)
            : FlatCodesDistanceComputer(
                      storage.codes.data(),
                      storage.code_size),
              pq(storage.pq) {
        precomputed_table.resize(pq.M * pq.ksub);
        nb = storage.ntotal;
        d = storage.d;
        metric = storage.metric_type;
        if (pq.sdc_table.size() == pq.ksub * pq.ksub * pq.M) {
            sdc = pq.sdc_table.data();
        } else {
            sdc = nullptr;
        }
        ndis = 0;
    }

    void set_query(const float* x) override {
        if (metric == METRIC_L2) {
            pq.compute_distance_table(x, precomputed_table.data());
        } else {
            pq.compute_inner_prod_table(x, precomputed_table.data());
        }
    }
};

} // namespace

FlatCodesDistanceComputer* IndexPQ::get_FlatCodesDistanceComputer() const {
    if (pq.nbits == 8) {
        return new PQDistanceComputer<PQDecoder8>(*this);
    } else if (pq.nbits == 16) {
        return new PQDistanceComputer<PQDecoder16>(*this);
    } else {
        return new PQDistanceComputer<PQDecoderGeneric>(*this);
    }
}

/*****************************************
 * IndexPQ polysemous search routines
 ******************************************/

void IndexPQ::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* iparams) const {
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(is_trained);

    const SearchParametersPQ* params = nullptr;
    Search_type_t param_search_type = this->search_type;

    if (iparams) {
        params = dynamic_cast<const SearchParametersPQ*>(iparams);
        FAISS_THROW_IF_NOT_MSG(params, "invalid search params");
        FAISS_THROW_IF_NOT_MSG(!params->sel, "selector not supported");
        param_search_type = params->search_type;
    }

    if (param_search_type == ST_PQ) { // Simple PQ search

        if (metric_type == METRIC_L2) {
            float_maxheap_array_t res = {
                    size_t(n), size_t(k), labels, distances};
            pq.search(x, n, codes.data(), ntotal, &res, true);
        } else {
            float_minheap_array_t res = {
                    size_t(n), size_t(k), labels, distances};
            pq.search_ip(x, n, codes.data(), ntotal, &res, true);
        }
        indexPQ_stats.nq += n;
        indexPQ_stats.ncode += n * ntotal;

    } else if (
            param_search_type == ST_polysemous ||
            param_search_type == ST_polysemous_generalize) {
        FAISS_THROW_IF_NOT(metric_type == METRIC_L2);
        int param_polysemous_ht =
                params ? params->polysemous_ht : this->polysemous_ht;
        search_core_polysemous(
                n,
                x,
                k,
                distances,
                labels,
                param_polysemous_ht,
                param_search_type == ST_polysemous_generalize);

    } else { // code-to-code distances

        std::unique_ptr<uint8_t[]> q_codes(new uint8_t[n * pq.code_size]);

        if (!encode_signs) {
            pq.compute_codes(x, q_codes.get(), n);
        } else {
            FAISS_THROW_IF_NOT(d == pq.nbits * pq.M);
            memset(q_codes.get(), 0, n * pq.code_size);
            for (size_t i = 0; i < n; i++) {
                const float* xi = x + i * d;
                uint8_t* code = q_codes.get() + i * pq.code_size;
                for (int j = 0; j < d; j++)
                    if (xi[j] > 0)
                        code[j >> 3] |= 1 << (j & 7);
            }
        }

        if (param_search_type == ST_SDC) {
            float_maxheap_array_t res = {
                    size_t(n), size_t(k), labels, distances};

            pq.search_sdc(q_codes.get(), n, codes.data(), ntotal, &res, true);

        } else {
            std::unique_ptr<int[]> idistances(new int[n * k]);

            int_maxheap_array_t res = {
                    size_t(n), size_t(k), labels, idistances.get()};

            if (param_search_type == ST_HE) {
                hammings_knn_hc(
                        &res,
                        q_codes.get(),
                        codes.data(),
                        ntotal,
                        pq.code_size,
                        true);

            } else if (param_search_type == ST_generalized_HE) {
                generalized_hammings_knn_hc(
                        &res,
                        q_codes.get(),
                        codes.data(),
                        ntotal,
                        pq.code_size,
                        true);
            }

            // convert distances to floats
            for (int i = 0; i < k * n; i++)
                distances[i] = idistances[i];
        }

        indexPQ_stats.nq += n;
        indexPQ_stats.ncode += n * ntotal;
    }
}

void IndexPQStats::reset() {
    nq = ncode = n_hamming_pass = 0;
}

IndexPQStats indexPQ_stats;

namespace {

template <class HammingComputer>
size_t polysemous_inner_loop(
        const IndexPQ* index,
        const float* dis_table_qi,
        const uint8_t* q_code,
        size_t k,
        float* heap_dis,
        int64_t* heap_ids,
        int ht) {
    int M = index->pq.M;
    int code_size = index->pq.code_size;
    int ksub = index->pq.ksub;
    size_t ntotal = index->ntotal;

    const uint8_t* b_code = index->codes.data();

    size_t n_pass_i = 0;

    HammingComputer hc(q_code, code_size);

    for (int64_t bi = 0; bi < ntotal; bi++) {
        int hd = hc.hamming(b_code);

        if (hd < ht) {
            n_pass_i++;

            float dis = 0;
            const float* dis_table = dis_table_qi;
            for (int m = 0; m < M; m++) {
                dis += dis_table[b_code[m]];
                dis_table += ksub;
            }

            if (dis < heap_dis[0]) {
                maxheap_replace_top(k, heap_dis, heap_ids, dis, bi);
            }
        }
        b_code += code_size;
    }
    return n_pass_i;
}

struct Run_polysemous_inner_loop {
    using T = size_t;
    template <class HammingComputer, class... Types>
    size_t f(Types... args) {
        return polysemous_inner_loop<HammingComputer>(args...);
    }
};

} // anonymous namespace

void IndexPQ::search_core_polysemous(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        int param_polysemous_ht,
        bool generalized_hamming) const {
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(pq.nbits == 8);

    if (param_polysemous_ht == 0) {
        param_polysemous_ht = pq.nbits * pq.M + 1;
    }

    // PQ distance tables
    std::unique_ptr<float[]> dis_tables(new float[n * pq.ksub * pq.M]);
    pq.compute_distance_tables(n, x, dis_tables.get());

    // Hamming embedding queries
    std::unique_ptr<uint8_t[]> q_codes(new uint8_t[n * pq.code_size]);

    if (false) {
        pq.compute_codes(x, q_codes.get(), n);
    } else {
#pragma omp parallel for
        for (idx_t qi = 0; qi < n; qi++) {
            pq.compute_code_from_distance_table(
                    dis_tables.get() + qi * pq.M * pq.ksub,
                    q_codes.get() + qi * pq.code_size);
        }
    }

    size_t n_pass = 0;

    int bad_code_size = 0;

#pragma omp parallel for reduction(+ : n_pass, bad_code_size)
    for (idx_t qi = 0; qi < n; qi++) {
        const uint8_t* q_code = q_codes.get() + qi * pq.code_size;

        const float* dis_table_qi = dis_tables.get() + qi * pq.M * pq.ksub;

        int64_t* heap_ids = labels + qi * k;
        float* heap_dis = distances + qi * k;
        maxheap_heapify(k, heap_dis, heap_ids);

        if (!generalized_hamming) {
            Run_polysemous_inner_loop r;
            n_pass += dispatch_HammingComputer(
                    pq.code_size,
                    r,
                    this,
                    dis_table_qi,
                    q_code,
                    k,
                    heap_dis,
                    heap_ids,
                    param_polysemous_ht);

        } else { // generalized hamming
            switch (pq.code_size) {
#define DISPATCH(cs)                                             \
    case cs:                                                     \
        n_pass += polysemous_inner_loop<GenHammingComputer##cs>( \
                this,                                            \
                dis_table_qi,                                    \
                q_code,                                          \
                k,                                               \
                heap_dis,                                        \
                heap_ids,                                        \
                param_polysemous_ht);                            \
        break;
                DISPATCH(8)
                DISPATCH(16)
                DISPATCH(32)
                default:
                    if (pq.code_size % 8 == 0) {
                        n_pass += polysemous_inner_loop<GenHammingComputerM8>(
                                this,
                                dis_table_qi,
                                q_code,
                                k,
                                heap_dis,
                                heap_ids,
                                param_polysemous_ht);
                    } else {
                        bad_code_size++;
                    }
                    break;
#undef DISPATCH
            }
        }
        maxheap_reorder(k, heap_dis, heap_ids);
    }

    if (bad_code_size) {
        FAISS_THROW_FMT(
                "code size %zd not supported for polysemous", pq.code_size);
    }

    indexPQ_stats.nq += n;
    indexPQ_stats.ncode += n * ntotal;
    indexPQ_stats.n_hamming_pass += n_pass;
}

/* The standalone codec interface (just remaps to the PQ functions) */

void IndexPQ::sa_encode(idx_t n, const float* x, uint8_t* bytes) const {
    pq.compute_codes(x, bytes, n);
}

void IndexPQ::sa_decode(idx_t n, const uint8_t* bytes, float* x) const {
    pq.decode(bytes, x, n);
}

/*****************************************
 * Stats of IndexPQ codes
 ******************************************/

void IndexPQ::hamming_distance_table(idx_t n, const float* x, int32_t* dis)
        const {
    std::unique_ptr<uint8_t[]> q_codes(new uint8_t[n * pq.code_size]);

    pq.compute_codes(x, q_codes.get(), n);

    hammings(q_codes.get(), codes.data(), n, ntotal, pq.code_size, dis);
}

void IndexPQ::hamming_distance_histogram(
        idx_t n,
        const float* x,
        idx_t nb,
        const float* xb,
        int64_t* hist) {
    FAISS_THROW_IF_NOT(metric_type == METRIC_L2);
    FAISS_THROW_IF_NOT(pq.code_size % 8 == 0);
    FAISS_THROW_IF_NOT(pq.nbits == 8);

    // Hamming embedding queries
    std::unique_ptr<uint8_t[]> q_codes(new uint8_t[n * pq.code_size]);
    pq.compute_codes(x, q_codes.get(), n);

    uint8_t* b_codes;
    std::unique_ptr<uint8_t[]> del_b_codes;

    if (xb) {
        b_codes = new uint8_t[nb * pq.code_size];
        del_b_codes.reset(b_codes);
        pq.compute_codes(xb, b_codes, nb);
    } else {
        nb = ntotal;
        b_codes = codes.data();
    }
    int nbits = pq.M * pq.nbits;
    memset(hist, 0, sizeof(*hist) * (nbits + 1));
    size_t bs = 256;

#pragma omp parallel
    {
        std::vector<int64_t> histi(nbits + 1);
        std::unique_ptr<hamdis_t[]> distances(new hamdis_t[nb * bs]);
#pragma omp for
        for (idx_t q0 = 0; q0 < n; q0 += bs) {
            // printf ("dis stats: %zd/%zd\n", q0, n);
            size_t q1 = q0 + bs;
            if (q1 > n)
                q1 = n;

            hammings(
                    q_codes.get() + q0 * pq.code_size,
                    b_codes,
                    q1 - q0,
                    nb,
                    pq.code_size,
                    distances.get());

            for (size_t i = 0; i < nb * (q1 - q0); i++)
                histi[distances[i]]++;
        }
#pragma omp critical
        {
            for (int i = 0; i <= nbits; i++)
                hist[i] += histi[i];
        }
    }
}

/*****************************************
 * MultiIndexQuantizer
 ******************************************/

namespace {

template <typename T>
struct PreSortedArray {
    const T* x;
    int N;

    explicit PreSortedArray(int N) : N(N) {}
    void init(const T* x_2) {
        this->x = x_2;
    }
    // get smallest value
    T get_0() {
        return x[0];
    }

    // get delta between n-smallest and n-1 -smallest
    T get_diff(int n) {
        return x[n] - x[n - 1];
    }

    // remap orders counted from smallest to indices in array
    int get_ord(int n) {
        return n;
    }
};

template <typename T>
struct ArgSort {
    const T* x;
    bool operator()(size_t i, size_t j) {
        return x[i] < x[j];
    }
};

/** Array that maintains a permutation of its elements so that the
 *  array's elements are sorted
 */
template <typename T>
struct SortedArray {
    const T* x;
    int N;
    std::vector<int> perm;

    explicit SortedArray(int N) {
        this->N = N;
        perm.resize(N);
    }

    void init(const T* x_2) {
        this->x = x_2;
        for (int n = 0; n < N; n++)
            perm[n] = n;
        ArgSort<T> cmp = {x_2};
        std::sort(perm.begin(), perm.end(), cmp);
    }

    // get smallest value
    T get_0() {
        return x[perm[0]];
    }

    // get delta between n-smallest and n-1 -smallest
    T get_diff(int n) {
        return x[perm[n]] - x[perm[n - 1]];
    }

    // remap orders counted from smallest to indices in array
    int get_ord(int n) {
        return perm[n];
    }
};

/** Array has n values. Sort the k first ones and copy the other ones
 *  into elements k..n-1
 */
template <class C>
void partial_sort(
        int k,
        int n,
        const typename C::T* vals,
        typename C::TI* perm) {
    // insert first k elts in heap
    for (int i = 1; i < k; i++) {
        indirect_heap_push<C>(i + 1, vals, perm, perm[i]);
    }

    // insert next n - k elts in heap
    for (int i = k; i < n; i++) {
        typename C::TI id = perm[i];
        typename C::TI top = perm[0];

        if (C::cmp(vals[top], vals[id])) {
            indirect_heap_pop<C>(k, vals, perm);
            indirect_heap_push<C>(k, vals, perm, id);
            perm[i] = top;
        } else {
            // nothing, elt at i is good where it is.
        }
    }

    // order the k first elements in heap
    for (int i = k - 1; i > 0; i--) {
        typename C::TI top = perm[0];
        indirect_heap_pop<C>(i + 1, vals, perm);
        perm[i] = top;
    }
}

/** same as SortedArray, but only the k first elements are sorted */
template <typename T>
struct SemiSortedArray {
    const T* x;
    int N;

    // type of the heap: CMax = sort ascending
    using HC = CMax<T, int>;
    std::vector<int> perm;

    int k; // k elements are sorted

    int initial_k, k_factor;

    explicit SemiSortedArray(int N) {
        this->N = N;
        perm.resize(N);
        perm.resize(N);
        initial_k = 3;
        k_factor = 4;
    }

    void init(const T* x_2) {
        this->x = x_2;
        for (int n = 0; n < N; n++)
            perm[n] = n;
        k = 0;
        grow(initial_k);
    }

    /// grow the sorted part of the array to size next_k
    void grow(int next_k) {
        if (next_k < N) {
            partial_sort<HC>(next_k - k, N - k, x, &perm[k]);
            k = next_k;
        } else { // full sort of remainder of array
            ArgSort<T> cmp = {x};
            std::sort(perm.begin() + k, perm.end(), cmp);
            k = N;
        }
    }

    // get smallest value
    T get_0() {
        return x[perm[0]];
    }

    // get delta between n-smallest and n-1 -smallest
    T get_diff(int n) {
        if (n >= k) {
            // want to keep powers of 2 - 1
            int next_k = (k + 1) * k_factor - 1;
            grow(next_k);
        }
        return x[perm[n]] - x[perm[n - 1]];
    }

    // remap orders counted from smallest to indices in array
    int get_ord(int n) {
        assert(n < k);
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
 * @param k       nb of results to retrieve
 * @param M       nb of terms
 * @param n       nb of distinct values
 * @param sums    output, size k, sorted
 * @param terms   output, size k, with encoding as above
 *
 ******************************************/
template <typename T, class SSA, bool use_seen>
struct MinSumK {
    int K;    ///< nb of sums to return
    int M;    ///< nb of elements to sum up
    int nbit; ///< nb of bits to encode one entry
    int N;    ///< nb of possible elements for each of the M terms

    /** the heap.
     * We use a heap to maintain a queue of sums, with the associated
     * terms involved in the sum.
     */
    using HC = CMin<T, int64_t>;
    size_t heap_capacity, heap_size;
    T* bh_val;
    int64_t* bh_ids;

    std::vector<SSA> ssx;

    // all results get pushed several times. When there are ties, they
    // are popped interleaved with others, so it is not easy to
    // identify them. Therefore, this bit array just marks elements
    // that were seen before.
    std::vector<uint8_t> seen;

    MinSumK(int K, int M, int nbit, int N) : K(K), M(M), nbit(nbit), N(N) {
        heap_capacity = K * M;
        assert(N <= (1 << nbit));

        // we'll do k steps, each step pushes at most M vals
        bh_val = new T[heap_capacity];
        bh_ids = new int64_t[heap_capacity];

        if (use_seen) {
            int64_t n_ids = weight(M);
            seen.resize((n_ids + 7) / 8);
        }

        for (int m = 0; m < M; m++)
            ssx.push_back(SSA(N));
    }

    int64_t weight(int i) {
        return 1 << (i * nbit);
    }

    bool is_seen(int64_t i) {
        return (seen[i >> 3] >> (i & 7)) & 1;
    }

    void mark_seen(int64_t i) {
        if (use_seen)
            seen[i >> 3] |= 1 << (i & 7);
    }

    void run(const T* x, int64_t ldx, T* sums, int64_t* terms) {
        heap_size = 0;

        for (int m = 0; m < M; m++) {
            ssx[m].init(x);
            x += ldx;
        }

        { // initial result: take min for all elements
            T sum = 0;
            terms[0] = 0;
            mark_seen(0);
            for (int m = 0; m < M; m++) {
                sum += ssx[m].get_0();
            }
            sums[0] = sum;
            for (int m = 0; m < M; m++) {
                heap_push<HC>(
                        ++heap_size,
                        bh_val,
                        bh_ids,
                        sum + ssx[m].get_diff(1),
                        weight(m));
            }
        }

        for (int k = 1; k < K; k++) {
            // pop smallest value from heap
            if (use_seen) { // skip already seen elements
                while (is_seen(bh_ids[0])) {
                    assert(heap_size > 0);
                    heap_pop<HC>(heap_size--, bh_val, bh_ids);
                }
            }
            assert(heap_size > 0);

            T sum = sums[k] = bh_val[0];
            int64_t ti = terms[k] = bh_ids[0];

            if (use_seen) {
                mark_seen(ti);
                heap_pop<HC>(heap_size--, bh_val, bh_ids);
            } else {
                do {
                    heap_pop<HC>(heap_size--, bh_val, bh_ids);
                } while (heap_size > 0 && bh_ids[0] == ti);
            }

            // enqueue followers
            int64_t ii = ti;
            for (int m = 0; m < M; m++) {
                int64_t n = ii & (((int64_t)1 << nbit) - 1);
                ii >>= nbit;
                if (n + 1 >= N)
                    continue;

                enqueue_follower(ti, m, n, sum);
            }
        }

        /*
        for (int k = 0; k < K; k++)
            for (int l = k + 1; l < K; l++)
                assert (terms[k] != terms[l]);
        */

        // convert indices by applying permutation
        for (int k = 0; k < K; k++) {
            int64_t ii = terms[k];
            if (use_seen) {
                // clear seen for reuse at next loop
                seen[ii >> 3] = 0;
            }
            int64_t ti = 0;
            for (int m = 0; m < M; m++) {
                int64_t n = ii & (((int64_t)1 << nbit) - 1);
                ti += int64_t(ssx[m].get_ord(n)) << (nbit * m);
                ii >>= nbit;
            }
            terms[k] = ti;
        }
    }

    void enqueue_follower(int64_t ti, int m, int n, T sum) {
        T next_sum = sum + ssx[m].get_diff(n + 1);
        int64_t next_ti = ti + weight(m);
        heap_push<HC>(++heap_size, bh_val, bh_ids, next_sum, next_ti);
    }

    ~MinSumK() {
        delete[] bh_ids;
        delete[] bh_val;
    }
};

} // anonymous namespace

MultiIndexQuantizer::MultiIndexQuantizer(int d, size_t M, size_t nbits)
        : Index(d, METRIC_L2), pq(d, M, nbits) {
    is_trained = false;
    pq.verbose = verbose;
}

void MultiIndexQuantizer::train(idx_t n, const float* x) {
    pq.verbose = verbose;
    pq.train(n, x);
    is_trained = true;
    // count virtual elements in index
    ntotal = 1;
    for (int m = 0; m < pq.M; m++)
        ntotal *= pq.ksub;
}

// block size used in MultiIndexQuantizer::search
int multi_index_quantizer_search_bs = 32768;

void MultiIndexQuantizer::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(
            !params, "search params not supported for this index");
    if (n == 0) {
        return;
    }
    FAISS_THROW_IF_NOT(k > 0);

    // the allocation just below can be severe...
    idx_t bs = multi_index_quantizer_search_bs;
    if (n > bs) {
        for (idx_t i0 = 0; i0 < n; i0 += bs) {
            idx_t i1 = std::min(i0 + bs, n);
            if (verbose) {
                printf("MultiIndexQuantizer::search: %" PRId64 ":%" PRId64
                       " / %" PRId64 "\n",
                       i0,
                       i1,
                       n);
            }
            search(i1 - i0, x + i0 * d, k, distances + i0 * k, labels + i0 * k);
        }
        return;
    }

    std::unique_ptr<float[]> dis_tables(new float[n * pq.ksub * pq.M]);

    pq.compute_distance_tables(n, x, dis_tables.get());

    if (k == 1) {
        // simple version that just finds the min in each table

#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            const float* dis_table = dis_tables.get() + i * pq.ksub * pq.M;
            float dis = 0;
            idx_t label = 0;

            for (int s = 0; s < pq.M; s++) {
                float vmin = HUGE_VALF;
                idx_t lmin = -1;

                for (idx_t j = 0; j < pq.ksub; j++) {
                    if (dis_table[j] < vmin) {
                        vmin = dis_table[j];
                        lmin = j;
                    }
                }
                dis += vmin;
                label |= lmin << (s * pq.nbits);
                dis_table += pq.ksub;
            }

            distances[i] = dis;
            labels[i] = label;
        }

    } else {
#pragma omp parallel if (n > 1)
        {
            MinSumK<float, SemiSortedArray<float>, false> msk(
                    k, pq.M, pq.nbits, pq.ksub);
#pragma omp for
            for (int i = 0; i < n; i++) {
                msk.run(dis_tables.get() + i * pq.ksub * pq.M,
                        pq.ksub,
                        distances + i * k,
                        labels + i * k);
            }
        }
    }
}

void MultiIndexQuantizer::reconstruct(idx_t key, float* recons) const {
    int64_t jj = key;
    for (int m = 0; m < pq.M; m++) {
        int64_t n = jj & (((int64_t)1 << pq.nbits) - 1);
        jj >>= pq.nbits;
        memcpy(recons, pq.get_centroids(m, n), sizeof(recons[0]) * pq.dsub);
        recons += pq.dsub;
    }
}

void MultiIndexQuantizer::add(idx_t /*n*/, const float* /*x*/) {
    FAISS_THROW_MSG(
            "This index has virtual elements, "
            "it does not support add");
}

void MultiIndexQuantizer::reset() {
    FAISS_THROW_MSG(
            "This index has virtual elements, "
            "it does not support reset");
}

/*****************************************
 * MultiIndexQuantizer2
 ******************************************/

MultiIndexQuantizer2::MultiIndexQuantizer2(
        int d,
        size_t M,
        size_t nbits,
        Index** indexes)
        : MultiIndexQuantizer(d, M, nbits) {
    assign_indexes.resize(M);
    for (int i = 0; i < M; i++) {
        FAISS_THROW_IF_NOT_MSG(
                indexes[i]->d == pq.dsub,
                "Provided sub-index has incorrect size");
        assign_indexes[i] = indexes[i];
    }
    own_fields = false;
}

MultiIndexQuantizer2::MultiIndexQuantizer2(
        int d,
        size_t nbits,
        Index* assign_index_0,
        Index* assign_index_1)
        : MultiIndexQuantizer(d, 2, nbits) {
    FAISS_THROW_IF_NOT_MSG(
            assign_index_0->d == pq.dsub && assign_index_1->d == pq.dsub,
            "Provided sub-index has incorrect size");
    assign_indexes.resize(2);
    assign_indexes[0] = assign_index_0;
    assign_indexes[1] = assign_index_1;
    own_fields = false;
}

void MultiIndexQuantizer2::train(idx_t n, const float* x) {
    MultiIndexQuantizer::train(n, x);
    // add centroids to sub-indexes
    for (int i = 0; i < pq.M; i++) {
        assign_indexes[i]->add(pq.ksub, pq.get_centroids(i, 0));
    }
}

void MultiIndexQuantizer2::search(
        idx_t n,
        const float* x,
        idx_t K,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(
            !params, "search params not supported for this index");

    if (n == 0) {
        return;
    }

    int k2 = std::min(K, int64_t(pq.ksub));
    FAISS_THROW_IF_NOT(k2);

    int64_t M = pq.M;
    int64_t dsub = pq.dsub, ksub = pq.ksub;

    // size (M, n, k2)
    std::vector<idx_t> sub_ids(n * M * k2);
    std::vector<float> sub_dis(n * M * k2);
    std::vector<float> xsub(n * dsub);

    for (int m = 0; m < M; m++) {
        float* xdest = xsub.data();
        const float* xsrc = x + m * dsub;
        for (int j = 0; j < n; j++) {
            memcpy(xdest, xsrc, dsub * sizeof(xdest[0]));
            xsrc += d;
            xdest += dsub;
        }

        assign_indexes[m]->search(
                n, xsub.data(), k2, &sub_dis[k2 * n * m], &sub_ids[k2 * n * m]);
    }

    if (K == 1) {
        // simple version that just finds the min in each table
        assert(k2 == 1);

        for (int i = 0; i < n; i++) {
            float dis = 0;
            idx_t label = 0;

            for (int m = 0; m < M; m++) {
                float vmin = sub_dis[i + m * n];
                idx_t lmin = sub_ids[i + m * n];
                dis += vmin;
                label |= lmin << (m * pq.nbits);
            }
            distances[i] = dis;
            labels[i] = label;
        }

    } else {
#pragma omp parallel if (n > 1)
        {
            MinSumK<float, PreSortedArray<float>, false> msk(
                    K, pq.M, pq.nbits, k2);
#pragma omp for
            for (int i = 0; i < n; i++) {
                idx_t* li = labels + i * K;
                msk.run(&sub_dis[i * k2], k2 * n, distances + i * K, li);

                // remap ids

                const idx_t* idmap0 = sub_ids.data() + i * k2;
                int64_t ld_idmap = k2 * n;
                int64_t mask1 = ksub - (int64_t)1;

                for (int k = 0; k < K; k++) {
                    const idx_t* idmap = idmap0;
                    int64_t vin = li[k];
                    int64_t vout = 0;
                    int bs = 0;
                    for (int m = 0; m < M; m++) {
                        int64_t s = vin & mask1;
                        vin >>= pq.nbits;
                        vout |= idmap[s] << bs;
                        bs += pq.nbits;
                        idmap += ld_idmap;
                    }
                    li[k] = vout;
                }
            }
        }
    }
}

} // namespace faiss
