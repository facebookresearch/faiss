/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/utils/sorting.h>

#include <omp.h>
#include <algorithm>

#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/utils.h>

namespace faiss {

/*****************************************************************************
 * Argsort
 ****************************************************************************/

namespace {
struct ArgsortComparator {
    const float* vals;
    bool operator()(const size_t a, const size_t b) const {
        return vals[a] < vals[b];
    }
};

struct SegmentS {
    size_t i0; // begin pointer in the permutation array
    size_t i1; // end
    size_t len() const {
        return i1 - i0;
    }
};

// see https://en.wikipedia.org/wiki/Merge_algorithm#Parallel_merge
// extended to > 1 merge thread

// merges 2 ranges that should be consecutive on the source into
// the union of the two on the destination
template <typename T>
void parallel_merge(
        const T* src,
        T* dst,
        SegmentS& s1,
        SegmentS& s2,
        int nt,
        const ArgsortComparator& comp) {
    if (s2.len() > s1.len()) { // make sure that s1 larger than s2
        std::swap(s1, s2);
    }

    // compute sub-ranges for each thread
    std::vector<SegmentS> s1s(nt), s2s(nt), sws(nt);
    s2s[0].i0 = s2.i0;
    s2s[nt - 1].i1 = s2.i1;

    // not sure parallel actually helps here
#pragma omp parallel for num_threads(nt)
    for (int t = 0; t < nt; t++) {
        s1s[t].i0 = s1.i0 + s1.len() * t / nt;
        s1s[t].i1 = s1.i0 + s1.len() * (t + 1) / nt;

        if (t + 1 < nt) {
            T pivot = src[s1s[t].i1];
            size_t i0 = s2.i0, i1 = s2.i1;
            while (i0 + 1 < i1) {
                size_t imed = (i1 + i0) / 2;
                if (comp(pivot, src[imed])) {
                    i1 = imed;
                } else {
                    i0 = imed;
                }
            }
            s2s[t].i1 = s2s[t + 1].i0 = i1;
        }
    }
    s1.i0 = std::min(s1.i0, s2.i0);
    s1.i1 = std::max(s1.i1, s2.i1);
    s2 = s1;
    sws[0].i0 = s1.i0;
    for (int t = 0; t < nt; t++) {
        sws[t].i1 = sws[t].i0 + s1s[t].len() + s2s[t].len();
        if (t + 1 < nt) {
            sws[t + 1].i0 = sws[t].i1;
        }
    }
    assert(sws[nt - 1].i1 == s1.i1);

    // do the actual merging
#pragma omp parallel for num_threads(nt)
    for (int t = 0; t < nt; t++) {
        SegmentS sw = sws[t];
        SegmentS s1t = s1s[t];
        SegmentS s2t = s2s[t];
        if (s1t.i0 < s1t.i1 && s2t.i0 < s2t.i1) {
            for (;;) {
                // assert (sw.len() == s1t.len() + s2t.len());
                if (comp(src[s1t.i0], src[s2t.i0])) {
                    dst[sw.i0++] = src[s1t.i0++];
                    if (s1t.i0 == s1t.i1) {
                        break;
                    }
                } else {
                    dst[sw.i0++] = src[s2t.i0++];
                    if (s2t.i0 == s2t.i1) {
                        break;
                    }
                }
            }
        }
        if (s1t.len() > 0) {
            assert(s1t.len() == sw.len());
            memcpy(dst + sw.i0, src + s1t.i0, s1t.len() * sizeof(dst[0]));
        } else if (s2t.len() > 0) {
            assert(s2t.len() == sw.len());
            memcpy(dst + sw.i0, src + s2t.i0, s2t.len() * sizeof(dst[0]));
        }
    }
}

}; // namespace

void fvec_argsort(size_t n, const float* vals, size_t* perm) {
    for (size_t i = 0; i < n; i++) {
        perm[i] = i;
    }
    ArgsortComparator comp = {vals};
    std::sort(perm, perm + n, comp);
}

void fvec_argsort_parallel(size_t n, const float* vals, size_t* perm) {
    size_t* perm2 = new size_t[n];
    // 2 result tables, during merging, flip between them
    size_t *permB = perm2, *permA = perm;

    int nt = omp_get_max_threads();
    { // prepare correct permutation so that the result ends in perm
      // at final iteration
        int nseg = nt;
        while (nseg > 1) {
            nseg = (nseg + 1) / 2;
            std::swap(permA, permB);
        }
    }

#pragma omp parallel
    for (size_t i = 0; i < n; i++) {
        permA[i] = i;
    }

    ArgsortComparator comp = {vals};

    std::vector<SegmentS> segs(nt);

    // independent sorts
#pragma omp parallel for
    for (int t = 0; t < nt; t++) {
        size_t i0 = t * n / nt;
        size_t i1 = (t + 1) * n / nt;
        SegmentS seg = {i0, i1};
        std::sort(permA + seg.i0, permA + seg.i1, comp);
        segs[t] = seg;
    }
    int prev_nested = omp_get_nested();
    omp_set_nested(1);

    int nseg = nt;
    while (nseg > 1) {
        int nseg1 = (nseg + 1) / 2;
        int sub_nt = nseg % 2 == 0 ? nt : nt - 1;
        int sub_nseg1 = nseg / 2;

#pragma omp parallel for num_threads(nseg1)
        for (int s = 0; s < nseg; s += 2) {
            if (s + 1 == nseg) { // otherwise isolated segment
                memcpy(permB + segs[s].i0,
                       permA + segs[s].i0,
                       segs[s].len() * sizeof(size_t));
            } else {
                int t0 = s * sub_nt / sub_nseg1;
                int t1 = (s + 1) * sub_nt / sub_nseg1;
                printf("merge %d %d, %d threads\n", s, s + 1, t1 - t0);
                parallel_merge(
                        permA, permB, segs[s], segs[s + 1], t1 - t0, comp);
            }
        }
        for (int s = 0; s < nseg; s += 2) {
            segs[s / 2] = segs[s];
        }
        nseg = nseg1;
        std::swap(permA, permB);
    }
    assert(permA == perm);
    omp_set_nested(prev_nested);
    delete[] perm2;
}

/*****************************************************************************
 * Bucket sort
 ****************************************************************************/

// extern symbol in the .h
int bucket_sort_verbose = 0;

namespace {

void bucket_sort_ref(
        size_t nval,
        const uint64_t* vals,
        uint64_t vmax,
        int64_t* lims,
        int64_t* perm) {
    double t0 = getmillisecs();
    memset(lims, 0, sizeof(*lims) * (vmax + 1));
    for (size_t i = 0; i < nval; i++) {
        FAISS_THROW_IF_NOT(vals[i] < vmax);
        lims[vals[i] + 1]++;
    }
    double t1 = getmillisecs();
    // compute cumulative sum
    for (size_t i = 0; i < vmax; i++) {
        lims[i + 1] += lims[i];
    }
    FAISS_THROW_IF_NOT(lims[vmax] == nval);
    double t2 = getmillisecs();
    // populate buckets
    for (size_t i = 0; i < nval; i++) {
        perm[lims[vals[i]]++] = i;
    }
    double t3 = getmillisecs();
    // reset pointers
    for (size_t i = vmax; i > 0; i--) {
        lims[i] = lims[i - 1];
    }
    lims[0] = 0;
    double t4 = getmillisecs();
    if (bucket_sort_verbose) {
        printf("times %.3f %.3f %.3f %.3f\n",
               t1 - t0,
               t2 - t1,
               t3 - t2,
               t4 - t3);
    }
}

void bucket_sort_parallel(
        size_t nval,
        const uint64_t* vals,
        uint64_t vmax,
        int64_t* lims,
        int64_t* perm,
        int nt_in) {
    memset(lims, 0, sizeof(*lims) * (vmax + 1));
#pragma omp parallel num_threads(nt_in)
    {
        int nt = omp_get_num_threads(); // might be different from nt_in
        int rank = omp_get_thread_num();
        std::vector<int64_t> local_lims(vmax + 1);

        // range of indices handled by this thread
        size_t i0 = nval * rank / nt;
        size_t i1 = nval * (rank + 1) / nt;

        // build histogram in local lims
        double t0 = getmillisecs();
        for (size_t i = i0; i < i1; i++) {
            local_lims[vals[i]]++;
        }
#pragma omp critical
        { // accumulate histograms (not shifted indices to prepare cumsum)
            for (size_t i = 0; i < vmax; i++) {
                lims[i + 1] += local_lims[i];
            }
        }
#pragma omp barrier

        double t1 = getmillisecs();
#pragma omp master
        {
            // compute cumulative sum
            for (size_t i = 0; i < vmax; i++) {
                lims[i + 1] += lims[i];
            }
            FAISS_THROW_IF_NOT(lims[vmax] == nval);
        }
#pragma omp barrier

#pragma omp critical
        { // current thread grabs a slot in the buckets
            for (size_t i = 0; i < vmax; i++) {
                size_t nv = local_lims[i];
                local_lims[i] = lims[i]; // where we should start writing
                lims[i] += nv;
            }
        }

        double t2 = getmillisecs();
#pragma omp barrier
        { // populate buckets, this is the slowest operation
            for (size_t i = i0; i < i1; i++) {
                perm[local_lims[vals[i]]++] = i;
            }
        }
#pragma omp barrier
        double t3 = getmillisecs();

#pragma omp master
        { // shift back lims
            for (size_t i = vmax; i > 0; i--) {
                lims[i] = lims[i - 1];
            }
            lims[0] = 0;
            double t4 = getmillisecs();
            if (bucket_sort_verbose) {
                printf("times %.3f %.3f %.3f %.3f\n",
                       t1 - t0,
                       t2 - t1,
                       t3 - t2,
                       t4 - t3);
            }
        }
    }
}

/***********************************************
 * in-place bucket sort
 */

void bucket_sort_inplace_ref(
        size_t nrow,
        size_t ncol,
        int32_t* vals,
        int32_t nbucket,
        int64_t* lims) {
    double t0 = getmillisecs();
    size_t nval = nrow * ncol;
    FAISS_THROW_IF_NOT(
            nbucket < nval); // unclear what would happen in this case...

    memset(lims, 0, sizeof(*lims) * (nbucket + 1));
    for (size_t i = 0; i < nval; i++) {
        FAISS_THROW_IF_NOT(vals[i] < nbucket);
        lims[vals[i] + 1]++;
    }
    double t1 = getmillisecs();
    // compute cumulative sum
    for (size_t i = 0; i < nbucket; i++) {
        lims[i + 1] += lims[i];
    }
    FAISS_THROW_IF_NOT(lims[nbucket] == nval);
    double t2 = getmillisecs();

    std::vector<size_t> ptrs(nbucket);
    for (size_t i = 0; i < nbucket; i++) {
        ptrs[i] = lims[i];
    }

    // find loops in the permutation and follow them
    int32_t row = -1;
    int32_t init_bucket_no = 0, bucket_no = 0;
    for (;;) {
        size_t idx = ptrs[bucket_no];
        if (row >= 0) {
            ptrs[bucket_no] += 1;
        }
        assert(idx < lims[bucket_no + 1]);
        int32_t next_bucket_no = vals[idx];
        vals[idx] = row;
        if (next_bucket_no != -1) {
            row = idx / ncol;
            bucket_no = next_bucket_no;
        } else {
            // start new loop
            for (; init_bucket_no < nbucket; init_bucket_no++) {
                if (ptrs[init_bucket_no] < lims[init_bucket_no + 1]) {
                    break;
                }
            }
            if (init_bucket_no == nbucket) { // we're done
                break;
            }
            bucket_no = init_bucket_no;
            row = -1;
        }
    }

    for (size_t i = 0; i < nbucket; i++) {
        assert(ptrs[i] == lims[i + 1]);
    }
    double t3 = getmillisecs();
    if (bucket_sort_verbose) {
        printf("times %.3f %.3f %.3f\n", t1 - t0, t2 - t1, t3 - t2);
    }
}

// collects row numbers to write into buckets
struct ToWrite {
    int32_t nbucket;
    std::vector<int32_t> buckets;
    std::vector<int32_t> rows;
    std::vector<size_t> lims;

    explicit ToWrite(int32_t nbucket) : nbucket(nbucket) {
        lims.resize(nbucket + 1);
    }

    /// add one element (row) to write in bucket b
    void add(int32_t row, int32_t b) {
        assert(b >= 0 && b < nbucket);
        rows.push_back(row);
        buckets.push_back(b);
    }

    void bucket_sort() {
        FAISS_THROW_IF_NOT(buckets.size() == rows.size());
        lims.resize(nbucket + 1);
        memset(lims.data(), 0, sizeof(lims[0]) * (nbucket + 1));

        for (size_t i = 0; i < buckets.size(); i++) {
            assert(buckets[i] >= 0 && buckets[i] < nbucket);
            lims[buckets[i] + 1]++;
        }
        // compute cumulative sum
        for (size_t i = 0; i < nbucket; i++) {
            lims[i + 1] += lims[i];
        }
        FAISS_THROW_IF_NOT(lims[nbucket] == buckets.size());

        // could also do a circular perm...
        std::vector<int32_t> new_rows(rows.size());
        std::vector<size_t> ptrs = lims;
        for (size_t i = 0; i < buckets.size(); i++) {
            int32_t b = buckets[i];
            assert(ptrs[b] < lims[b + 1]);
            new_rows[ptrs[b]++] = rows[i];
        }
        buckets.resize(0);
        std::swap(rows, new_rows);
    }

    void swap(ToWrite& other) {
        assert(nbucket == other.nbucket);
        buckets.swap(other.buckets);
        rows.swap(other.rows);
        lims.swap(other.lims);
    }
};

void bucket_sort_inplace_parallel(
        size_t nrow,
        size_t ncol,
        int32_t* vals,
        int32_t nbucket,
        int64_t* lims,
        int nt_in) {
    int verbose = bucket_sort_verbose;
    memset(lims, 0, sizeof(*lims) * (nbucket + 1));
    std::vector<ToWrite> all_to_write;
    size_t nval = nrow * ncol;
    FAISS_THROW_IF_NOT(
            nbucket < nval); // unclear what would happen in this case...

    // try to keep size of all_to_write < 5GiB
    // but we need at least one element per bucket
    size_t init_to_write = std::max(
            size_t(nbucket),
            std::min(
                    nval / 10,
                    ((size_t)5 << 30) / (sizeof(int32_t) * 3 * nt_in)));
    if (verbose > 0) {
        printf("init_to_write=%zd\n", init_to_write);
    }

    std::vector<size_t> ptrs(nbucket); // ptrs is shared across all threads
    std::vector<char> did_wrap(
            nbucket); // DON'T use std::vector<bool> that cannot be accessed
                      // safely from multiple threads!!!

#pragma omp parallel num_threads(nt_in)
    {
        int nt = omp_get_num_threads(); // might be different from nt_in (?)
        int rank = omp_get_thread_num();
        std::vector<int64_t> local_lims(nbucket + 1);

        // range of indices handled by this thread
        size_t i0 = nval * rank / nt;
        size_t i1 = nval * (rank + 1) / nt;

        // build histogram in local lims
        for (size_t i = i0; i < i1; i++) {
            local_lims[vals[i]]++;
        }
#pragma omp critical
        { // accumulate histograms (not shifted indices to prepare cumsum)
            for (size_t i = 0; i < nbucket; i++) {
                lims[i + 1] += local_lims[i];
            }
            all_to_write.push_back(ToWrite(nbucket));
        }

#pragma omp barrier
        // this thread's things to write
        ToWrite& to_write = all_to_write[rank];

#pragma omp master
        {
            // compute cumulative sum
            for (size_t i = 0; i < nbucket; i++) {
                lims[i + 1] += lims[i];
            }
            FAISS_THROW_IF_NOT(lims[nbucket] == nval);
            // at this point lims is final (read only!)

            memcpy(ptrs.data(), lims, sizeof(lims[0]) * nbucket);

            // initial values to write (we write -1s to get the process running)
            // make sure at least one element per bucket
            size_t written = 0;
            for (int32_t b = 0; b < nbucket; b++) {
                size_t l0 = lims[b], l1 = lims[b + 1];
                size_t target_to_write = l1 * init_to_write / nval;
                do {
                    if (l0 == l1) {
                        break;
                    }
                    to_write.add(-1, b);
                    l0++;
                    written++;
                } while (written < target_to_write);
            }

            to_write.bucket_sort();
        }

        // this thread writes only buckets b0:b1
        size_t b0 = (rank * nbucket + nt - 1) / nt;
        size_t b1 = ((rank + 1) * nbucket + nt - 1) / nt;

        // in this loop, we write elements collected in the previous round
        // and collect the elements that are overwritten for the next round
        size_t tot_written = 0;
        int round = 0;
        for (;;) {
#pragma omp barrier

            size_t n_to_write = 0;
            for (const ToWrite& to_write_2 : all_to_write) {
                n_to_write += to_write_2.lims.back();
            }

            tot_written += n_to_write;
            // assert(tot_written <= nval);

#pragma omp master
            {
                if (verbose >= 1) {
                    printf("ROUND %d n_to_write=%zd\n", round, n_to_write);
                }
                if (verbose > 2) {
                    for (size_t b = 0; b < nbucket; b++) {
                        printf("   b=%zd [", b);
                        for (size_t i = lims[b]; i < lims[b + 1]; i++) {
                            printf(" %s%d", ptrs[b] == i ? ">" : "", vals[i]);
                        }
                        printf(" %s] %s\n",
                               ptrs[b] == lims[b + 1] ? ">" : "",
                               did_wrap[b] ? "w" : "");
                    }
                    printf("To write\n");
                    for (size_t b = 0; b < nbucket; b++) {
                        printf("   b=%zd ", b);
                        const char* sep = "[";
                        for (const ToWrite& to_write_2 : all_to_write) {
                            printf("%s", sep);
                            sep = " |";
                            size_t l0 = to_write_2.lims[b];
                            size_t l1 = to_write_2.lims[b + 1];
                            for (size_t i = l0; i < l1; i++) {
                                printf(" %d", to_write_2.rows[i]);
                            }
                        }
                        printf(" ]\n");
                    }
                }
            }
            if (n_to_write == 0) {
                break;
            }
            round++;

#pragma omp barrier

            ToWrite next_to_write(nbucket);

            for (size_t b = b0; b < b1; b++) {
                for (const ToWrite& to_write_2 : all_to_write) {
                    size_t l0 = to_write_2.lims[b];
                    size_t l1 = to_write_2.lims[b + 1];
                    for (size_t i = l0; i < l1; i++) {
                        int32_t row = to_write_2.rows[i];
                        size_t idx = ptrs[b];
                        if (verbose > 2) {
                            printf("    bucket %d (rank %d) idx %zd\n",
                                   row,
                                   rank,
                                   idx);
                        }
                        if (idx < lims[b + 1]) {
                            ptrs[b]++;
                        } else {
                            // wrapping around
                            assert(!did_wrap[b]);
                            did_wrap[b] = true;
                            idx = lims[b];
                            ptrs[b] = idx + 1;
                        }

                        // check if we need to remember the overwritten number
                        if (vals[idx] >= 0) {
                            int32_t new_row = idx / ncol;
                            next_to_write.add(new_row, vals[idx]);
                            if (verbose > 2) {
                                printf("       new_row=%d\n", new_row);
                            }
                        } else {
                            assert(did_wrap[b]);
                        }

                        vals[idx] = row;
                    }
                }
            }
            next_to_write.bucket_sort();
#pragma omp barrier
            all_to_write[rank].swap(next_to_write);
        }
    }
}

} // anonymous namespace

void bucket_sort(
        size_t nval,
        const uint64_t* vals,
        uint64_t vmax,
        int64_t* lims,
        int64_t* perm,
        int nt) {
    if (nt == 0) {
        bucket_sort_ref(nval, vals, vmax, lims, perm);
    } else {
        bucket_sort_parallel(nval, vals, vmax, lims, perm, nt);
    }
}

void matrix_bucket_sort_inplace(
        size_t nrow,
        size_t ncol,
        int32_t* vals,
        int32_t vmax,
        int64_t* lims,
        int nt) {
    if (nt == 0) {
        bucket_sort_inplace_ref(nrow, ncol, vals, vmax, lims);
    } else {
        bucket_sort_inplace_parallel(nrow, ncol, vals, vmax, lims, nt);
    }
}

} // namespace faiss
