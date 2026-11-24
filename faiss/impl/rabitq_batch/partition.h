/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

/** Coarse-partitioned batched HNSW graph build for large inputs.
 *
 * The exact whole-batch k-NN compares all n^2 / 2 pairs, which stops paying
 * off above a few million vectors. Here the candidates of each vector are
 * searched only among the vectors that share a coarse bucket with it:
 *
 *  1. Deterministic spherical k-means (k-means++ init, 10 Lloyd iterations,
 *     BLAS sgemm) on a subsample picks P centroids, so that buckets hold
 *     about 350k vectors when each vector goes to 3 of them.
 *  2. Every vector is assigned to its 3 nearest centroids. Assigning to
 *     several buckets recovers most neighbors that sit across a bucket
 *     border. Oversized buckets are split by rank (median cut plus a 15%
 *     overlap, see split.h) until every bucket is within the size limit.
 *  3. For each bucket, only its codes are decoded and the AMX/VNNI k-NN of
 *     knn.h runs on them; the bucket top-k lists are merged into a global
 *     candidate table (uint32 ids, bf16 scores).
 *  4. The score table is freed, level 0 is linked from the codes with a
 *     small per-thread buffer, and the upper levels use the subset k-NN.
 *
 * Since each vector goes to 3 buckets of about 350k vectors, the work grows
 * linearly with n, up to the cap of 128 buckets (about 15M vectors).
 */

#include <faiss/impl/HNSW.h>
#include <faiss/impl/RaBitQCodePairIP.h>
#include <faiss/impl/rabitq_batch/knn.h>
#include <faiss/impl/rabitq_batch/split.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <vector>

extern "C" {
#ifndef FINTEGER
#define FINTEGER int
#endif
int sgemm_(
        const char* transa,
        const char* transb,
        FINTEGER* m,
        FINTEGER* n,
        FINTEGER* k,
        const float* alpha,
        const float* a,
        FINTEGER* lda,
        const float* b,
        FINTEGER* ldb,
        float* beta,
        float* c,
        FINTEGER* ldc);
}

namespace faiss {
namespace rabitq_batch {

inline uint16_t f32_to_bf16(float x) {
    uint32_t u;
    std::memcpy(&u, &x, 4);
    if ((u & 0x7FFFFFFFu) < 0x7F800000u) {
        u += 0x7FFFu + ((u >> 16) & 1u);
    }
    return uint16_t(u >> 16);
}

inline float bf16_to_f32(uint16_t b) {
    uint32_t u = uint32_t(b) << 16;
    float x;
    std::memcpy(&x, &u, 4);
    return x;
}

/// Computes Out[N_rows x P] = X[N_rows x d] * C[P x d]^T (all row-major).
inline void sgemm_x_ct(
        const float* X,
        const float* C,
        float* Out,
        size_t n_rows,
        size_t P,
        size_t d) {
    const char ta = 'T', tb = 'N';
    FINTEGER m = FINTEGER(P);
    FINTEGER n = FINTEGER(n_rows);
    FINTEGER k = FINTEGER(d);
    FINTEGER lda = FINTEGER(d);
    FINTEGER ldb = FINTEGER(d);
    FINTEGER ldc = FINTEGER(P);
    const float alpha = 1.0f;
    float beta = 0.0f;
    sgemm_(&ta, &tb, &m, &n, &k, &alpha, C, &lda, X, &ldb, &beta, Out, &ldc);
}

inline void l2_normalize_row(float* v, size_t d) {
    double s = 0.0;
    for (size_t j = 0; j < d; ++j) {
        s += double(v[j]) * double(v[j]);
    }
    if (s > 1e-24) {
        const float inv = float(1.0 / std::sqrt(s));
        for (size_t j = 0; j < d; ++j) {
            v[j] *= inv;
        }
    }
}

/// Deterministic spherical k-means on unit-normalized sample vectors.
inline void train_spherical_kmeans(
        const float* x,
        size_t n,
        size_t d,
        size_t P,
        size_t threads,
        std::vector<float>& centroids,
        uint64_t seed = 42,
        const uint32_t* subset = nullptr,
        size_t subset_n = 0) {
    const size_t total = subset ? subset_n : n;
    const size_t S = std::min<size_t>(
            total, std::max<size_t>(std::min<size_t>(total, 65536), P * 512));
    std::vector<float> xs(S * d);
    {
        std::mt19937_64 rng(seed);
        std::vector<size_t> idx(S);
        // Deterministic stride + jitter sampling without O(n) allocation
        const double step = double(total) / double(S);
        for (size_t i = 0; i < S; ++i) {
            size_t lo = size_t(i * step);
            size_t hi = std::min(total, size_t((i + 1) * step));
            size_t pick = (hi > lo) ? lo + (rng() % (hi - lo)) : lo;
            idx[i] = subset ? size_t(subset[pick]) : pick;
        }
#pragma omp parallel for num_threads(threads) schedule(static)
        for (size_t i = 0; i < S; ++i) {
            std::memcpy(xs.data() + i * d, x + idx[i] * d, d * sizeof(float));
            l2_normalize_row(xs.data() + i * d, d);
        }
    }

    // k-means++ initialization on up to 8192 samples (deterministic)
    centroids.assign(P * d, 0.0f);
    const size_t S_init = std::min<size_t>(S, 8192);
    std::mt19937_64 rng_init(seed + 1337);
    {
        size_t first = rng_init() % S_init;
        std::memcpy(centroids.data(), xs.data() + first * d, d * sizeof(float));
        std::vector<float> best_sim(S_init, -2.0f);
        std::vector<double> weights(S_init, 0.0);
        for (size_t c = 1; c < P; ++c) {
            const float* prev = centroids.data() + (c - 1) * d;
#pragma omp parallel for if (S_init >= 1024) num_threads(threads) \
        schedule(static)
            for (size_t i = 0; i < S_init; ++i) {
                const float* row = xs.data() + i * d;
                float dot = 0.0f;
                for (size_t j = 0; j < d; ++j) {
                    dot += row[j] * prev[j];
                }
                if (dot > best_sim[i]) {
                    best_sim[i] = dot;
                }
                double dist2 = std::max(0.0, 2.0 - 2.0 * double(best_sim[i]));
                weights[i] = dist2;
            }
            double wsum = 0.0;
            for (size_t i = 0; i < S_init; ++i) {
                wsum += weights[i];
            }
            size_t chosen = c % S_init;
            if (wsum > 0.0) {
                double target = std::uniform_real_distribution<double>(
                        0.0, wsum)(rng_init);
                double acc = 0.0;
                for (size_t i = 0; i < S_init; ++i) {
                    acc += weights[i];
                    if (acc >= target) {
                        chosen = i;
                        break;
                    }
                }
            }
            std::memcpy(
                    centroids.data() + c * d,
                    xs.data() + chosen * d,
                    d * sizeof(float));
        }
    }

    // 10 Lloyd iterations using BLAS sgemm_
    std::vector<float> dots(S * P);
    std::vector<uint32_t> owner(S);
    std::vector<float> max_sim(S);
    const int niter = 10;
    for (int it = 0; it < niter; ++it) {
        sgemm_x_ct(xs.data(), centroids.data(), dots.data(), S, P, d);
#pragma omp parallel for num_threads(threads) schedule(static)
        for (size_t i = 0; i < S; ++i) {
            const float* row = dots.data() + i * P;
            uint32_t best_p = 0;
            float best_s = row[0];
            for (size_t p = 1; p < P; ++p) {
                if (row[p] > best_s) {
                    best_s = row[p];
                    best_p = uint32_t(p);
                }
            }
            owner[i] = best_p;
            max_sim[i] = best_s;
        }
        std::vector<double> acc(P * d, 0.0);
        std::vector<size_t> cnt(P, 0);
        // Thread-local accumulation over samples
#pragma omp parallel num_threads(threads)
        {
            std::vector<double> lacc(P * d, 0.0);
            std::vector<size_t> lcnt(P, 0);
#pragma omp for schedule(static)
            for (size_t i = 0; i < S; ++i) {
                const uint32_t p = owner[i];
                lcnt[p] += 1;
                double* dst = lacc.data() + size_t(p) * d;
                const float* src = xs.data() + i * d;
                for (size_t j = 0; j < d; ++j) {
                    dst[j] += double(src[j]);
                }
            }
#pragma omp critical
            {
                for (size_t p = 0; p < P; ++p) {
                    cnt[p] += lcnt[p];
                    double* dst = acc.data() + p * d;
                    const double* src = lacc.data() + p * d;
                    for (size_t j = 0; j < d; ++j) {
                        dst[j] += src[j];
                    }
                }
            }
        }
        for (size_t p = 0; p < P; ++p) {
            if (cnt[p] == 0) {
                // Re-seed empty cluster from sample with lowest max_sim
                size_t worst_i = 0;
                float worst_s = max_sim[0];
                for (size_t i = 1; i < S; ++i) {
                    if (max_sim[i] < worst_s) {
                        worst_s = max_sim[i];
                        worst_i = i;
                    }
                }
                max_sim[worst_i] = 2.0f;
                std::memcpy(
                        centroids.data() + p * d,
                        xs.data() + worst_i * d,
                        d * sizeof(float));
            } else {
                float* dst = centroids.data() + p * d;
                const double* src = acc.data() + p * d;
                for (size_t j = 0; j < d; ++j) {
                    dst[j] = float(src[j]);
                }
                l2_normalize_row(dst, d);
            }
        }
    }
}

/// Splits an oversized bucket into 2 sub-buckets: spherical 2-means gives the
/// split direction, then the members are cut at the median of their
/// projection with a 15% overlap band copied into both halves so
/// cross-boundary neighbors remain connected. Each half is at most ~57.5% of
/// the bucket regardless of data skew or duplicates (see split.h).
inline void split_oversized_bucket(
        const float* x,
        size_t d,
        size_t threads,
        const std::vector<uint32_t>& bucket,
        std::vector<uint32_t>& out0,
        std::vector<uint32_t>& out1,
        uint64_t seed) {
    const size_t nb = bucket.size();
    std::vector<float> c2;
    train_spherical_kmeans(x, 0, d, 2, threads, c2, seed, bucket.data(), nb);
    std::vector<float> diff(nb);
    const float* c0 = c2.data();
    const float* c1 = c2.data() + d;
#pragma omp parallel for num_threads(threads) schedule(static)
    for (size_t i = 0; i < nb; ++i) {
        const float* row = x + size_t(bucket[i]) * d;
        float s0 = 0.0f, s1 = 0.0f;
        for (size_t j = 0; j < d; ++j) {
            s0 += row[j] * c0[j];
            s1 += row[j] * c1[j];
        }
        diff[i] = s0 - s1;
    }
    split_bucket_by_rank(bucket, diff, 0.15, out0, out1);
}

/// AVX512-VNNI inner product sum(w_a * w_b) on two L2-resident decoded rows.
inline int32_t ww_ptr(
        const uint8_t* wa,
        const uint8_t* wb,
        int32_t wsum_a,
        size_t kc) {
    const __m512i flip = _mm512_set1_epi8(char(0x80));
    __m512i acc = _mm512_setzero_si512();
    for (size_t c = 0; c < kc; ++c) {
        acc = _mm512_dpbusd_epi32(
                acc,
                _mm512_load_si512(wa + c * 64),
                _mm512_xor_si512(_mm512_load_si512(wb + c * 64), flip));
    }
    return wrap(
            uint32_t(_mm512_reduce_add_epi32(acc)) + 128u * uint32_t(wsum_a));
}

struct LocalPool {
    Dims m;
    const uint8_t* codes = nullptr;
    size_t code_size = 0;
    double cns = 0.0;
    uint8_t* lw = nullptr;
    size_t cap_rows = 0;
    std::vector<int32_t> lwsum;
    std::vector<uint32_t> gids;

    LocalPool(
            const Dims& dims,
            const uint8_t* codes_in,
            size_t code_size_in,
            double cns_in)
            : m(dims), codes(codes_in), code_size(code_size_in), cns(cns_in) {}
    LocalPool(const LocalPool&) = delete;
    ~LocalPool() {
        std::free(lw);
    }

    void ensure(size_t nrows) {
        if (nrows > cap_rows) {
            std::free(lw);
            cap_rows = std::max<size_t>(nrows, cap_rows * 2);
            lw = alloc<uint8_t>(cap_rows * m.stride);
        }
        lwsum.resize(nrows);
        gids.resize(nrows);
    }

    void load(uint32_t center, const uint32_t* cands, size_t nc) {
        ensure(1 + nc);
        gids[0] = center;
        lwsum[0] = decode_row(codes + size_t(center) * code_size, m, lw);
        for (size_t i = 0; i < nc; ++i) {
            const uint32_t g = cands[i];
            gids[1 + i] = g;
            lwsum[1 + i] = decode_row(
                    codes + size_t(g) * code_size, m, lw + (1 + i) * m.stride);
        }
    }

    inline float sim_idx(size_t iu, size_t iv) const {
        uint32_t gu = gids[iu], gv = gids[iv];
        if (gu > gv) {
            std::swap(gu, gv);
            std::swap(iu, iv);
        }
        const int32_t core =
                ww_ptr(lw + iu * m.stride, lw + iv * m.stride, lwsum[iu], m.kc);
        const int64_t dot = 4 * int64_t(core) -
                510 * int64_t(lwsum[iu] + lwsum[iv]) + 65025 * int64_t(m.d);
        return rabitq_ip::restore(
                codes + size_t(gu) * code_size,
                codes + size_t(gv) * code_size,
                m.d,
                cns,
                dot);
    }
};

struct LocalCand {
    float s;
    uint32_t id;
    uint32_t idx; // index into LocalPool
};

inline bool better_loc(const LocalCand& a, const LocalCand& b) {
    return a.s > b.s || (a.s == b.s && a.id < b.id);
}

inline void shrink_local(
        const LocalPool& pool,
        const std::vector<LocalCand>& sorted,
        size_t max_size,
        std::vector<Cand>& out) {
    out.clear();
    if (sorted.size() < max_size) {
        out.reserve(sorted.size());
        for (const auto& v : sorted) {
            out.push_back({v.s, v.id});
        }
        return;
    }
    std::vector<uint32_t> out_idx;
    out_idx.reserve(max_size);
    for (const LocalCand& v : sorted) {
        bool good = true;
        for (uint32_t ui : out_idx) {
            if (v.s < pool.sim_idx(ui, v.idx)) {
                good = false;
                break;
            }
        }
        if (good) {
            out.push_back({v.s, v.id});
            out_idx.push_back(v.idx);
            if (out.size() >= max_size) {
                return;
            }
        }
    }
}

/// Links one level without a global decoded copy of the codes: the codes of
/// each node and of its candidates are decoded into a small per-thread
/// buffer, which saves O(n * d) bytes of memory.
template <class CandFn>
inline void build_level_codes(
        const uint8_t* codes,
        size_t n,
        const Dims& m_dims,
        size_t code_size,
        double cns,
        HNSW& h,
        int level,
        const std::vector<uint32_t>& nodes,
        CandFn cands,
        size_t threads) {
    const size_t nl = nodes.size(), m = h.nb_neighbors(level);
    std::vector<int32_t> pos(n, -1);
    for (size_t i = 0; i < nl; ++i) {
        pos[nodes[i]] = int32_t(i);
    }
    std::vector<Cand> fwd(nl * m);
    std::vector<uint32_t> fcnt(nl, 0);
    std::vector<std::atomic<uint32_t>> indeg(nl);
    for (auto& x : indeg) {
        x.store(0, std::memory_order_relaxed);
    }
#pragma omp parallel num_threads(threads)
    {
        LocalPool pool(m_dims, codes, code_size, cns);
        std::vector<LocalCand> list;
        std::vector<Cand> out;
        std::vector<uint32_t> ids;
#pragma omp for schedule(dynamic, 256)
        for (size_t li = 0; li < nl; ++li) {
            const uint32_t a = nodes[li];
            cands(a, ids);
            pool.load(a, ids.data(), ids.size());
            list.clear();
            list.reserve(ids.size());
            for (size_t j = 0; j < ids.size(); ++j) {
                list.push_back(
                        {pool.sim_idx(0, 1 + j), ids[j], uint32_t(1 + j)});
            }
            std::sort(list.begin(), list.end(), better_loc);
            shrink_local(pool, list, m, out);
            fcnt[li] = out.size();
            std::copy(out.begin(), out.end(), fwd.begin() + li * m);
            for (const Cand& v : out) {
                indeg[pos[v.id]].fetch_add(1, std::memory_order_relaxed);
            }
        }
    }
    std::vector<size_t> start(nl + 1, 0);
    for (size_t i = 0; i < nl; ++i) {
        start[i + 1] = start[i] + indeg[i].load();
    }
    std::vector<uint32_t> rev(start[nl]);
    std::vector<std::atomic<uint32_t>> cursor(nl);
    for (size_t i = 0; i < nl; ++i) {
        cursor[i].store(0, std::memory_order_relaxed);
    }
#pragma omp parallel for num_threads(threads) schedule(static)
    for (size_t li = 0; li < nl; ++li) {
        for (size_t t = 0; t < fcnt[li]; ++t) {
            const size_t lb = pos[fwd[li * m + t].id];
            rev[start[lb] +
                cursor[lb].fetch_add(1, std::memory_order_relaxed)] = nodes[li];
        }
    }
#pragma omp parallel num_threads(threads)
    {
        LocalPool pool(m_dims, codes, code_size, cns);
        std::vector<uint32_t> uniq_ids;
        std::vector<LocalCand> list;
        std::vector<Cand> out;
#pragma omp for schedule(dynamic, 256)
        for (size_t li = 0; li < nl; ++li) {
            const uint32_t b = nodes[li];
            std::sort(rev.begin() + start[li], rev.begin() + start[li + 1]);
            uniq_ids.clear();
            const size_t fc = fcnt[li];
            for (size_t t = 0; t < fc; ++t) {
                uniq_ids.push_back(fwd[li * m + t].id);
            }
            for (size_t r = start[li]; r < start[li + 1]; ++r) {
                const uint32_t a = rev[r];
                bool seen = false;
                for (size_t t = 0; t < fc && !seen; ++t) {
                    seen = fwd[li * m + t].id == a;
                }
                if (!seen) {
                    uniq_ids.push_back(a);
                }
            }
            if (uniq_ids.size() <= m) {
                // Fast path: no shrink needed, only score unseen reverse edges
                // for deterministic ordering
                out.clear();
                out.reserve(uniq_ids.size());
                for (size_t t = 0; t < fc; ++t) {
                    out.push_back(fwd[li * m + t]);
                }
                const size_t extra = uniq_ids.size() - fc;
                if (extra > 0) {
                    pool.load(b, uniq_ids.data() + fc, extra);
                    for (size_t j = 0; j < extra; ++j) {
                        out.push_back(
                                {pool.sim_idx(0, 1 + j), uniq_ids[fc + j]});
                    }
                }
                std::sort(out.begin(), out.end(), better);
            } else {
                pool.load(b, uniq_ids.data(), uniq_ids.size());
                list.clear();
                list.reserve(uniq_ids.size());
                for (size_t t = 0; t < fc; ++t) {
                    list.push_back(
                            {fwd[li * m + t].s,
                             fwd[li * m + t].id,
                             uint32_t(1 + t)});
                }
                for (size_t j = fc; j < uniq_ids.size(); ++j) {
                    list.push_back(
                            {pool.sim_idx(0, 1 + j),
                             uniq_ids[j],
                             uint32_t(1 + j)});
                }
                std::sort(list.begin(), list.end(), better_loc);
                shrink_local(pool, list, m, out);
            }
            size_t begin, end;
            h.neighbor_range(b, level, &begin, &end);
            size_t i = begin;
            for (const Cand& v : out) {
                h.neighbors[i++] = HNSW::storage_idx_t(v.id);
            }
            for (; i < end; ++i) {
                h.neighbors[i] = -1;
            }
        }
    }
}

/// Builds upper levels (level >= 1) directly from codes without global Data::w.
inline void build_upper_levels_codes(
        const uint8_t* codes,
        size_t n,
        const Dims& m_dims,
        size_t code_size,
        double cns,
        HNSW& h,
        size_t k,
        bool vnni,
        size_t threads) {
    for (int level = 1; level <= h.max_level; ++level) {
        std::vector<uint32_t> nodes;
        for (size_t i = 0; i < n; ++i) {
            if (h.levels[i] > level) {
                nodes.push_back(uint32_t(i));
            }
        }
        if (nodes.size() <= 1) {
            continue;
        }
        const size_t kk = std::min(k, nodes.size() - 1), nl = nodes.size();
        std::vector<uint32_t> top(nl * kk);
        if (nl >= 4096) {
            Data sub;
            sub.m = m_dims;
            sub.n = nl;
            sub.threads =
                    std::min<size_t>(threads, std::max<size_t>(1, nl / 1024));
            sub.codes = codes;
            sub.code_size = code_size;
            const Layout subL = allocate(sub, vnni);
            prepare_subset(sub, nodes.data());
            Rows st(sub.pad, kk, sub.threads);
            knn(sub, st, subL, vnni);
#pragma omp parallel for if (sub.threads > 1) num_threads(sub.threads) \
        schedule(static)
            for (size_t li = 0; li < nl; ++li) {
                const uint32_t* row_ids = st.ids + li * kk;
                uint32_t* dst = top.data() + li * kk;
                const uint32_t f = st.fill[li];
                for (size_t j = 0; j < f; ++j) {
                    dst[j] = nodes[row_ids[j]];
                }
            }
        } else {
            // Small upper level (< 4096): decode subset and run brute force
            Data sub;
            sub.m = m_dims;
            sub.n = nl;
            sub.threads =
                    std::min<size_t>(threads, std::max<size_t>(1, nl / 256));
            sub.codes = codes;
            sub.code_size = code_size;
            allocate(sub, false);
            prepare_subset(sub, nodes.data());
#pragma omp parallel num_threads(threads)
            {
                std::vector<Cand> row;
#pragma omp for schedule(dynamic, 16)
                for (size_t li = 0; li < nl; ++li) {
                    row.clear();
                    const uint32_t ga = nodes[li];
                    for (size_t lj = 0; lj < nl; ++lj) {
                        if (lj == li)
                            continue;
                        const uint32_t gb = nodes[lj];
                        size_t u = li, v = lj;
                        uint32_t gu = ga, gv = gb;
                        if (gu > gv) {
                            std::swap(gu, gv);
                            std::swap(u, v);
                        }
                        const int32_t core =
                                ww_ptr(sub.w + u * m_dims.stride,
                                       sub.w + v * m_dims.stride,
                                       sub.wsum[u],
                                       m_dims.kc);
                        const int64_t dot = 4 * int64_t(core) -
                                510 * int64_t(sub.wsum[u] + sub.wsum[v]) +
                                65025 * int64_t(m_dims.d);
                        float s = rabitq_ip::restore(
                                codes + size_t(gu) * code_size,
                                codes + size_t(gv) * code_size,
                                m_dims.d,
                                cns,
                                dot);
                        row.push_back({s, gb});
                    }
                    std::partial_sort(
                            row.begin(), row.begin() + kk, row.end(), better);
                    for (size_t i = 0; i < kk; ++i) {
                        top[li * kk + i] = row[i].id;
                    }
                }
            }
        }
        std::vector<int32_t> pos(n, -1);
        for (size_t i = 0; i < nl; ++i) {
            pos[nodes[i]] = int32_t(i);
        }
        build_level_codes(
                codes,
                n,
                m_dims,
                code_size,
                cns,
                h,
                level,
                nodes,
                [&](uint32_t a, std::vector<uint32_t>& ids) {
                    ids.assign(
                            top.begin() + size_t(pos[a]) * kk,
                            top.begin() + size_t(pos[a] + 1) * kk);
                },
                threads);
    }
}

inline void build_graph_partitioned(
        const uint8_t* codes,
        const float* x,
        size_t n,
        size_t d,
        size_t code_size,
        double cns,
        HNSW& h,
        size_t k,
        bool vnni,
        size_t threads) {
    const Dims m_dims(d);

    // 1. Number of assignments per vector r and of coarse buckets P
    size_t r = 3;
    const size_t b_target = 350000;
    size_t P = std::clamp<size_t>((r * n + b_target / 2) / b_target, 6, 128);
    P = std::min<size_t>(P, std::max<size_t>(2, n / 1024));
    r = std::min<size_t>(r, P);

    // 2. Train spherical k-means centroids
    std::vector<float> centroids;
    train_spherical_kmeans(x, n, d, P, threads, centroids, 42);

    // 3. Streaming chunked multi-assign using BLAS sgemm_
    std::vector<uint16_t> assign(n * r);
    {
        const size_t chunk = 32768;
        std::vector<float> dots(chunk * P);
        for (size_t i0 = 0; i0 < n; i0 += chunk) {
            const size_t cnt = std::min(chunk, n - i0);
            sgemm_x_ct(x + i0 * d, centroids.data(), dots.data(), cnt, P, d);
#pragma omp parallel for num_threads(threads) schedule(static)
            for (size_t i = 0; i < cnt; ++i) {
                const float* row = dots.data() + i * P;
                float best_s[4];
                uint16_t best_p[4];
                for (size_t t = 0; t < r; ++t) {
                    best_s[t] = -INFINITY;
                    best_p[t] = 0;
                }
                for (size_t p = 0; p < P; ++p) {
                    const float s = row[p];
                    if (s > best_s[r - 1]) {
                        size_t pos = r - 1;
                        while (pos > 0 && s > best_s[pos - 1]) {
                            best_s[pos] = best_s[pos - 1];
                            best_p[pos] = best_p[pos - 1];
                            --pos;
                        }
                        best_s[pos] = s;
                        best_p[pos] = uint16_t(p);
                    }
                }
                uint16_t* dst = assign.data() + (i0 + i) * r;
                for (size_t t = 0; t < r; ++t) {
                    dst[t] = best_p[t];
                }
            }
        }
    }
    centroids.clear();
    centroids.shrink_to_fit();

    // Build per-bucket sorted vector ID lists
    std::vector<size_t> bcounts(P, 0);
    for (size_t i = 0; i < n * r; ++i) {
        ++bcounts[assign[i]];
    }
    std::vector<std::vector<uint32_t>> buckets(P);
    for (size_t p = 0; p < P; ++p) {
        buckets[p].reserve(bcounts[p]);
    }
    for (size_t i = 0; i < n; ++i) {
        for (size_t t = 0; t < r; ++t) {
            buckets[assign[i * r + t]].push_back(uint32_t(i));
        }
    }
    assign.clear();
    assign.shrink_to_fit();

    // Protect against oversized buckets: split until every bucket is within
    // the limit. Each rank split shrinks a bucket to <= ~57.5%, so the pass
    // cap is only a safety net (it is never reached for n < 2^32).
    const size_t split_limit =
            std::max<size_t>(650000, size_t(2.2 * double(r * n) / double(P)));
    constexpr int kMaxSplitPasses = 64;
    for (int pass = 0; pass < kMaxSplitPasses; ++pass) {
        std::vector<std::vector<uint32_t>> next_buckets;
        next_buckets.reserve(buckets.size());
        bool did_split = false;
        for (size_t bi = 0; bi < buckets.size(); ++bi) {
            if (buckets[bi].size() > split_limit) {
                std::vector<uint32_t> b0, b1;
                split_oversized_bucket(
                        x,
                        d,
                        threads,
                        buckets[bi],
                        b0,
                        b1,
                        1000 + uint64_t(pass) * 1000 + bi);
                buckets[bi].clear();
                buckets[bi].shrink_to_fit();
                if (!b0.empty())
                    next_buckets.push_back(std::move(b0));
                if (!b1.empty())
                    next_buckets.push_back(std::move(b1));
                did_split = true;
            } else if (!buckets[bi].empty()) {
                next_buckets.push_back(std::move(buckets[bi]));
            }
        }
        buckets = std::move(next_buckets);
        if (!did_split)
            break;
    }

    // 4. Per-bucket AMX/VNNI symmetric knn() + compact global merge
    uint32_t* g_ids = alloc<uint32_t>(n * k);
    uint16_t* g_bf16 = alloc<uint16_t>(n * k);
    uint32_t* g_fill = alloc<uint32_t>(n);
    std::memset(g_fill, 0, n * sizeof(uint32_t));

    for (size_t bi = 0; bi < buckets.size(); ++bi) {
        const auto& b = buckets[bi];
        const size_t nb = b.size();
        if (nb <= 1)
            continue;
        const size_t bk = std::min(k, nb - 1);

        Data sub;
        sub.m = m_dims;
        sub.n = nb;
        sub.threads = std::min<size_t>(threads, std::max<size_t>(1, nb / 1024));
        sub.codes = codes;
        sub.code_size = code_size;
        const Layout subL = allocate(sub, vnni);
        prepare_subset(sub, b.data());
        Rows st(sub.pad, bk, sub.threads);
        knn(sub, st, subL, vnni, /*do_compact=*/false);

        // Race-free merge: every global id appears at most once in bucket `b`.
#pragma omp parallel num_threads(threads)
        {
            std::vector<Cand> comb;
            comb.reserve(2 * k);
#pragma omp for schedule(static)
            for (size_t li = 0; li < nb; ++li) {
                const uint32_t g = b[li];
                const Cand* src = st.buf + li * bk;
                const uint32_t f_new = st.fill[li];
                uint32_t* dst_id = g_ids + size_t(g) * k;
                uint16_t* dst_sc = g_bf16 + size_t(g) * k;
                const uint32_t f_old = g_fill[g];
                if (f_old == 0) {
                    const uint32_t take =
                            std::min<uint32_t>(f_new, uint32_t(k));
                    for (uint32_t j = 0; j < take; ++j) {
                        dst_id[j] = b[src[j].id];
                        dst_sc[j] = f32_to_bf16(src[j].s);
                    }
                    g_fill[g] = take;
                } else {
                    comb.clear();
                    for (uint32_t j = 0; j < f_old; ++j) {
                        comb.push_back({bf16_to_f32(dst_sc[j]), dst_id[j]});
                    }
                    for (uint32_t j = 0; j < f_new; ++j) {
                        comb.push_back(
                                {bf16_to_f32(f32_to_bf16(src[j].s)),
                                 b[src[j].id]});
                    }
                    // Deduplicate by id, keeping highest score
                    std::sort(
                            comb.begin(),
                            comb.end(),
                            [](const Cand& u, const Cand& v) {
                                return u.id < v.id ||
                                        (u.id == v.id && u.s > v.s);
                            });
                    size_t w = 0;
                    for (size_t j = 0; j < comb.size(); ++j) {
                        if (w == 0 || comb[j].id != comb[w - 1].id) {
                            comb[w++] = comb[j];
                        }
                    }
                    comb.resize(w);
                    const size_t take = std::min(comb.size(), k);
                    if (comb.size() > take) {
                        std::nth_element(
                                comb.begin(),
                                comb.begin() + take,
                                comb.end(),
                                better);
                        comb.resize(take);
                    }
                    std::sort(comb.begin(), comb.end(), better);
                    for (size_t j = 0; j < take; ++j) {
                        dst_id[j] = comb[j].id;
                        dst_sc[j] = f32_to_bf16(comb[j].s);
                    }
                    g_fill[g] = uint32_t(take);
                }
            }
        }
    }
    buckets.clear();
    buckets.shrink_to_fit();

    // The scores are no longer needed: free them before linking level 0,
    // which is where the peak memory is reached.
    std::free(g_bf16);
    g_bf16 = nullptr;

    // 5. Level-0 linking directly from the codes
    std::vector<uint32_t> all(n);
    std::iota(all.begin(), all.end(), 0u);
    build_level_codes(
            codes,
            n,
            m_dims,
            code_size,
            cns,
            h,
            0,
            all,
            [&](uint32_t a, std::vector<uint32_t>& ids) {
                ids.assign(
                        g_ids + size_t(a) * k,
                        g_ids + size_t(a) * k + g_fill[a]);
            },
            threads);
    std::free(g_ids);
    std::free(g_fill);
    all.clear();
    all.shrink_to_fit();

    // 6. Upper levels via subset AMX/VNNI knn()
    build_upper_levels_codes(
            codes, n, m_dims, code_size, cns, h, k, vnni, threads);
}

} // namespace rabitq_batch
} // namespace faiss
