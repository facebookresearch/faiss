/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

/** Builds all levels of an HNSW graph at once from code-to-code neighbors.
 *
 * Internal header, included only by batch_build-avx512-spr.cpp.
 *
 * Level 0: knn.h computes the exact top-k of each row under the
 * code-to-code inner product, the list is pruned to 2M with the HNSW
 * heuristic (shrink_neighbor_list), reverse edges are added and the lists
 * that overflow are pruned again. Upper levels: brute-force top-k within the
 * level, then the same pruning to M and reverse edges.
 *
 * The similarity used for pruning is bitwise identical to
 * RaBitQuantizer's symmetric_dis for 8-bit inner product codes.
 */

#include <faiss/impl/HNSW.h>
#include <faiss/impl/RaBitQCodePairIP.h>
#include <faiss/impl/rabitq_batch/knn.h>
#include <faiss/utils/random.h>

#include <atomic>
#include <numeric>
#include <vector>

namespace faiss {
namespace rabitq_batch {

struct Ctx {
    const Data& d;
    double center_norm_sq;
};

/// sum(w_a w_b). vpdpbusd multiplies unsigned by signed bytes: computes
/// sum(w_a (w_b - 128)) then adds back 128 * W_a.
inline int32_t ww(const Data& d, uint32_t a, uint32_t b) {
    const __m512i flip = _mm512_set1_epi8(char(0x80));
    const size_t s = d.m.stride;
    __m512i acc = _mm512_setzero_si512();
    for (size_t c = 0; c < d.m.kc; ++c) {
        acc = _mm512_dpbusd_epi32(
                acc,
                _mm512_load_si512(d.w + size_t(a) * s + c * 64),
                _mm512_xor_si512(
                        _mm512_load_si512(d.w + size_t(b) * s + c * 64), flip));
    }
    return wrap(
            uint32_t(_mm512_reduce_add_epi32(acc)) +
            128u * uint32_t(d.wsum[a]));
}

/// Same value as symmetric_dis: smaller id first, exact integer dot, then
/// rabitq_ip::restore.
inline float sim(const Ctx& c, uint32_t i, uint32_t j) {
    if (i > j) {
        std::swap(i, j);
    }
    const int64_t dot = full_dot(c.d, i, j, ww(c.d, i, j));
    return rabitq_ip::restore(
            c.d.codes + size_t(i) * c.d.code_size,
            c.d.codes + size_t(j) * c.d.code_size,
            c.d.m.d,
            c.center_norm_sq,
            dot);
}

/** Similarity version of HNSW::shrink_neighbor_list. `sorted` is ordered by
 * decreasing similarity to the node; v is kept only if sim(v, node) is not
 * lower than its similarity to any neighbor already kept. If there are
 * fewer than max_size candidates, all are kept, as in HNSW. */
inline void shrink(
        const Ctx& c,
        const std::vector<Cand>& sorted,
        size_t max_size,
        std::vector<Cand>& out) {
    out.clear();
    if (sorted.size() < max_size) {
        out = sorted;
        return;
    }
    for (const Cand& v : sorted) {
        bool good = true;
        for (const Cand& u : out) {
            if (v.s < sim(c, u.id, v.id)) {
                good = false;
                break;
            }
        }
        if (good) {
            out.push_back(v);
            if (out.size() >= max_size) {
                return;
            }
        }
    }
}

/// Pruning and reverse edges of one level. nodes: the nodes of the level in
/// increasing order; cands(a, ids) fills the candidate ids of a.
template <class CandFn>
inline void build_level(
        const Ctx& c,
        HNSW& h,
        int level,
        const std::vector<uint32_t>& nodes,
        CandFn cands,
        size_t threads) {
    const size_t nl = nodes.size(), m = h.nb_neighbors(level);
    std::vector<int32_t> pos(c.d.n, -1);
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
        std::vector<Cand> list, out;
        std::vector<uint32_t> ids;
#pragma omp for schedule(dynamic, 256)
        for (size_t li = 0; li < nl; ++li) {
            const uint32_t a = nodes[li];
            cands(a, ids);
            list.clear();
            for (uint32_t b : ids) {
                list.push_back({sim(c, a, b), b});
            }
            std::sort(list.begin(), list.end(), better);
            shrink(c, list, m, out);
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
        std::vector<Cand> list, out;
#pragma omp for schedule(dynamic, 256)
        for (size_t li = 0; li < nl; ++li) {
            const uint32_t b = nodes[li];
            // arrival order is not deterministic, sort for reproducibility
            std::sort(rev.begin() + start[li], rev.begin() + start[li + 1]);
            list.assign(fwd.begin() + li * m, fwd.begin() + li * m + fcnt[li]);
            for (size_t r = start[li]; r < start[li + 1]; ++r) {
                const uint32_t a = rev[r];
                bool seen = false;
                for (size_t t = 0; t < fcnt[li] && !seen; ++t) {
                    seen = fwd[li * m + t].id == a;
                }
                if (!seen) {
                    list.push_back({sim(c, b, a), a});
                }
            }
            std::sort(list.begin(), list.end(), better);
            if (list.size() > m) {
                shrink(c, list, m, out);
            } else {
                out.swap(list);
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

/// Builds all levels; d.n, d.threads, d.m, d.codes and d.code_size must be
/// set, and the levels, offsets and entry point of h must be in place.
inline void build_graph(Data& d, double cns, HNSW& h, size_t k, bool vnni) {
    const Layout L = allocate(d, vnni);
    prepare(d);
    Rows st(d.pad, k, d.threads);
    knn(d, st, L, vnni);
    const Ctx c{d, cns};
    std::vector<uint32_t> all(d.n);
    std::iota(all.begin(), all.end(), 0u);
    build_level(
            c,
            h,
            0,
            all,
            [&](uint32_t a, std::vector<uint32_t>& ids) {
                ids.assign(
                        st.ids + size_t(a) * k,
                        st.ids + size_t(a) * k + st.fill[a]);
            },
            d.threads);
    for (int level = 1; level <= h.max_level; ++level) {
        std::vector<uint32_t> nodes;
        for (size_t i = 0; i < d.n; ++i) {
            if (h.levels[i] > level) {
                nodes.push_back(uint32_t(i));
            }
        }
        // brute-force top-k within the level: upper levels are small
        // (about n / 32 nodes at level 1)
        const size_t kk = std::min(k, nodes.size() - 1), nl = nodes.size();
        std::vector<uint32_t> top(nl * kk);
#pragma omp parallel num_threads(d.threads)
        {
            std::vector<Cand> row;
#pragma omp for schedule(dynamic, 16)
            for (size_t li = 0; li < nl; ++li) {
                row.clear();
                for (uint32_t b : nodes) {
                    if (b != nodes[li]) {
                        row.push_back({sim(c, nodes[li], b), b});
                    }
                }
                std::partial_sort(
                        row.begin(), row.begin() + kk, row.end(), better);
                for (size_t i = 0; i < kk; ++i) {
                    top[li * kk + i] = row[i].id;
                }
            }
        }
        std::vector<int32_t> pos(d.n, -1);
        for (size_t i = 0; i < nl; ++i) {
            pos[nodes[i]] = int32_t(i);
        }
        build_level(
                c,
                h,
                level,
                nodes,
                [&](uint32_t a, std::vector<uint32_t>& ids) {
                    ids.assign(
                            top.begin() + size_t(pos[a]) * kk,
                            top.begin() + size_t(pos[a] + 1) * kk);
                },
                d.threads);
    }
}

/** Same entry point as hnsw_add_vertices_deterministic: nodes are bucketed
 * by level, in increasing id order within a bucket; the top bucket is
 * shuffled with seed 789 and its first node becomes the entry point. */
inline void choose_entry(HNSW& h, size_t n) {
    std::vector<size_t> hist;
    for (size_t i = 0; i < n; ++i) {
        const size_t l = size_t(h.levels[i] - 1);
        if (l >= hist.size()) {
            hist.resize(l + 1, 0);
        }
        ++hist[l];
    }
    const size_t top = hist.size() - 1, i0 = n - hist[top], i1 = n;
    std::vector<int> order;
    for (size_t i = 0; i < n; ++i) {
        if (size_t(h.levels[i] - 1) == top) {
            order.push_back(int(i));
        }
    }
    RandomGenerator rng2(789);
    for (size_t j = i0; j < i1; ++j) {
        std::swap(order[j - i0], order[j - i0 + rng2.rand_int(int(i1 - j))]);
    }
    h.max_level = int(top);
    h.entry_point = order[0];
}

} // namespace rabitq_batch
} // namespace faiss
