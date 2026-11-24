/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

/** Exact top-k of a vector set under the 8-bit RaBitQ code-to-code inner
 * product, computed with AMX-INT8 or AVX512-VNNI.
 *
 *  - Rows keeps a strict length-k min-heap per vector, so the admission
 *    threshold is always tight; compact() sorts the heaps in parallel.
 *  - post2_full() keeps the parameters of the 32 columns of a block in
 *    registers across its 32 rows; pushes take a cold out-of-line path.
 *  - Row blocks are at most max_block_rows rows high, which bounds the
 *    packing overhead on small inputs and keeps the thresholds in L2 on
 *    large ones.
 *  - prepare_subset() decodes an arbitrary subset of the codes, so the same
 *    kernel serves the upper levels and the buckets of a partitioned build.
 */

#include <faiss/impl/rabitq_batch/kernels.h>

#include <omp.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <new>
#include <utility>
#include <vector>

#ifdef __linux__
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace faiss {
namespace rabitq_batch {

/// Upper bound on the height of a row block.
constexpr size_t max_block_rows = 24576;

template <class T>
T* alloc(size_t count) {
    size_t bytes = (count * sizeof(T) + 63) / 64 * 64;
    if (bytes == 0) {
        bytes = 64;
    }
    void* p = std::aligned_alloc(64, bytes);
    if (!p) {
        throw std::bad_alloc();
    }
    return static_cast<T*>(p);
}

inline int32_t wrap(uint32_t x) {
    return int32_t(x);
}

inline size_t factor_offset(size_t d) {
    return (d + 7) / 8 + 12 + (7 * d + 7) / 8;
}

struct Data {
    Dims m;
    size_t n = 0, pad = 0, threads = 0, code_size = 0;
    const uint8_t* codes = nullptr;
    uint8_t *w = nullptr, *signed_rows = nullptr;
    int32_t *wsum = nullptr, *colint = nullptr;
    float *f0 = nullptr, *f1 = nullptr;
    Data() = default;
    Data(const Data&) = delete;
    ~Data() {
        for (void* p :
             {(void*)w,
              (void*)signed_rows,
              (void*)wsum,
              (void*)colint,
              (void*)f0,
              (void*)f1}) {
            std::free(p);
        }
    }
};

struct Layout {
    size_t blocks, block_rows, chunks;
};

inline Layout allocate(Data& d, bool vnni) {
    Layout L;
    const size_t min_blocks = 2 * d.threads;
    const size_t max_br = max_block_rows;
    size_t mult = 1;
    if (d.n > min_blocks * max_br) {
        mult = (d.n + min_blocks * max_br - 1) / (min_blocks * max_br);
    }
    L.blocks = min_blocks * mult;
    const size_t unit = L.blocks * 32;
    d.pad = (d.n + unit - 1) / unit * unit;
    L.block_rows = d.pad / L.blocks;
    L.chunks = L.block_rows / 32;
    d.w = alloc<uint8_t>(d.pad * d.m.stride);
    d.signed_rows = vnni ? alloc<uint8_t>(d.pad * d.m.stride) : nullptr;
    d.wsum = alloc<int32_t>(d.pad);
    d.colint = alloc<int32_t>(d.pad);
    d.f0 = alloc<float>(d.pad);
    d.f1 = alloc<float>(d.pad);
    std::fill(d.wsum, d.wsum + d.pad, 0);
    std::fill(d.colint, d.colint + d.pad, 0);
    std::fill(d.f0, d.f0 + d.pad, 0.f);
    std::fill(d.f1, d.f1 + d.pad, 0.f);
    return L;
}

inline int32_t decode_row(const uint8_t* code, const Dims& m, uint8_t* out_w) {
    __m512i sum = _mm512_setzero_si512();
    for (size_t c = 0; c < m.kc; ++c) {
        const __m512i w = decode64(code, m.d, c);
        _mm512_store_si512(out_w + c * 64, w);
        sum = _mm512_add_epi64(sum, _mm512_sad_epu8(w, _mm512_setzero_si512()));
    }
    return int32_t(_mm512_reduce_add_epi64(sum));
}

inline void prepare(Data& d) {
    const Dims& m = d.m;
    const size_t off = factor_offset(m.d);
    std::memset(d.w + d.n * m.stride, 0, (d.pad - d.n) * m.stride);
#pragma omp parallel for num_threads(d.threads) schedule(static)
    for (size_t i = 0; i < d.n; ++i) {
        const uint8_t* code = d.codes + i * d.code_size;
        d.wsum[i] = decode_row(code, m, d.w + i * m.stride);
        d.colint[i] =
                wrap(0u - 510u * uint32_t(d.wsum[i]) + 65025u * uint32_t(m.d));
        std::memcpy(d.f0 + i, code + off, 4);
        std::memcpy(d.f1 + i, code + off + 4, 4);
    }
    if (d.signed_rows) {
#pragma omp parallel for num_threads(d.threads) schedule(static)
        for (size_t i = 0; i < d.pad * m.stride; ++i) {
            d.signed_rows[i] = d.w[i] ^ 0x80;
        }
    }
}

inline void prepare_subset(Data& d, const uint32_t* gids) {
    const Dims& m = d.m;
    const size_t off = factor_offset(m.d);
    std::memset(d.w + d.n * m.stride, 0, (d.pad - d.n) * m.stride);
#pragma omp parallel for num_threads(d.threads) schedule(static)
    for (size_t i = 0; i < d.n; ++i) {
        const uint8_t* code = d.codes + size_t(gids[i]) * d.code_size;
        d.wsum[i] = decode_row(code, m, d.w + i * m.stride);
        d.colint[i] =
                wrap(0u - 510u * uint32_t(d.wsum[i]) + 65025u * uint32_t(m.d));
        std::memcpy(d.f0 + i, code + off, 4);
        std::memcpy(d.f1 + i, code + off + 4, 4);
    }
    if (d.signed_rows) {
#pragma omp parallel for num_threads(d.threads) schedule(static)
        for (size_t i = 0; i < d.pad * m.stride; ++i) {
            d.signed_rows[i] = d.w[i] ^ 0x80;
        }
    }
}

inline void pack(const Data& d, size_t b0, size_t groups, uint8_t* out) {
    const Dims& m = d.m;
    for (size_t g = 0; g < groups; ++g) {
        for (size_t c = 0; c < m.kc; ++c) {
            for (size_t r = 0; r < 16; ++r) {
                for (size_t col = 0; col < 16; ++col) {
                    std::memcpy(
                            out + g * m.block + c * 1024 + r * 64 + col * 4,
                            d.w + (b0 + g * 16 + col) * m.stride + c * 64 +
                                    r * 4,
                            4);
                }
            }
        }
    }
}

struct Cand {
    float s;
    uint32_t id;
};

inline bool better(const Cand& a, const Cand& b) {
    return a.s > b.s || (a.s == b.s && a.id < b.id);
}

struct Rows {
    size_t count, k;
    float* thr;
    uint32_t* fill;
    Cand* buf;
    /// after compact(), k ids per row, sharing the memory of buf
    uint32_t* ids = nullptr;
    Rows(size_t n, size_t kk, size_t threads)
            : count(n),
              k(kk),
              thr(alloc<float>(n)),
              fill(alloc<uint32_t>(n)),
              buf(alloc<Cand>(n * kk)) {
#pragma omp parallel for if (threads > 1) num_threads(threads) schedule(static)
        for (size_t i = 0; i < count; ++i) {
            thr[i] = -INFINITY;
            fill[i] = 0;
            std::memset(buf + i * k, 0, k * sizeof(Cand));
        }
    }
    Rows(const Rows&) = delete;
    ~Rows() {
        std::free(thr);
        std::free(fill);
        std::free(buf);
    }
    __attribute__((noinline, cold)) void init_heap(size_t i) {
        Cand* b = buf + i * k;
        std::make_heap(b, b + k, better);
        thr[i] = b[0].s;
    }
    inline void push(size_t i, float s, uint32_t id) {
        Cand* b = buf + i * k;
        uint32_t f = fill[i];
        if (__builtin_expect(f < k, 0)) {
            b[f] = {s, id};
            fill[i] = ++f;
            if (f == k) {
                init_heap(i);
            }
            return;
        }
        const Cand c{s, id};
        if (!better(c, b[0])) {
            return;
        }
        size_t p = 0;
        const size_t half = k >> 1;
        while (p < half) {
            size_t ch = (p << 1) + 1;
            if (ch + 1 < k && better(b[ch], b[ch + 1])) {
                ++ch;
            }
            if (!better(c, b[ch])) {
                break;
            }
            b[p] = b[ch];
            p = ch;
        }
        b[p] = c;
        thr[i] = b[0].s;
    }
    inline void settle(size_t i) {
        Cand* b = buf + i * k;
        std::sort(b, b + fill[i], better);
    }
    void compact(size_t n, size_t threads) {
        ids = reinterpret_cast<uint32_t*>(buf);
        const size_t base = std::min<size_t>(n, 1024);
        {
            std::vector<uint32_t> row(k);
            for (size_t i = 0; i < base; ++i) {
                const Cand* b = buf + i * k;
                const uint32_t f = fill[i];
                for (size_t t = 0; t < f; ++t) {
                    row[t] = b[t].id;
                }
                std::memcpy(ids + i * k, row.data(), f * sizeof(uint32_t));
            }
        }
        for (size_t lo = base; lo < n;) {
            const size_t hi = std::min<size_t>(n, lo * 2);
#pragma omp parallel for if (threads > 1 && hi - lo >= 2048) \
        num_threads(threads) schedule(static)
            for (size_t i = lo; i < hi; ++i) {
                const Cand* b = buf + i * k;
                uint32_t* dst = ids + i * k;
                const uint32_t f = fill[i];
                for (size_t t = 0; t < f; ++t) {
                    dst[t] = b[t].id;
                }
            }
            lo = hi;
        }
#ifdef __linux__
        const size_t page = size_t(sysconf(_SC_PAGESIZE));
        const uintptr_t used =
                (reinterpret_cast<uintptr_t>(ids + n * k) + page - 1) / page *
                page;
        const uintptr_t end =
                reinterpret_cast<uintptr_t>(buf + count * k) / page * page;
        if (end > used) {
            madvise(reinterpret_cast<void*>(used), end - used, MADV_DONTNEED);
        }
#endif
    }
};

inline int64_t full_dot(const Data& d, size_t a, size_t b, int32_t core) {
    return 4 * int64_t(core) - 510 * int64_t(d.wsum[a] + d.wsum[b]) +
            65025 * int64_t(d.m.d);
}

inline std::pair<size_t, size_t> pairing(size_t P, size_t round, size_t t) {
    const size_t m = P - 1;
    if (t == 0) {
        return {round, m};
    }
    return {(round + t) % m, (round + m - t) % m};
}

__attribute__((noinline, cold)) inline void post2_flush_half(
        Rows& st,
        size_t a,
        size_t cb,
        __mmask16 mr,
        __mmask16 mc,
        __m512 key_row,
        __m512 key_col) {
    if (mr) {
        alignas(64) float keys[16];
        _mm512_store_ps(keys, key_row);
        for (; mr; mr &= mr - 1) {
            const unsigned i = __builtin_ctz(mr);
            if (keys[i] >= st.thr[a]) {
                st.push(a, keys[i], uint32_t(cb + i));
            }
        }
    }
    if (mc) {
        alignas(64) float keys[16];
        _mm512_store_ps(keys, key_col);
        for (; mc; mc &= mc - 1) {
            const unsigned i = __builtin_ctz(mc);
            if (keys[i] >= st.thr[cb + i]) {
                st.push(cb + i, keys[i], uint32_t(a));
            }
        }
    }
}

/// Fast register-hoisted post2 for full (non-diagonal, non-tail) 32x32 blocks.
/// Hoists all 32 column parameters (colint, f1, f0, thr) into ZMM registers,
/// folds exact 0.25f into row f1, and moves rare pushes to a noinline cold
/// helper.
inline void post2_full(
        const Data& d,
        Rows& st,
        size_t a0,
        size_t b0,
        const int32_t* cbuf) {
    const __m512i ci0 = _mm512_load_si512(d.colint + b0);
    const __m512i ci1 = _mm512_load_si512(d.colint + b0 + 16);
    const __m512 f1_0 = _mm512_load_ps(d.f1 + b0);
    const __m512 f1_1 = _mm512_load_ps(d.f1 + b0 + 16);
    const __m512 f0_0 = _mm512_load_ps(d.f0 + b0);
    const __m512 f0_1 = _mm512_load_ps(d.f0 + b0 + 16);
    __m512 th_b0 = _mm512_load_ps(st.thr + b0);
    __m512 th_b1 = _mm512_load_ps(st.thr + b0 + 16);

    for (size_t r = 0; r < 32; ++r) {
        const size_t a = a0 + r;
        const __m512i rowint =
                _mm512_set1_epi32(wrap(0u - 510u * uint32_t(d.wsum[a])));
        const __m512 faq = _mm512_set1_ps(d.f1[a] * 0.25f);
        const __m512 f0a = _mm512_set1_ps(d.f0[a]);
        __m512 th_a = _mm512_set1_ps(st.thr[a]);

        // Half 0
        {
            const __m512i x = _mm512_load_si512(cbuf + r * 32);
            const __m512i dot = _mm512_add_epi32(
                    _mm512_slli_epi32(x, 2), _mm512_add_epi32(rowint, ci0));
            const __m512 fab = _mm512_mul_ps(faq, f1_0);
            const __m512 fdot = _mm512_cvtepi32_ps(dot);
            const __m512 key_row = _mm512_fmadd_ps(fab, fdot, f0_0);
            const __m512 key_col = _mm512_fmadd_ps(fab, fdot, f0a);
            const __mmask16 mr = _mm512_cmp_ps_mask(key_row, th_a, _CMP_GE_OQ);
            const __mmask16 mc = _mm512_cmp_ps_mask(key_col, th_b0, _CMP_GE_OQ);
            if (__builtin_expect((mr | mc) != 0, 0)) {
                post2_flush_half(st, a, b0, mr, mc, key_row, key_col);
                if (mr)
                    th_a = _mm512_set1_ps(st.thr[a]);
                if (mc)
                    th_b0 = _mm512_load_ps(st.thr + b0);
            }
        }
        // Half 1
        {
            const __m512i x = _mm512_load_si512(cbuf + r * 32 + 16);
            const __m512i dot = _mm512_add_epi32(
                    _mm512_slli_epi32(x, 2), _mm512_add_epi32(rowint, ci1));
            const __m512 fab = _mm512_mul_ps(faq, f1_1);
            const __m512 fdot = _mm512_cvtepi32_ps(dot);
            const __m512 key_row = _mm512_fmadd_ps(fab, fdot, f0_1);
            const __m512 key_col = _mm512_fmadd_ps(fab, fdot, f0a);
            const __mmask16 mr = _mm512_cmp_ps_mask(key_row, th_a, _CMP_GE_OQ);
            const __mmask16 mc = _mm512_cmp_ps_mask(key_col, th_b1, _CMP_GE_OQ);
            if (__builtin_expect((mr | mc) != 0, 0)) {
                post2_flush_half(st, a, b0 + 16, mr, mc, key_row, key_col);
                if (mc)
                    th_b1 = _mm512_load_ps(st.thr + b0 + 16);
            }
        }
    }
}

inline void post2(
        const Data& d,
        Rows& st,
        size_t a0,
        size_t b0,
        bool diagonal,
        const int32_t* cbuf) {
    if (!diagonal && a0 + 32 <= d.n && b0 + 32 <= d.n) {
        post2_full(d, st, a0, b0, cbuf);
        return;
    }
    const __m512 quarter = _mm512_set1_ps(0.25f);
    const __m512i lane = _mm512_set_epi32(
            15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0);
    for (size_t r = 0; r < 32; ++r) {
        const size_t a = a0 + r;
        if (a >= d.n) {
            break;
        }
        const __m512i rowint =
                _mm512_set1_epi32(wrap(0u - 510u * uint32_t(d.wsum[a])));
        const __m512 fa = _mm512_set1_ps(d.f1[a]),
                     f0a = _mm512_set1_ps(d.f0[a]);
        for (size_t h = 0; h < 2; ++h) {
            const size_t cb = b0 + 16 * h;
            __mmask16 valid = cb >= d.n ? 0
                    : cb + 16 <= d.n    ? 0xFFFF
                                        : __mmask16((1u << (d.n - cb)) - 1);
            if (diagonal) {
                valid &= _mm512_cmpgt_epi32_mask(
                        _mm512_add_epi32(_mm512_set1_epi32(int32_t(cb)), lane),
                        _mm512_set1_epi32(int32_t(a)));
            }
            if (!valid) {
                continue;
            }
            const __m512i x = _mm512_load_si512(cbuf + r * 32 + 16 * h);
            const __m512i dot = _mm512_add_epi32(
                    _mm512_slli_epi32(x, 2),
                    _mm512_add_epi32(
                            rowint, _mm512_loadu_si512(d.colint + cb)));
            const __m512 fab = _mm512_mul_ps(fa, _mm512_loadu_ps(d.f1 + cb));
            const __m512 dq = _mm512_mul_ps(_mm512_cvtepi32_ps(dot), quarter);
            const __m512 key_row =
                    _mm512_fmadd_ps(fab, dq, _mm512_loadu_ps(d.f0 + cb));
            const __m512 key_col = _mm512_fmadd_ps(fab, dq, f0a);
            __mmask16 mr =
                    valid &
                    _mm512_cmp_ps_mask(
                            key_row, _mm512_set1_ps(st.thr[a]), _CMP_GE_OQ);
            __mmask16 mc =
                    valid &
                    _mm512_cmp_ps_mask(
                            key_col, _mm512_loadu_ps(st.thr + cb), _CMP_GE_OQ);
            if (mr) {
                alignas(64) float keys[16];
                _mm512_store_ps(keys, key_row);
                for (; mr; mr &= mr - 1) {
                    const unsigned i = __builtin_ctz(mr);
                    if (keys[i] >= st.thr[a]) {
                        st.push(a, keys[i], uint32_t(cb + i));
                    }
                }
            }
            if (mc) {
                alignas(64) float keys[16];
                _mm512_store_ps(keys, key_col);
                for (; mc; mc &= mc - 1) {
                    const unsigned i = __builtin_ctz(mc);
                    if (keys[i] >= st.thr[cb + i]) {
                        st.push(cb + i, keys[i], uint32_t(a));
                    }
                }
            }
        }
    }
}

inline size_t parts_for(const Data& d, bool vnni) {
    return std::min<size_t>(vnni ? 4 : 2, d.m.kc);
}

inline void tile(
        const Data& d,
        Rows& st,
        const Layout& L,
        bool vnni,
        size_t I,
        size_t J,
        bool diagonal,
        int32_t* cseg,
        uint8_t* segbuf) {
    constexpr size_t seg = 16;
    const Dims& m = d.m;
    const size_t parts = parts_for(d, vnni);
    const size_t rbase = I * L.block_rows, cbase = J * L.block_rows;
    for (size_t cs = 0; cs < L.chunks; cs += seg) {
        const size_t ce = std::min(cs + seg, L.chunks);
        pack(d, cbase + cs * 32, 2 * (ce - cs), segbuf);
        for (size_t rc = 0; rc < L.chunks; ++rc) {
            const size_t c_first = diagonal ? std::max(cs, rc) : cs;
            if (c_first >= ce) {
                continue;
            }
            const size_t a0 = rbase + rc * 32;
            const uint8_t* rows =
                    vnni ? d.signed_rows + a0 * m.stride : d.w + a0 * m.stride;
            for (size_t part = 0; part < parts; ++part) {
                const size_t k0 = m.kc * part / parts,
                             k1 = m.kc * (part + 1) / parts;
                for (size_t cc = c_first; cc < ce; ++cc) {
                    const size_t b0 = cbase + cc * 32;
                    int32_t* out = cseg + (cc - cs) * 1024;
                    const uint8_t* bp = segbuf + (cc - cs) * 2 * m.block;
                    if (vnni) {
                        vnni_block_part(
                                m,
                                rows,
                                bp,
                                d.wsum + b0,
                                out,
                                k0,
                                k1,
                                part == 0);
                    } else {
                        amx_block_part(m, rows, bp, out, k0, k1, part == 0);
                    }
                }
            }
            for (size_t cc = c_first; cc < ce; ++cc) {
                post2(d,
                      st,
                      a0,
                      cbase + cc * 32,
                      diagonal && cc == rc,
                      cseg + (cc - cs) * 1024);
            }
        }
    }
}

/** Exact top-k of all d.n vectors under the code-to-code inner product,
 * left in st. Block pairs are visited in round-robin rounds so that each
 * pair is computed once and updates the lists of both sides. If do_compact
 * is false, the per-vector heaps are left unsorted. */
inline void knn(
        const Data& d,
        Rows& st,
        const Layout& L,
        bool vnni,
        bool do_compact = true) {
#pragma omp parallel num_threads(d.threads)
    {
        const size_t me = omp_get_thread_num(), T = d.threads;
        if (!vnni) {
            amx_configure();
        }
        int32_t* cseg = alloc<int32_t>(16 * 1024);
        uint8_t* segbuf = alloc<uint8_t>(32 * d.m.block);
        for (size_t round = 0; round + 1 < L.blocks; ++round) {
            for (size_t q = me; q < L.blocks / 2; q += T) {
                const auto [I, J] = pairing(L.blocks, round, q);
                tile(d, st, L, vnni, I, J, false, cseg, segbuf);
            }
#pragma omp barrier
        }
        for (size_t b = me; b < L.blocks; b += T) {
            tile(d, st, L, vnni, b, b, true, cseg, segbuf);
        }
        std::free(cseg);
        std::free(segbuf);
        if (!vnni) {
            amx_release();
        }
    }
#pragma omp parallel for num_threads(d.threads) schedule(static)
    for (size_t i = 0; i < d.n; ++i) {
        st.settle(i);
    }
    if (do_compact) {
        st.compact(d.n, d.threads);
    }
}

} // namespace rabitq_batch
} // namespace faiss
