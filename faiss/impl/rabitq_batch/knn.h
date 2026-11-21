/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

/** Exact top-k of the whole database under the code-to-code inner product.
 *
 * Internal header, included only by batch_build-avx512-spr.cpp.
 *
 * Each pair (a, b) is computed once and the same integer updates the lists
 * of both a and b. The database is padded to a multiple of 32 rows and cut
 * into P = 2 * nthreads row blocks; the off-diagonal block pairs are
 * scheduled as a round-robin tournament, so that in each round the threads
 * own disjoint pairs of row blocks; finally each thread processes the
 * diagonal blocks it owns.
 *
 * Lists are ordered by (score descending, id ascending), so the result does
 * not depend on the number of threads or on the order of arrival.
 *
 * Memory: the candidate side is packed on the fly into a per-thread buffer;
 * list buffers hold 1.25 k entries per row and are compacted to k ids at the
 * end, the unused pages being returned to the system.
 */

#include <faiss/impl/rabitq_batch/kernels.h>

#include <omp.h>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <new>
#include <utility>
#include <vector>

#ifdef __linux__
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace faiss {
namespace rabitq_batch {

template <class T>
T* alloc(size_t count) {
    void* p = std::aligned_alloc(64, (count * sizeof(T) + 63) / 64 * 64);
    if (!p) {
        throw std::bad_alloc();
    }
    return static_cast<T*>(p);
}

/// Intermediate values may overflow int32 and wrap around (two's
/// complement); the final dot is below 65025 * d < 2^31, so it is exact.
inline int32_t wrap(uint32_t x) {
    return int32_t(x);
}

/// Offset of the two per-vector factors in an 8-bit code: after the sign
/// bits, the 12-byte 1-bit factors and the 7-bit extra codes.
inline size_t factor_offset(size_t d) {
    return (d + 7) / 8 + 12 + (7 * d + 7) / 8;
}

struct Data {
    Dims m;
    /// pad: number of rows after padding, the padded rows are all 0
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
    /// number of row blocks P, rows per block, groups of 32 rows per block
    size_t blocks, block_rows, chunks;
};

/// Pads to the row-block size and allocates; d.n, d.threads and d.m must be
/// set.
inline Layout allocate(Data& d, bool vnni) {
    Layout L;
    L.blocks = 2 * d.threads;
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

/// Decodes w, the byte sum of each row and the two factors; for VNNI also
/// stores w - 128. The candidate side is packed later, when used.
inline void prepare(Data& d) {
    const Dims& m = d.m;
    const size_t off = factor_offset(m.d);
    std::memset(d.w + d.n * m.stride, 0, (d.pad - d.n) * m.stride);
#pragma omp parallel for num_threads(d.threads) schedule(static)
    for (size_t i = 0; i < d.n; ++i) {
        const uint8_t* code = d.codes + i * d.code_size;
        __m512i sum = _mm512_setzero_si512();
        for (size_t c = 0; c < m.kc; ++c) {
            const __m512i w = decode64(code, m.d, c);
            _mm512_store_si512(d.w + i * m.stride + c * 64, w);
            sum = _mm512_add_epi64(
                    sum, _mm512_sad_epu8(w, _mm512_setzero_si512()));
        }
        d.wsum[i] = int32_t(_mm512_reduce_add_epi64(sum));
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

/// Packs the groups of 16 columns starting at b0 into out (m.block bytes per
/// group): a transpose of 16 x 16 words of 4 bytes.
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

/** Per-row candidate lists with a buffer of cap = k + k / 4 entries. When a
 * buffer is full it keeps its top k and raises its threshold; the threshold
 * only drops candidates that cannot enter the top k, so the final lists are
 * the top k in the total order of better(), whatever cap is. A row is only
 * written by one thread at a time. */
struct Rows {
    size_t count, k, cap;
    float* thr;
    uint32_t* fill;
    Cand* buf;
    /// after compact(), k ids per row, sharing the memory of buf
    uint32_t* ids = nullptr;
    Rows(size_t count, size_t k, size_t threads)
            : count(count),
              k(k),
              cap(k + std::max<size_t>(1, k / 4)),
              thr(alloc<float>(count)),
              fill(alloc<uint32_t>(count)),
              buf(alloc<Cand>(count * cap)) {
#pragma omp parallel for if (threads > 1) num_threads(threads) schedule(static)
        for (size_t i = 0; i < count; ++i) {
            thr[i] = -INFINITY;
            fill[i] = 0;
            std::memset(buf + i * cap, 0, cap * sizeof(Cand));
        }
    }
    Rows(const Rows&) = delete;
    ~Rows() {
        std::free(thr);
        std::free(fill);
        std::free(buf);
    }
    void push(size_t i, float s, uint32_t id) {
        Cand* b = buf + i * cap;
        b[fill[i]++] = {s, id};
        if (fill[i] == cap) {
            std::nth_element(b, b + k - 1, b + cap, better);
            fill[i] = k;
            thr[i] = b[k - 1].s;
        }
    }
    /// Sorts the top k of row i in place, at buf + i * cap, length fill[i].
    void settle(size_t i) {
        Cand* b = buf + i * cap;
        if (fill[i] > k) {
            std::nth_element(b, b + k - 1, b + fill[i], better);
            fill[i] = k;
        }
        std::sort(b, b + fill[i], better);
    }
    /// After all rows are settled, compacts the lists in place to k ids per
    /// row (row i goes to ids + i * k, always before what is still to be
    /// read) and returns the unused pages. Only ids is valid afterwards.
    void compact(size_t n) {
        ids = reinterpret_cast<uint32_t*>(buf);
        std::vector<uint32_t> row(k);
        for (size_t i = 0; i < n; ++i) {
            const Cand* b = buf + i * cap;
            for (size_t t = 0; t < fill[i]; ++t) {
                row[t] = b[t].id;
            }
            std::memcpy(ids + i * k, row.data(), fill[i] * sizeof(uint32_t));
        }
#ifdef __linux__
        const size_t page = size_t(sysconf(_SC_PAGESIZE));
        const uintptr_t used =
                (reinterpret_cast<uintptr_t>(ids + n * k) + page - 1) / page *
                page;
        const uintptr_t end =
                reinterpret_cast<uintptr_t>(buf + count * cap) / page * page;
        if (end > used) {
            madvise(reinterpret_cast<void*>(used), end - used, MADV_DONTNEED);
        }
#endif
    }
};

/// Converts the integer core to the full code dot product:
/// 4 sum(w_a w_b) - 510 (W_a + W_b) + 65025 d, equal to rabitq_ip::dot_scalar.
inline int64_t full_dot(const Data& d, size_t a, size_t b, int32_t core) {
    return 4 * int64_t(core) - 510 * int64_t(d.wsum[a] + d.wsum[b]) +
            65025 * int64_t(d.m.d);
}

/// Round-robin tournament: pair t of round `round`. The last block is fixed
/// and the others rotate; the P / 2 pairs of a round are disjoint.
inline std::pair<size_t, size_t> pairing(size_t P, size_t round, size_t t) {
    const size_t m = P - 1;
    if (t == 0) {
        return {round, m};
    }
    return {(round + t) % m, (round + m - t) % m};
}

/// Updates both sides of a 32 x 32 block: row a receives candidate b with
/// key f0_b + f1_a f1_b dot / 4, row b receives candidate a with key
/// f0_a + f1_a f1_b dot / 4.
inline void post2(
        const Data& d,
        Rows& st,
        size_t a0,
        size_t b0,
        bool diagonal,
        const int32_t* cbuf) {
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
            if (diagonal) { // diagonal groups only need b > a
                valid &= _mm512_cmpgt_epi32_mask(
                        _mm512_add_epi32(_mm512_set1_epi32(int32_t(cb)), lane),
                        _mm512_set1_epi32(int32_t(a)));
            }
            if (!valid) {
                continue;
            }
            const __m512i x = _mm512_loadu_si512(cbuf + r * 32 + 16 * h);
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

/// sum(w_a w_b) for the 32 rows starting at a0 and the 32 columns starting
/// at b0, accumulated over `parts` slices of the dimensions.
inline void block(
        const Data& d,
        bool vnni,
        size_t a0,
        size_t b0,
        int32_t* out,
        size_t parts) {
    const Dims& m = d.m;
    uint8_t* bp = alloc<uint8_t>(2 * m.block);
    pack(d, b0, 2, bp);
    for (size_t part = 0; part < parts; ++part) {
        const size_t k0 = m.kc * part / parts, k1 = m.kc * (part + 1) / parts;
        if (vnni) {
            vnni_block_part(
                    m,
                    d.signed_rows + a0 * m.stride,
                    bp,
                    d.wsum + b0,
                    out,
                    k0,
                    k1,
                    part == 0);
        } else {
            amx_block_part(m, d.w + a0 * m.stride, bp, out, k0, k1, part == 0);
        }
    }
    std::free(bp);
}

inline size_t parts_for(const Data& d, bool vnni) {
    return std::min<size_t>(vnni ? 4 : 2, d.m.kc);
}

/** Row block I against row block J. Each segment of 16 groups of 32 columns
 * is packed into the per-thread segbuf (kept in L2); within a segment, each
 * group of 32 rows of I is accumulated slice by slice, then the lists are
 * updated. Diagonal blocks only compute the upper triangle. segbuf holds at
 * least 32 * m.block bytes. */
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

/// P / 2 pairs per round, thread t takes pairs t, t + T, t + 2T...; then each
/// thread processes its 2 diagonal blocks, and the lists are sorted.
inline void knn(const Data& d, Rows& st, const Layout& L, bool vnni) {
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
        for (size_t b = me * 2; b < (me + 1) * 2; ++b) {
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
    st.compact(d.n);
}

} // namespace rabitq_batch
} // namespace faiss
