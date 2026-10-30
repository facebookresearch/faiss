/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Native RVV implementations of the approx_topk kernels.
//
// bs_addn vectorizes over the buckets of one bucket block (rvv_plan picks
// LMUL/chunk from the hardware VLEN), replacing the AVX2/NEON 8-lane
// grouping and its per-vector index fixup. accum_and_*_tab strip-mines K
// with a runtime vsetvl_e32m8, independent of NK.
//
// Both kernels produce bit-identical results to the SIMDLevel::NONE
// reference; the tie and max-side rules are documented at scan_block.

#ifdef COMPILE_SIMD_RISCV_RVV

#include <cstddef>
#include <cstdint>
#include <limits>

#include <riscv_vector.h>

#include <faiss/impl/approx_topk/approx_topk.h>
#include <faiss/impl/approx_topk/rq_beam_search_tab-inl.h>
#include <faiss/utils/Heap.h>

namespace faiss {

namespace {

// ---------------------------------------------------------------------------
// Per-LMUL bucket operators.
//
// RVV intrinsic vector types are sizeless: they are legal as locals, function
// parameters/returns and references, but not as struct members or array
// elements. The traits below therefore only carry type aliases and static
// functions; all state lives in named locals in scan_chunk.
//
// vmerge(op1, op2, mask) selects op2 when the mask bit is set, i.e.
// "mask ? op2 : op1".
// ---------------------------------------------------------------------------

// INDEX LMUL: every index vector has the same lane count as the value
// vectors but SEW=16, so its LMUL (and register footprint) is halved.
// SEW/LMUL is unchanged, hence the mask type (vbool) is identical.
#define FAISS_RVV_BUCKET_OPS(NAME, LM, VF, VI, VB, MB, IL, II)                \
    struct NAME {                                                             \
        using vf = VF;                                                        \
        using vi = VI;                                                        \
        using vi16 = II;                                                      \
        using vb = VB;                                                        \
                                                                              \
        static inline size_t setvl(size_t avl) {                              \
            return __riscv_vsetvl_e32##LM(avl);                               \
        }                                                                     \
        static inline vf splat_f(float x, size_t vl) {                        \
            return __riscv_vfmv_v_f_f32##LM(x, vl);                           \
        }                                                                     \
        static inline vi vid(size_t vl) {                                     \
            return __riscv_vid_v_u32##LM(vl);                                 \
        }                                                                     \
        static inline vi add_x(vi a, uint32_t x, size_t vl) {                 \
            return __riscv_vadd_vx_u32##LM(a, x, vl);                         \
        }                                                                     \
        static inline vf load_f(const float* p, size_t vl) {                  \
            return __riscv_vle32_v_f32##LM(p, vl);                            \
        }                                                                     \
        static inline void store_f(float* p, vf v, size_t vl) {               \
            __riscv_vse32_v_f32##LM(p, v, vl);                                \
        }                                                                     \
        static inline void store_u(uint32_t* p, vi v, size_t vl) {            \
            __riscv_vse32_v_u32##LM(p, v, vl);                                \
        }                                                                     \
        static inline vb lt(vf a, vf b, size_t vl) {                          \
            return __riscv_vmflt_vv_f32##LM##_b##MB(a, b, vl);                \
        }                                                                     \
        /* minimumNumber; only feeds the dead max side of the last depth. */  \
        static inline vf vmin(vf a, vf b, size_t vl) {                        \
            return __riscv_vfmin_vv_f32##LM(a, b, vl);                        \
        }                                                                     \
        /* mask ? t : f, via the integer merge on the float bit patterns. */  \
        static inline vf select_f(vb m, vf t, vf f, size_t vl) {              \
            vi fu = __riscv_vreinterpret_v_f32##LM##_u32##LM(f);              \
            vi tu = __riscv_vreinterpret_v_f32##LM##_u32##LM(t);              \
            vi r = __riscv_vmerge_vvm_u32##LM(fu, tu, m, vl);                 \
            return __riscv_vreinterpret_v_u32##LM##_f32##LM(r);               \
        }                                                                     \
        static inline vi select_i(vb m, vi t, vi f, size_t vl) {              \
            return __riscv_vmerge_vvm_u32##LM(f, t, m, vl);                   \
        }                                                                     \
        /* u16 index state: same lane count, half the register groups. */     \
        static inline vi16 idx_narrow(vi a, size_t vl) {                      \
            return __riscv_vncvt_x_x_w_u16##IL(a, vl);                        \
        }                                                                     \
        static inline vi16 idx_add(vi16 a, uint32_t x, size_t vl) {           \
            return __riscv_vadd_vx_u16##IL(a, (uint16_t)x, vl);               \
        }                                                                     \
        static inline vi16 idx_select(vb m, vi16 t, vi16 f, size_t vl) {      \
            return __riscv_vmerge_vvm_u16##IL(f, t, m, vl);                   \
        }                                                                     \
        static inline void idx_store(uint32_t* p, vi16 v, size_t vl) {        \
            __riscv_vse32_v_u32##LM(p, __riscv_vzext_vf2_u32##LM(v, vl), vl); \
        }                                                                     \
    };

// Mask ratio is SEW/LMUL: m1 -> 32, m2 -> 16, m4 -> 8, m8 -> 4.
FAISS_RVV_BUCKET_OPS(
        rvv_e32m1,
        m1,
        vfloat32m1_t,
        vuint32m1_t,
        vbool32_t,
        32,
        mf2,
        vuint16mf2_t)
FAISS_RVV_BUCKET_OPS(
        rvv_e32m2,
        m2,
        vfloat32m2_t,
        vuint32m2_t,
        vbool16_t,
        16,
        m1,
        vuint16m1_t)
FAISS_RVV_BUCKET_OPS(
        rvv_e32m4,
        m4,
        vfloat32m4_t,
        vuint32m4_t,
        vbool8_t,
        8,
        m2,
        vuint16m2_t)
FAISS_RVV_BUCKET_OPS(
        rvv_e32m8,
        m8,
        vfloat32m8_t,
        vuint32m8_t,
        vbool4_t,
        4,
        m4,
        vuint16m4_t)

#undef FAISS_RVV_BUCKET_OPS

// Index-state ops: u32 (identity, always valid) or u16 (caller guarantees
// every index fits in 16 bits). Values/masks are always the O ops.
template <typename O, bool Narrow>
struct rvv_index_ops;

template <typename O>
struct rvv_index_ops<O, false> {
    using t = typename O::vi;
    static inline t narrow(t a, size_t) {
        return a;
    }
    static inline t add(t a, uint32_t x, size_t vl) {
        return O::add_x(a, x, vl);
    }
    static inline t select(typename O::vb m, t t_, t f, size_t vl) {
        return O::select_i(m, t_, f, vl);
    }
    static inline void store(uint32_t* p, t v, size_t vl) {
        O::store_u(p, v, vl);
    }
};

template <typename O>
struct rvv_index_ops<O, true> {
    using t = typename O::vi16;
    static inline t narrow(typename O::vi a, size_t vl) {
        return O::idx_narrow(a, vl);
    }
    static inline t add(t a, uint32_t x, size_t vl) {
        return O::idx_add(a, x, vl);
    }
    static inline t select(typename O::vb m, t t_, t f, size_t vl) {
        return O::idx_select(m, t_, f, vl);
    }
    static inline void store(uint32_t* p, t v, size_t vl) {
        O::idx_store(p, v, vl);
    }
};

// One block of the bucket reduction, following the reference chain in
// simdlib256-inl.h (cmplt_min_max_fast + simd8float32_max):
//   fm = candidate < current             -> min values/indices (tie -> current)
//   fx = current   < fresh input         -> max values (tie -> current)
//   maxIndex = fm ? current : candidate  (tie -> candidate)
//
// Values are completed depth by depth before any index is written, and the
// max value of depth p feeds depth p+1 immediately, so at most six m4
// register groups are live at once in the N == 2 m4 plans. For N == 3 the
// index updates are deferred to the end, once the nmax values are dead.
// These merges are independent, so values, masks and indices are unchanged.

// Exact specialization of heap_replace_top<CMax<float, int>> for k == 8:
//   0 -> {1,2} -> {3,4 | 5,6} -> {7 for node 3 only}
// Comparisons and write-back order match the runtime-k heap_replace_top
// exactly.
inline void heap_replace_top_k8(
        float* bh_val,
        int32_t* bh_ids,
        float val,
        int32_t id) {
    using C = CMax<float, int>;
    uint32_t i;
    if (C::cmp2(bh_val[1], bh_val[2], bh_ids[1], bh_ids[2])) {
        if (C::cmp2(val, bh_val[1], id, bh_ids[1])) {
            bh_val[0] = val;
            bh_ids[0] = id;
            return;
        }
        bh_val[0] = bh_val[1];
        bh_ids[0] = bh_ids[1];
        i = 1;
    } else {
        if (C::cmp2(val, bh_val[2], id, bh_ids[2])) {
            bh_val[0] = val;
            bh_ids[0] = id;
            return;
        }
        bh_val[0] = bh_val[2];
        bh_ids[0] = bh_ids[2];
        i = 2;
    }
    {
        const uint32_t i1 = 2 * i + 1;
        const uint32_t i2 = i1 + 1;
        if (C::cmp2(bh_val[i1], bh_val[i2], bh_ids[i1], bh_ids[i2])) {
            if (C::cmp2(val, bh_val[i1], id, bh_ids[i1])) {
                bh_val[i] = val;
                bh_ids[i] = id;
                return;
            }
            bh_val[i] = bh_val[i1];
            bh_ids[i] = bh_ids[i1];
            i = i1;
        } else {
            if (C::cmp2(val, bh_val[i2], id, bh_ids[i2])) {
                bh_val[i] = val;
                bh_ids[i] = id;
                return;
            }
            bh_val[i] = bh_val[i2];
            bh_ids[i] = bh_ids[i2];
            i = i2;
        }
    }
    if (i == 3) {
        if (C::cmp2(val, bh_val[7], id, bh_ids[7])) {
            bh_val[3] = val;
            bh_ids[3] = id;
            return;
        }
        bh_val[3] = bh_val[7];
        bh_ids[3] = bh_ids[7];
        i = 7;
    }
    bh_val[i] = val;
    bh_ids[i] = id;
}

// Chunk / LMUL policy, selected once per call from the hardware VLEN.
//
// Narrow (VLEN=128, SEW=32): m1 = 4 lanes, m2 = 8, m4 = 16, m8 = 32.
// B32 with a single m8 group would need 4 state groups (32 vregs) plus the
// working set and spill badly, so it is processed as two m4 chunks. Buckets
// are independent, so chunking only changes the loop nesting, never results.
//
// Wide (VLEN>=256): the same bucket counts fit in narrower LMULs, which
// frees registers, and B32 fits one m4 chunk instead of two. The same
// kernel adapts to both boards.
template <uint32_t NBUCKETS, bool Wide>
struct rvv_plan;

template <>
struct rvv_plan<8, false> {
    static constexpr uint32_t kChunk = 8;
    using ops = rvv_e32m2;
};

template <>
struct rvv_plan<8, true> {
    static constexpr uint32_t kChunk = 8;
    using ops = rvv_e32m1;
};

template <>
struct rvv_plan<16, false> {
    static constexpr uint32_t kChunk = 16;
    using ops = rvv_e32m4;
};

template <>
struct rvv_plan<16, true> {
    static constexpr uint32_t kChunk = 16;
    using ops = rvv_e32m2;
};

template <>
struct rvv_plan<32, false> {
    static constexpr uint32_t kChunk = 16;
    using ops = rvv_e32m4;
};

template <>
struct rvv_plan<32, true> {
    static constexpr uint32_t kChunk = 32;
    using ops = rvv_e32m4;
};

// Reduce one block of P::kChunk lanes into the N per-depth bucket states.
// State is carried in v0..v2 / i0..i2; the candidate value/index of depth 0
// is (fresh, base).
template <uint32_t N, typename O, typename I>
inline __attribute__((always_inline)) void scan_block(
        const typename O::vf fresh,
        const typename I::t base,
        typename O::vf& v0,
        typename I::t& i0,
        typename O::vf& v1,
        typename I::t& i1,
        typename O::vf& v2,
        typename I::t& i2,
        const size_t vl) {
    using vf = typename O::vf;
    using vb = typename O::vb;
    using it = typename I::t;

    if constexpr (N == 1) {
        const vb fm = O::lt(fresh, v0, vl);
        v0 = O::select_f(fm, fresh, v0, vl);
        i0 = I::select(fm, base, i0, vl);
    } else if constexpr (N == 2) {
        // Depth 0: candidate value/index is (fresh, base). The fx0 merge
        // comes first (it needs the old v0), then the three fm0 consumers,
        // then depth 1. Keeping each mask's consumers contiguous minimizes
        // register moves.
        const vb fm0 = O::lt(fresh, v0, vl);
        const vb fx0 = O::lt(v0, fresh, vl);
        const vf nmax0 = O::select_f(fx0, fresh, v0, vl);
        const it nmaxi0 = I::select(fm0, i0, base, vl);
        i0 = I::select(fm0, base, i0, vl);
        v0 = O::select_f(fm0, fresh, v0, vl);

        // Depth 1 (last): the max side is dead; vfmin keeps v1 on a tie and
        // the fm1 mask is still needed for the index merge.
        const vb fm1 = O::lt(nmax0, v1, vl);
        v1 = O::vmin(v1, nmax0, vl);
        i1 = I::select(fm1, nmaxi0, i1, vl);
    } else {
        static_assert(N == 3, "N outside 1..3");
        // Depth 0: candidate value/index is (fresh, base).
        const vb fm0 = O::lt(fresh, v0, vl);
        const vb fx0 = O::lt(v0, fresh, vl);
        const vf nmax0 = O::select_f(fx0, fresh, v0, vl);
        v0 = O::select_f(fm0, fresh, v0, vl);

        // Depth 1: candidate value is nmax0; the max side compares the
        // original fresh input, as the reference chain does.
        const vb fm1 = O::lt(nmax0, v1, vl);
        const vb fx1 = O::lt(v1, fresh, vl);
        const vf nmax1 = O::select_f(fx1, fresh, v1, vl);
        v1 = O::select_f(fm1, nmax0, v1, vl);

        // Depth 2 (last): the max side is dead, so vfmin is only needed
        // for the value.
        const vb fm2 = O::lt(nmax1, v2, vl);
        v2 = O::vmin(v2, nmax1, vl);

        // Index side, deferred until the nmax values have died.
        const it nmaxi0 = I::select(fm0, i0, base, vl);
        const it nmaxi1 = I::select(fm1, i1, nmaxi0, vl);
        i0 = I::select(fm0, base, i0, vl);
        i1 = I::select(fm1, nmaxi0, i1, vl);
        i2 = I::select(fm2, nmaxi1, i2, vl);
    }
}

// Scan one chunk of buckets over all full NBUCKETS-sized blocks.
//
// The per-lane invariant of the reference chain is preserved: bucket
// (chunk_off + lane) sees the same sequence of values in the same order as
// the 8-lane grouping, and the stored buffers are read back in
// (depth, bucket) order during the merge.
//
// The block loop is software-pipelined: the load of block ip+1 is issued
// before block ip is reduced, so the L1 load-to-use latency overlaps the
// (independent) vector ALU work. The last block is peeled out of the loop
// to keep the prefetch in bounds.
template <uint32_t NBUCKETS, uint32_t N, typename O, bool Idx16>
inline void scan_chunk(
        const float* const __restrict src,
        const uint32_t n_per_beam,
        const uint32_t chunk_off,
        const uint32_t beam_off,
        float (&min_v)[N][NBUCKETS],
        uint32_t (&min_i)[N][NBUCKETS],
        const size_t vl) {
    using vf = typename O::vf;
    using I = rvv_index_ops<O, Idx16>;
    using it = typename I::t;

    const it idx0 = I::narrow(O::add_x(O::vid(vl), chunk_off, vl), vl);

    // RVV vector types cannot be array elements, so the N depths are named
    // locals; unused depths are removed by if constexpr for N = 1/2.
    [[maybe_unused]] vf v0, v1, v2;
    [[maybe_unused]] it i0, i1, i2;
    v0 = O::splat_f(std::numeric_limits<float>::max(), vl);
    i0 = idx0;
    if constexpr (N >= 2) {
        v1 = v0;
        i1 = idx0;
    }
    if constexpr (N >= 3) {
        v2 = v0;
        i2 = idx0;
    }

    const uint32_t nb = (n_per_beam / NBUCKETS) * NBUCKETS;
    const float* const bp = src + chunk_off;
    it base = idx0;

    if (nb != 0) {
        vf fresh = O::load_f(bp, vl);
        for (uint32_t ip = 0; ip + NBUCKETS < nb; ip += NBUCKETS) {
            const vf next = O::load_f(bp + ip + NBUCKETS, vl);
            scan_block<N, O, I>(fresh, base, v0, i0, v1, i1, v2, i2, vl);
            // Candidate index for the next block is ip + NBUCKETS + lane.
            base = I::add(base, NBUCKETS, vl);
            fresh = next;
        }
        scan_block<N, O, I>(fresh, base, v0, i0, v1, i1, v2, i2, vl);
    }

    // Lane ids already contain the bucket offset inside the NBUCKETS block,
    // so only the beam base remains to be added.
    O::store_f(&min_v[0][chunk_off], v0, vl);
    I::store(&min_i[0][chunk_off], I::add(i0, beam_off, vl), vl);
    if constexpr (N >= 2) {
        O::store_f(&min_v[1][chunk_off], v1, vl);
        I::store(&min_i[1][chunk_off], I::add(i1, beam_off, vl), vl);
    }
    if constexpr (N >= 3) {
        O::store_f(&min_v[2][chunk_off], v2, vl);
        I::store(&min_i[2][chunk_off], I::add(i2, beam_off, vl), vl);
    }
}

} // namespace

// RVV-tuned bucket scan: full specialization of the class template for
// SIMDLevel::RISCV_RVV. rvv.cpp is the only TU that instantiates
// <..., RISCV_RVV>; the explicit instantiations are at the end of this block.
template <uint32_t NBUCKETS, uint32_t N>
struct HeapWithBucketsCMaxFloat<NBUCKETS, N, SIMDLevel::RISCV_RVV> {
    static_assert(
            (NBUCKETS) > 0 && ((NBUCKETS % 8) == 0),
            "Number of buckets needs to be 8, 16, 24, ...");
    static_assert(N >= 1 && N <= 3, "RVV bucket scan supports N in 1..3");

    static void addn(
            const uint32_t n,
            const float* const __restrict distances,
            const uint32_t k,
            float* const __restrict bh_val,
            int32_t* const __restrict bh_ids) {
        bs_addn(1, n, distances, k, bh_val, bh_ids);
    }

    static void bs_addn(
            const uint32_t beam_size,
            const uint32_t n_per_beam,
            const float* const __restrict distances,
            const uint32_t k,
            float* const __restrict bh_val,
            int32_t* const __restrict bh_ids) {
        // Narrow boards have VLEN=128 (vlenb=16), wide ones VLEN>=256
        // (vlenb>=32); one branch per call. V guarantees VLEN >= 128, so
        // the narrow plan always has enough lanes for its chunks. Every
        // index is < beam_size * n_per_beam; when that fits in 16 bits the
        // scan keeps its index state at SEW=16 (half the register groups),
        // otherwise it falls back to u32.
        const bool idx16 = size_t(n_per_beam) * size_t(beam_size) <= 0xFFFFu;
        if (__riscv_vlenb() >= 32) {
            if (idx16) {
                bs_addn_impl<rvv_plan<NBUCKETS, true>, true>(
                        beam_size, n_per_beam, distances, k, bh_val, bh_ids);
            } else {
                bs_addn_impl<rvv_plan<NBUCKETS, true>, false>(
                        beam_size, n_per_beam, distances, k, bh_val, bh_ids);
            }
        } else {
            if (idx16) {
                bs_addn_impl<rvv_plan<NBUCKETS, false>, true>(
                        beam_size, n_per_beam, distances, k, bh_val, bh_ids);
            } else {
                bs_addn_impl<rvv_plan<NBUCKETS, false>, false>(
                        beam_size, n_per_beam, distances, k, bh_val, bh_ids);
            }
        }
    }

    template <typename P, bool Idx16>
    static void bs_addn_impl(
            const uint32_t beam_size,
            const uint32_t n_per_beam,
            const float* const __restrict distances,
            const uint32_t k,
            float* const __restrict bh_val,
            int32_t* const __restrict bh_ids) {
        using C = CMax<float, int>;
        using O = typename P::ops;

        static_assert(NBUCKETS % P::kChunk == 0, "chunk must divide NBUCKETS");

        // AVL is a compile-time constant; V guarantees VLEN >= 128, so this
        // yields exactly CHUNK lanes for the chosen LMUL.
        const size_t vl = O::setvl(P::kChunk);
        const uint32_t nb = (n_per_beam / NBUCKETS) * NBUCKETS;

        for (uint32_t beam_index = 0; beam_index < beam_size; beam_index++) {
            alignas(32) float min_values[N][NBUCKETS];
            alignas(32) uint32_t min_indices[N][NBUCKETS];

            const float* const src_base =
                    distances + size_t(n_per_beam) * beam_index;
            const uint32_t beam_off = n_per_beam * beam_index;

            // Buckets are independent; chunks only reorder the nesting.
            for (uint32_t chunk_off = 0; chunk_off < NBUCKETS;
                 chunk_off += P::kChunk) {
                scan_chunk<NBUCKETS, N, O, Idx16>(
                        src_base,
                        n_per_beam,
                        chunk_off,
                        beam_off,
                        min_values,
                        min_indices,
                        vl);
            }

            // merge every bucket into the regular heap.
            // k==1 fast path: heap_replace_top on a 1-element heap is a
            // plain assignment, so the merge is a pure min-scan.
            if (k == 1) {
                float top_val = bh_val[0];
                int32_t top_id = bh_ids[0];
                for (uint32_t p = 0; p < N; p++) {
                    for (uint32_t b = 0; b < NBUCKETS; b++) {
                        const auto value = min_values[p][b];
                        const auto index =
                                static_cast<int32_t>(min_indices[p][b]);
                        if (C::cmp2(top_val, value, top_id, index)) {
                            top_val = value;
                            top_id = index;
                        }
                    }
                }
                // process leftovers
                for (uint32_t ip = nb; ip < n_per_beam; ip++) {
                    const int32_t index = ip + n_per_beam * beam_index;
                    const float value = distances[index];

                    if (C::cmp(top_val, value)) {
                        top_val = value;
                        top_id = index;
                    }
                }
                bh_val[0] = top_val;
                bh_ids[0] = top_id;
            } else if (k == 8) {
                // Branch hoisted out of the candidate loop so the loop shape
                // stays exactly the same as in the k-generic branch below.
                for (uint32_t p = 0; p < N; p++) {
                    for (uint32_t b = 0; b < NBUCKETS; b++) {
                        const auto value = min_values[p][b];
                        const auto index =
                                static_cast<int32_t>(min_indices[p][b]);
                        if (C::cmp2(bh_val[0], value, bh_ids[0], index)) {
                            heap_replace_top_k8(bh_val, bh_ids, value, index);
                        }
                    }
                }
                // process leftovers
                for (uint32_t ip = nb; ip < n_per_beam; ip++) {
                    const int32_t index = ip + n_per_beam * beam_index;
                    const float value = distances[index];

                    if (C::cmp(bh_val[0], value)) {
                        heap_replace_top_k8(bh_val, bh_ids, value, index);
                    }
                }
            } else {
                for (uint32_t p = 0; p < N; p++) {
                    for (uint32_t b = 0; b < NBUCKETS; b++) {
                        const auto value = min_values[p][b];
                        const auto index =
                                static_cast<int32_t>(min_indices[p][b]);
                        if (C::cmp2(bh_val[0], value, bh_ids[0], index)) {
                            heap_replace_top<C>(
                                    k, bh_val, bh_ids, value, index);
                        }
                    }
                }
                // process leftovers
                for (uint32_t ip = nb; ip < n_per_beam; ip++) {
                    const int32_t index = ip + n_per_beam * beam_index;
                    const float value = distances[index];

                    if (C::cmp(bh_val[0], value)) {
                        heap_replace_top<C>(k, bh_val, bh_ids, value, index);
                    }
                }
            }
        }
    }
};

template struct HeapWithBucketsCMaxFloat<8, 3, SIMDLevel::RISCV_RVV>;
template struct HeapWithBucketsCMaxFloat<8, 2, SIMDLevel::RISCV_RVV>;
template struct HeapWithBucketsCMaxFloat<16, 2, SIMDLevel::RISCV_RVV>;
template struct HeapWithBucketsCMaxFloat<16, 1, SIMDLevel::RISCV_RVV>;
template struct HeapWithBucketsCMaxFloat<32, 2, SIMDLevel::RISCV_RVV>;

// ---------------------------------------------------------------------------
// Native RVV accum_and_*_tab kernels.
//
// K is strip-mined with a runtime vsetvl_e32m8, so these kernels are
// independent of the NK template parameter and have no scalar tail. Row
// additions keep the reference order (cbs[0] + cbs[1] + ...), so results
// are bit-identical to the NONE body in rq_beam_search_tab-inl.h.
// ---------------------------------------------------------------------------

namespace {

// Row pointers for the M codebooks of this call.
// - store/add: codes are indexed at b * m_offset (m_offset is the caller's
//   full chunk stride, a runtime value).
// - finalize: codes are indexed at b * M.
template <size_t M>
inline void accum_row_ptrs(
        const float* const __restrict codebook_cross_norms,
        const uint64_t* const __restrict codebook_offsets,
        const int32_t* const __restrict codes_i,
        // b * m_offset for store/add, b * M for finalize.
        const size_t code_base,
        const size_t ldc,
        const float* (&cbs)[M]) {
    for (size_t ij = 0; ij < M; ij++) {
        const size_t code = static_cast<size_t>(codes_i[code_base + ij]);
        cbs[ij] = &codebook_cross_norms[(codebook_offsets[ij] + code) * ldc];
    }
}

// sum_{ij=0}^{M-1} cbs[ij][kk .. kk+vl)
template <size_t M>
inline __attribute__((always_inline)) vfloat32m8_t
accum_sum_rows(const float* const (&cbs)[M], const size_t kk, const size_t vl) {
    vfloat32m8_t r = __riscv_vle32_v_f32m8(cbs[0] + kk, vl);
    for (size_t ij = 1; ij < M; ij++) {
        r = __riscv_vfadd_vv_f32m8(
                r, __riscv_vle32_v_f32m8(cbs[ij] + kk, vl), vl);
    }
    return r;
}

// Cursor form of accum_sum_rows: each row pointer advances by vl, so the
// block loop carries no "base + kk" address arithmetic. Summation order is
// unchanged.
template <size_t M>
inline __attribute__((always_inline)) vfloat32m8_t
accum_sum_rows_adv(const float* (&p)[M], const size_t vl) {
    vfloat32m8_t r = __riscv_vle32_v_f32m8(p[0], vl);
    p[0] += vl;
    for (size_t ij = 1; ij < M; ij++) {
        r = __riscv_vfadd_vv_f32m8(r, __riscv_vle32_v_f32m8(p[ij], vl), vl);
        p[ij] += vl;
    }
    return r;
}

} // namespace

template <size_t M>
void accum_tab_rvv_store(
        const size_t m_offset,
        const float* const __restrict codebook_cross_norms,
        const uint64_t* const __restrict codebook_offsets,
        const int32_t* const __restrict codes_i,
        const size_t b,
        const size_t ldc,
        const size_t K,
        float* const __restrict output) {
    const float* cbs[M];
    accum_row_ptrs<M>(
            codebook_cross_norms,
            codebook_offsets,
            codes_i,
            b * m_offset,
            ldc,
            cbs);

    // The main loop runs at the vl returned once per call (VLMAX, or K
    // itself when K < VLMAX); only a partial tail block re-programs vl.
    // With 32 lanes, K = 1024 gives 32 blocks and no in-loop vsetvli.
    const size_t vlmax = __riscv_vsetvl_e32m8(K);
    const float* p[M];
    for (size_t ij = 0; ij < M; ij++) {
        p[ij] = cbs[ij];
    }
    float* po = output;
    size_t kk = 0;
    for (; kk + vlmax <= K; kk += vlmax) {
        const vfloat32m8_t r = accum_sum_rows_adv<M>(p, vlmax);
        __riscv_vse32_v_f32m8(po, r, vlmax);
        po += vlmax;
    }
    if (kk < K) {
        const size_t vl = __riscv_vsetvl_e32m8(K - kk);
        const vfloat32m8_t r = accum_sum_rows<M>(cbs, kk, vl);
        __riscv_vse32_v_f32m8(output + kk, r, vl);
    }
}

template <size_t M>
void accum_tab_rvv_add(
        const size_t m_offset,
        const float* const __restrict codebook_cross_norms,
        const uint64_t* const __restrict codebook_offsets,
        const int32_t* const __restrict codes_i,
        const size_t b,
        const size_t ldc,
        const size_t K,
        float* const __restrict output) {
    const float* cbs[M];
    accum_row_ptrs<M>(
            codebook_cross_norms,
            codebook_offsets,
            codes_i,
            b * m_offset,
            ldc,
            cbs);

    // Fixed-vl main loop plus a single re-programmed tail block; row and
    // output cursors replace "base + kk" address arithmetic.
    const size_t vlmax = __riscv_vsetvl_e32m8(K);
    const float* p[M];
    for (size_t ij = 0; ij < M; ij++) {
        p[ij] = cbs[ij];
    }
    float* po = output;
    size_t kk = 0;
    for (; kk + vlmax <= K; kk += vlmax) {
        const vfloat32m8_t sum = accum_sum_rows_adv<M>(p, vlmax);
        // Reference order: output += sum.
        const vfloat32m8_t r = __riscv_vfadd_vv_f32m8(
                __riscv_vle32_v_f32m8(po, vlmax), sum, vlmax);
        __riscv_vse32_v_f32m8(po, r, vlmax);
        po += vlmax;
    }
    if (kk < K) {
        const size_t vl = __riscv_vsetvl_e32m8(K - kk);
        const vfloat32m8_t sum = accum_sum_rows<M>(cbs, kk, vl);
        const vfloat32m8_t r = __riscv_vfadd_vv_f32m8(
                __riscv_vle32_v_f32m8(output + kk, vl), sum, vl);
        __riscv_vse32_v_f32m8(output + kk, r, vl);
    }
}

template <size_t M>
void accum_tab_rvv_finalize(
        const float* const __restrict codebook_cross_norms,
        const uint64_t* const __restrict codebook_offsets,
        const int32_t* const __restrict codes_i,
        const size_t b,
        const size_t ldc,
        const size_t K,
        const float* const __restrict distances_i,
        const float* const __restrict cd_common,
        float* const __restrict output) {
    const float* cbs[M];
    accum_row_ptrs<M>(
            codebook_cross_norms, codebook_offsets, codes_i, b * M, ldc, cbs);

    // output[b][k] = (2 * sum[k] + cd_common[k]) + distances_i[b].
    // 2 * sum is exact, so the fused multiply-add matches the reference
    // fmadd(two, regs, common) bit for bit.
    const float dist_b = distances_i[b];
    float* const __restrict out_row = output + b * K;

    // Fixed-vl main loop plus a single re-programmed tail block; row,
    // cd_common and output cursors replace "base + kk" arithmetic.
    const size_t vlmax = __riscv_vsetvl_e32m8(K);
    const float* p[M];
    for (size_t ij = 0; ij < M; ij++) {
        p[ij] = cbs[ij];
    }
    const float* pc = cd_common;
    float* po = out_row;
    size_t kk = 0;
    for (; kk + vlmax <= K; kk += vlmax) {
        // Issue the cd_common load before the row chain so its latency
        // overlaps the M row loads and (M - 1)-add reduction. The rounding
        // order (2*sum + c) then + dist is unchanged.
        const vfloat32m8_t ccur = __riscv_vle32_v_f32m8(pc, vlmax);
        const vfloat32m8_t sum = accum_sum_rows_adv<M>(p, vlmax);
        vfloat32m8_t r = __riscv_vfmacc_vf_f32m8(ccur, 2.0f, sum, vlmax);
        r = __riscv_vfadd_vf_f32m8(r, dist_b, vlmax);
        __riscv_vse32_v_f32m8(po, r, vlmax);
        pc += vlmax;
        po += vlmax;
    }
    if (kk < K) {
        const size_t vl = __riscv_vsetvl_e32m8(K - kk);
        const vfloat32m8_t sum = accum_sum_rows<M>(cbs, kk, vl);
        vfloat32m8_t r = __riscv_vfmacc_vf_f32m8(
                __riscv_vle32_v_f32m8(cd_common + kk, vl), 2.0f, sum, vl);
        r = __riscv_vfadd_vf_f32m8(r, dist_b, vl);
        __riscv_vse32_v_f32m8(out_row + kk, r, vl);
    }
}

#define INSTANTIATE_ACCUM_TAB(M)                                      \
    template void accum_and_store_tab<M, 4, SIMDLevel::RISCV_RVV>(    \
            size_t,                                                   \
            const float* __restrict,                                  \
            const uint64_t* __restrict,                               \
            const int32_t* __restrict,                                \
            size_t,                                                   \
            size_t,                                                   \
            size_t,                                                   \
            float* __restrict);                                       \
    template void accum_and_add_tab<M, 4, SIMDLevel::RISCV_RVV>(      \
            size_t,                                                   \
            const float* __restrict,                                  \
            const uint64_t* __restrict,                               \
            const int32_t* __restrict,                                \
            size_t,                                                   \
            size_t,                                                   \
            size_t,                                                   \
            float* __restrict);                                       \
    template void accum_and_finalize_tab<M, 4, SIMDLevel::RISCV_RVV>( \
            const float* __restrict,                                  \
            const uint64_t* __restrict,                               \
            const int32_t* __restrict,                                \
            size_t,                                                   \
            size_t,                                                   \
            size_t,                                                   \
            const float* __restrict,                                  \
            const float* __restrict,                                  \
            float* __restrict);

INSTANTIATE_ACCUM_TAB(1)
INSTANTIATE_ACCUM_TAB(2)
INSTANTIATE_ACCUM_TAB(3)
INSTANTIATE_ACCUM_TAB(4)
INSTANTIATE_ACCUM_TAB(5)
INSTANTIATE_ACCUM_TAB(6)
INSTANTIATE_ACCUM_TAB(7)
INSTANTIATE_ACCUM_TAB(8)

#undef INSTANTIATE_ACCUM_TAB

// The NK == 4 wrappers above are the only references to the native kernels
// in this TU, while callers may instantiate accum_and_*_tab<M, NK != 4, RVV>
// themselves. Emit the native kernels explicitly so out-of-line definitions
// exist regardless of whether the compiler inlines them into the wrappers.
#define INSTANTIATE_ACCUM_RVV(M)             \
    template void accum_tab_rvv_store<M>(    \
            size_t,                          \
            const float* __restrict,         \
            const uint64_t* __restrict,      \
            const int32_t* __restrict,       \
            size_t,                          \
            size_t,                          \
            size_t,                          \
            float* __restrict);              \
    template void accum_tab_rvv_add<M>(      \
            size_t,                          \
            const float* __restrict,         \
            const uint64_t* __restrict,      \
            const int32_t* __restrict,       \
            size_t,                          \
            size_t,                          \
            size_t,                          \
            float* __restrict);              \
    template void accum_tab_rvv_finalize<M>( \
            const float* __restrict,         \
            const uint64_t* __restrict,      \
            const int32_t* __restrict,       \
            size_t,                          \
            size_t,                          \
            size_t,                          \
            const float* __restrict,         \
            const float* __restrict,         \
            float* __restrict);

INSTANTIATE_ACCUM_RVV(1)
INSTANTIATE_ACCUM_RVV(2)
INSTANTIATE_ACCUM_RVV(3)
INSTANTIATE_ACCUM_RVV(4)
INSTANTIATE_ACCUM_RVV(5)
INSTANTIATE_ACCUM_RVV(6)
INSTANTIATE_ACCUM_RVV(7)
INSTANTIATE_ACCUM_RVV(8)

#undef INSTANTIATE_ACCUM_RVV

} // namespace faiss

#endif // COMPILE_SIMD_RISCV_RVV
