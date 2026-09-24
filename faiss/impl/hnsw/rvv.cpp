/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifdef COMPILE_SIMD_RISCV_RVV

#include <faiss/impl/hnsw/MinimaxHeap.h>

#include <riscv_vector.h>
#include <cassert>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace faiss {

namespace {

template <class HC>
int pop_best_rvv(MinimaxHeapT<HC>& heap, float* vmin_out) {
    using storage_idx_t = typename MinimaxHeapT<HC>::storage_idx_t;
    static_assert(
            std::is_same<storage_idx_t, int32_t>::value,
            "This code expects storage_idx_t to be int32_t");
    assert(heap.k > 0);
    constexpr float worst_v = HC::is_max
            ? std::numeric_limits<float>::infinity()
            : -std::numeric_limits<float>::infinity();
    const size_t vlmax = __riscv_vsetvlmax_e32m4();
    vfloat32m4_t best_dis = __riscv_vfmv_v_f_f32m4(worst_v, vlmax);
    vint32m4_t best_idx = __riscv_vmv_v_x_i32m4(-1, vlmax);

    const vuint32m4_t lanes_u = __riscv_vid_v_u32m4(vlmax);
    const vint32m4_t lanes_i = __riscv_vreinterpret_v_u32m4_i32m4(lanes_u);

    const int32_t* ids = heap.ids.data();
    const float* dis = heap.dis.data();
    const size_t k = size_t(heap.k);
    size_t i = 0;
    while (i < k) {
        const size_t vl = __riscv_vsetvl_e32m4(k - i);
        const vint32m4_t ids_v = __riscv_vle32_v_i32m4(ids + i, vl);
        const vfloat32m4_t dis_v = __riscv_vle32_v_f32m4(dis + i, vl);

        const vbool8_t empty = __riscv_vmslt_vx_i32m4_b8(ids_v, 0, vl);
        const vbool8_t keep = HC::is_max
                ? __riscv_vmflt_vv_f32m4_b8(best_dis, dis_v, vl)
                : __riscv_vmfgt_vv_f32m4_b8(best_dis, dis_v, vl);
        const vbool8_t upd = __riscv_vmnor_mm_b8(empty, keep, vl);

        const vbool8_t in_range = __riscv_vmsltu_vx_u32m4_b8(
                lanes_u, uint32_t(vl), vlmax);
        const vbool8_t upd_w = __riscv_vmand_mm_b8(upd, in_range, vlmax);

        best_dis = __riscv_vmerge_vvm_f32m4(best_dis, dis_v, upd_w, vlmax);
        best_idx = __riscv_vmerge_vvm_i32m4(
                best_idx,
                __riscv_vadd_vx_i32m4(lanes_i, int32_t(i), vlmax),
                upd_w,
                vlmax);

        i += vl;
    }

    const vfloat32m1_t seed = __riscv_vfmv_s_f_f32m1(worst_v, 1);
    const vfloat32m1_t red = HC::is_max
            ? __riscv_vfredmin_vs_f32m4_f32m1(best_dis, seed, vlmax)
            : __riscv_vfredmax_vs_f32m4_f32m1(best_dis, seed, vlmax);
    const float best_val = __riscv_vfmv_f_s_f32m1_f32(red);

    const vbool8_t is_min =
            __riscv_vmfeq_vf_f32m4_b8(best_dis, best_val, vlmax);
    const vbool8_t has_idx = __riscv_vmsge_vx_i32m4_b8(best_idx, 0, vlmax);
    const vbool8_t take = __riscv_vmand_mm_b8(is_min, has_idx, vlmax);

    if (__riscv_vfirst_m_b8(take, vlmax) < 0) {
        return -1;
    }

    const vuint32m4_t uidx = __riscv_vreinterpret_v_i32m4_u32m4(best_idx);
    const vuint32m1_t useed = __riscv_vmv_s_x_u32m1(0, 1);
    const vuint32m1_t ured = __riscv_vredmaxu_vs_u32m4_u32m1_m(
            take, uidx, useed, vlmax);
    const int best_i = int(__riscv_vmv_x_s_u32m1_u32(ured));

    if (vmin_out) {
        *vmin_out = best_val;
    }
    const int ret = heap.ids[best_i];
    heap.ids[best_i] = -1;
    --heap.nvalid;
    return ret;
}

} // namespace

template <>
int pop_min_tpl<CMax<float, int32_t>, SIMDLevel::RISCV_RVV>(
        MinimaxHeapT<CMax<float, int32_t>>* heap,
        float* vmin_out) {
    return pop_best_rvv<CMax<float, int32_t>>(*heap, vmin_out);
}

template <>
int pop_min_tpl<CMin<float, int32_t>, SIMDLevel::RISCV_RVV>(
        MinimaxHeapT<CMin<float, int32_t>>* heap,
        float* vmin_out) {
    return pop_best_rvv<CMin<float, int32_t>>(*heap, vmin_out);
}

} // namespace faiss

#endif // COMPILE_SIMD_RISCV_RVV
