/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// RISC-V Vector version of the fused L2 + top-1 kernel, see distances_fused.h.
//
// Layout follows avx512.cpp: y is transposed once so that the values of one
// dimension of consecutive database vectors are contiguous, and each loop
// iteration computes the dot products of a few queries with a whole block of
// database vectors at once.
//
// Differences from the AVX-512 kernel:
//   - The -2 * x[d] coefficients are broadcast with vfmacc.vf, which reads them
//     from scalar registers. AVX-512 keeps one vector register per
//     (query, dimension) pair, RVV does not, so the saved registers go to
//     more database vectors per iteration instead.
//   - The running minimum is kept per lane (16 lanes for e32m4 at VLEN=128)
//     together with the j that reached it, and the final scan over the lanes
//     applies the same tie-breaking rule as the x86 kernels: strictly
//     smaller distance wins, on equality the smaller index wins.
//   - The main loop runs with one vsetvli configuration; only the trailing
//     block reconfigures vl.
//
// GCC rejects RVV types in arrays and struct members, so the per-query state
// is a fixed set of named variables rather than an array indexed by query.

#include <faiss/impl/platform_macros.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/distances_fused/distances_fused.h>

#include <riscv_vector.h>

namespace faiss {

namespace {

// It makes sense to overload certain cases because the further
// kernels are in need of registers. So, let's tell compiler
// not to waste registers on a bit faster code, if needed.
template <size_t DIM>
float l2_sqr(const float* const x) {
    float output = x[0] * x[0];
    for (size_t i = 1; i < DIM; i++) {
        output += x[i] * x[i];
    }

    return output;
}

template <size_t DIM>
float dot_product(
        const float* const __restrict x,
        const float* const __restrict y) {
    float output = x[0] * y[0];
    for (size_t i = 1; i < DIM; i++) {
        output += x[i] * y[i];
    }

    return output;
}

// Largest possible number of e32 lanes per group for one vsetvlmax (VLEN up
// to 2048 bits with LMUL=4). Only used to size the stack scratch below.
constexpr size_t MAX_LANES = 64;

// Turn the per-lane minima of one query into a single result: scan the lanes,
// add |x|^2 back, clamp negative roundoff to 0, apply the tie-breaking rule of
// the original implementation (equal distance -> smaller index), then run the
// scalar tail [j, ny).
template <size_t DIM>
inline void dump_and_add(
        const vfloat32m4_t min_d,
        const vint32m4_t min_j,
        const size_t lanes_seen,
        const size_t vlmax,
        const size_t j,
        const float x_norm,
        const float* const __restrict x_row,
        const float* const __restrict y,
        const size_t ny,
        const float* __restrict y_norms,
        Top1BlockResultHandler<CMax<float, int64_t>>& res,
        const size_t i) {
    float min_distances_scalar[MAX_LANES];
    int32_t min_indices_scalar[MAX_LANES];
    __riscv_vse32_v_f32m4(min_distances_scalar, min_d, vlmax);
    __riscv_vse32_v_i32m4(min_indices_scalar, min_j, vlmax);

    float current_min_distance = res.dis_tab[i];
    int64_t current_min_index = res.ids_tab[i];

    // This unusual comparison is needed to maintain the behavior
    // of the original implementation: if two indices are
    // represented with equal distance values, then
    // the index with the min value is returned.
    for (size_t jv = 0; jv < lanes_seen; jv++) {
        if (min_indices_scalar[jv] < 0) {
            continue; // lane never saw a database vector
        }
        // add missing x_norms[i]
        float distance_candidate = min_distances_scalar[jv] + x_norm;

        // negative values can occur for identical vectors
        //    due to roundoff errors.
        if (distance_candidate < 0) {
            distance_candidate = 0;
        }

        const int64_t index_candidate = min_indices_scalar[jv];

        if (current_min_distance > distance_candidate) {
            current_min_distance = distance_candidate;
            current_min_index = index_candidate;
        } else if (
                current_min_distance == distance_candidate &&
                (current_min_index < 0 ||
                 current_min_index > index_candidate)) {
            current_min_index = index_candidate;
        }
    }

    // process leftovers
    for (size_t j0 = j; j0 < ny; j0++) {
        const float dp = dot_product<DIM>(x_row, y + j0 * DIM);
        float dis = x_norm + y_norms[j0] - 2 * dp;
        // negative values can occur for identical vectors
        //    due to roundoff errors.
        if (dis < 0) {
            dis = 0;
        }

        if (current_min_distance > dis) {
            current_min_distance = dis;
            current_min_index = j0;
        }
    }

    // done
    res.add_result(i, current_min_distance, current_min_index);
}

// The kernel for low dimensionality vectors.
// Finds the closest one from y for every given NX_POINTS_PER_LOOP points from x
//
// DIM is the dimensionality of the data
// NX_POINTS_PER_LOOP is the number of x points that get processed
//   simultaneously.
template <size_t DIM, size_t NX_POINTS_PER_LOOP>
void kernel(
        const float* const __restrict x,
        const float* const __restrict y,
        const float* const __restrict y_transposed,
        const size_t ny,
        const size_t y_transposed_stride,
        Top1BlockResultHandler<CMax<float, int64_t>>& res,
        const float* __restrict y_norms,
        const size_t i) {
    static_assert(NX_POINTS_PER_LOOP == 1 || NX_POINTS_PER_LOOP == 2);

    const float* const __restrict xd_0 = x + i * DIM;

    // load -2 * value, and compute x_norm
    float x_i[NX_POINTS_PER_LOOP][DIM];
    float x_norm_i[NX_POINTS_PER_LOOP];
    for (size_t nx_k = 0; nx_k < NX_POINTS_PER_LOOP; nx_k++) {
        for (size_t dd = 0; dd < DIM; dd++) {
            x_i[nx_k][dd] = -2 * *(xd_0 + nx_k * DIM + dd);
        }
        x_norm_i[nx_k] = l2_sqr<DIM>(xd_0 + nx_k * DIM);
    }

    const size_t vlmax = __riscv_vsetvlmax_e32m4();
    // y^2 - 2<x,y> running minimum per lane, and the j that reached it
    const vfloat32m4_t inf = __riscv_vfmv_v_f_f32m4(__builtin_inff(), vlmax);
    vfloat32m4_t min_d0 = inf;
    vint32m4_t min_j0 = __riscv_vmv_v_x_i32m4(-1, vlmax);
    vfloat32m4_t min_d1 = inf;
    vint32m4_t min_j1 = __riscv_vmv_v_x_i32m4(-1, vlmax);

    size_t lanes_seen = 0; // lanes touched by at least one iteration

    size_t j = 0;
    auto body = [&](const size_t vl) {
        vfloat32m4_t dp0 = __riscv_vfmv_v_f_f32m4(0.0f, vl);
        vfloat32m4_t dp1 = dp0;

        // technically, we're multiplying -2x and y
        for (size_t dd = 0; dd < DIM; dd++) {
            const vfloat32m4_t y_i = __riscv_vle32_v_f32m4(
                    y_transposed + j + y_transposed_stride * dd, vl);
            dp0 = __riscv_vfmacc_vf_f32m4(dp0, x_i[0][dd], y_i, vl);
            if constexpr (NX_POINTS_PER_LOOP == 2) {
                dp1 = __riscv_vfmacc_vf_f32m4(dp1, x_i[1][dd], y_i, vl);
            }
        }

        // compute y^2 + (-2x,y)
        const vfloat32m4_t y_l2_sqr = __riscv_vle32_v_f32m4(y_norms + j, vl);
        // the j of each lane, for the index bookkeeping
        const vint32m4_t j_lane = __riscv_vadd_vx_i32m4(
                __riscv_vreinterpret_v_u32m4_i32m4(__riscv_vid_v_u32m4(vl)),
                int32_t(j),
                vl);

        // do the comparisons and alter the min values and indices
        const vfloat32m4_t dis0 = __riscv_vfadd_vv_f32m4(dp0, y_l2_sqr, vl);
        const vbool8_t c0 = __riscv_vmflt_vv_f32m4_b8(dis0, min_d0, vl);
        min_d0 = __riscv_vmerge_vvm_f32m4(min_d0, dis0, c0, vl);
        min_j0 = __riscv_vmerge_vvm_i32m4(min_j0, j_lane, c0, vl);

        if constexpr (NX_POINTS_PER_LOOP == 2) {
            const vfloat32m4_t dis1 = __riscv_vfadd_vv_f32m4(dp1, y_l2_sqr, vl);
            const vbool8_t c1 = __riscv_vmflt_vv_f32m4_b8(dis1, min_d1, vl);
            min_d1 = __riscv_vmerge_vvm_f32m4(min_d1, dis1, c1, vl);
            min_j1 = __riscv_vmerge_vvm_i32m4(min_j1, j_lane, c1, vl);
        }

        if (vl > lanes_seen) {
            lanes_seen = vl;
        }
    };

    // The main loop runs only with vl = vlmax and leaves the remaining
    // ny - j < vlmax vectors to the scalar tail. A short-vector iteration
    // would be the natural alternative, but GCC 15 at -O3 miscompiles it (it
    // returns wrong distances for some queries when ny is not a multiple of
    // vlmax), and a tail of at most vlmax vectors is negligible when the
    // database is in the thousands.
    for (; j + vlmax <= ny; j += vlmax) {
        body(vlmax);
    }

    // dump values and find the minimum distance / minimum index
    dump_and_add<DIM>(
            min_d0,
            min_j0,
            lanes_seen,
            vlmax,
            j,
            x_norm_i[0],
            x + i * DIM,
            y,
            ny,
            y_norms,
            res,
            i);

    if constexpr (NX_POINTS_PER_LOOP == 2) {
        dump_and_add<DIM>(
                min_d1,
                min_j1,
                lanes_seen,
                vlmax,
                j,
                x_norm_i[1],
                x + (i + 1) * DIM,
                y,
                ny,
                y_norms,
                res,
                i + 1);
    }
}

template <size_t DIM, size_t NX_POINTS_PER_LOOP>
void exhaustive_L2sqr_fused_cmax(
        const float* const __restrict x,
        const float* const __restrict y,
        size_t nx,
        size_t ny,
        Top1BlockResultHandler<CMax<float, int64_t>>& res,
        const float* __restrict y_norms) {
    // BLAS does not like empty matrices
    if (nx == 0 || ny == 0) {
        return;
    }

    // compute norms for y
    std::unique_ptr<float[]> del2;
    if (!y_norms) {
        float* y_norms2 = new float[ny];
        del2.reset(y_norms2);
        fvec_norms_L2sqr(y_norms2, y, DIM, ny);
        y_norms = y_norms2;
    }

    // initialize res
    res.begin_multiple(0, nx);

    // transpose y: the values of one dimension of consecutive database
    // vectors become contiguous, so they can be loaded with vle32.
    constexpr size_t ALIASING_STRIDE_BYTES = 4096;
    size_t y_transposed_stride = (ny + 15) / 16 * 16;
    if (y_transposed_stride * sizeof(float) % ALIASING_STRIDE_BYTES == 0) {
        y_transposed_stride += 16;
    }
    std::vector<float> y_transposed(DIM * y_transposed_stride);
    {
        const size_t src_stride = DIM * sizeof(float);
        for (size_t dd = 0; dd < DIM; dd++) {
            float* dst = y_transposed.data() + dd * y_transposed_stride;
            size_t jt = 0;
            while (jt < ny) {
                const size_t vl = __riscv_vsetvl_e32m4(ny - jt);
                const vfloat32m4_t v = __riscv_vlse32_v_f32m4(
                        y + dd + jt * DIM, src_stride, vl);
                __riscv_vse32_v_f32m4(dst + jt, v, vl);
                jt += vl;
            }
        }
    }

    const size_t nx_p = (nx / NX_POINTS_PER_LOOP) * NX_POINTS_PER_LOOP;
    // the main loop.
#pragma omp parallel for schedule(FAISS_OMP_DYNAMIC)
    for (int64_t i = 0; i < static_cast<int64_t>(nx_p);
         i += NX_POINTS_PER_LOOP) {
        kernel<DIM, NX_POINTS_PER_LOOP>(
                x,
                y,
                y_transposed.data(),
                ny,
                y_transposed_stride,
                res,
                y_norms,
                i);
    }

    for (size_t i = nx_p; i < nx; i++) {
        kernel<DIM, 1>(
                x,
                y,
                y_transposed.data(),
                ny,
                y_transposed_stride,
                res,
                y_norms,
                i);
    }

    // Does nothing for Top1BlockResultHandler, but
    // keeping the call for the consistency.
    res.end_multiple();
    InterruptCallback::check();
}

} // namespace

template <>
bool exhaustive_L2sqr_fused_cmax<SIMDLevel::RISCV_RVV>(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        Top1BlockResultHandler<CMax<float, int64_t>>& res,
        const float* y_norms) {
    // process only cases with certain dimensionalities

#define DISPATCH(DIM, NX_POINTS_PER_LOOP)                     \
    case DIM: {                                               \
        exhaustive_L2sqr_fused_cmax<DIM, NX_POINTS_PER_LOOP>( \
                x, y, nx, ny, res, y_norms);                  \
        return true;                                          \
    }

    switch (d) {
        DISPATCH(1, 2)
        DISPATCH(2, 2)
        DISPATCH(3, 2)
        DISPATCH(4, 2)
        DISPATCH(5, 2)
        DISPATCH(6, 2)
        DISPATCH(7, 2)
        DISPATCH(8, 2)
        DISPATCH(9, 2)
        DISPATCH(10, 2)
        DISPATCH(11, 2)
        DISPATCH(12, 2)
        DISPATCH(13, 2)
        DISPATCH(14, 2)
        DISPATCH(15, 2)
        DISPATCH(16, 2)
        DISPATCH(17, 2)
        DISPATCH(18, 2)
        DISPATCH(19, 2)
        DISPATCH(20, 2)
        DISPATCH(21, 2)
        DISPATCH(22, 2)
        DISPATCH(23, 2)
        DISPATCH(24, 2)
        DISPATCH(25, 2)
        DISPATCH(26, 2)
        DISPATCH(27, 2)
        DISPATCH(28, 2)
        DISPATCH(29, 2)
        DISPATCH(30, 2)
        DISPATCH(31, 2)
        DISPATCH(32, 2)
    }

    return false;
#undef DISPATCH
}

} // namespace faiss
