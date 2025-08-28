/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <arm_sve.h>
#include <omp.h>

#include <faiss/utils/simd_impl/exhaustive_search_ops.h>

namespace faiss {

bool exhaustive_L2sqr_fused_cmax_simdlib<SIMDLevel::ARM_SVE>(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        Top1BlockResultHandler<CMax<float, int64_t>>& res,
        const float* y_norms = nullptr) {
    // BLAS does not like empty matrices
    if (nx == 0 || ny == 0)
        return false;

    /* block sizes */
    const size_t bs_x = distance_compute_blas_query_bs;
    const size_t bs_y = distance_compute_blas_database_bs;
    // const size_t bs_x = 16, bs_y = 16;
    std::unique_ptr<float[]> ip_block(new float[bs_x * bs_y]);
    std::unique_ptr<float[]> x_norms(new float[nx]);
    std::unique_ptr<float[]> del2;

    fvec_norms_L2sqr(x_norms.get(), x, d, nx);

    const size_t lanes = svcntw();

    if (!y_norms) {
        float* y_norms2 = new float[ny];
        del2.reset(y_norms2);
        fvec_norms_L2sqr(y_norms2, y, d, ny);
        y_norms = y_norms2;
    }

    for (size_t i0 = 0; i0 < nx; i0 += bs_x) {
        size_t i1 = i0 + bs_x;
        if (i1 > nx)
            i1 = nx;

        res.begin_multiple(i0, i1);

        for (size_t j0 = 0; j0 < ny; j0 += bs_y) {
            size_t j1 = j0 + bs_y;
            if (j1 > ny)
                j1 = ny;
            /* compute the actual dot products */
            {
                float one = 1, zero = 0;
                FINTEGER nyi = j1 - j0, nxi = i1 - i0, di = d;
                sgemm_("Transpose",
                       "Not transpose",
                       &nyi,
                       &nxi,
                       &di,
                       &one,
                       y + j0 * d,
                       &di,
                       x + i0 * d,
                       &di,
                       &zero,
                       ip_block.get(),
                       &nyi);
            }
#pragma omp parallel for
            for (int64_t i = i0; i < i1; i++) {
                const size_t count = j1 - j0;
                float* ip_line = ip_block.get() + (i - i0) * count;

                svprfw(svwhilelt_b32_u64(0, count), ip_line, SV_PLDL1KEEP);
                svprfw(svwhilelt_b32_u64(lanes, count),
                       ip_line + lanes,
                       SV_PLDL1KEEP);

                // Track lanes min distances + lanes min indices.
                // All the distances tracked do not take x_norms[i]
                //   into account in order to get rid of extra
                //   vaddq_f32(x_norms[i], ...) instructions
                //   is distance computations.
                auto min_distances = svdup_n_f32(res.dis_tab[i] - x_norms[i]);

                // these indices are local and are relative to j0.
                // so, value 0 means j0.
                auto min_indices = svdup_n_u32(0u);

                auto current_indices = svindex_u32(0u, 1u);

                // process lanes * 2 elements per loop
                for (size_t idx_j = 0; idx_j < count;
                     idx_j += lanes * 2, ip_line += lanes * 2) {
                    svprfw(svwhilelt_b32_u64(idx_j + lanes * 2, count),
                           ip_line + lanes * 2,
                           SV_PLDL1KEEP);
                    svprfw(svwhilelt_b32_u64(idx_j + lanes * 3, count),
                           ip_line + lanes * 3,
                           SV_PLDL1KEEP);

                    // mask
                    const auto mask_0 = svwhilelt_b32_u64(idx_j, count);
                    const auto mask_1 = svwhilelt_b32_u64(idx_j + lanes, count);

                    // load values for norms
                    const auto y_norm_0 =
                            svld1_f32(mask_0, y_norms + idx_j + j0 + 0);
                    const auto y_norm_1 =
                            svld1_f32(mask_1, y_norms + idx_j + j0 + lanes);

                    // load values for dot products
                    const auto ip_0 = svld1_f32(mask_0, ip_line + 0);
                    const auto ip_1 = svld1_f32(mask_1, ip_line + lanes);

                    // compute dis = y_norm[j] - 2 * dot(x_norm[i], y_norm[j]).
                    // x_norm[i] was dropped off because it is a constant for a
                    // given i. We'll deal with it later.
                    const auto distances_0 =
                            svmla_n_f32_z(mask_0, y_norm_0, ip_0, -2.f);
                    const auto distances_1 =
                            svmla_n_f32_z(mask_1, y_norm_1, ip_1, -2.f);

                    // compare the new distances to the min distances
                    // for each of the first group of 4 ARM SIMD components.
                    auto comparison =
                            svcmpgt_f32(mask_0, min_distances, distances_0);

                    // update min distances and indices with closest vectors if
                    // needed.
                    min_distances =
                            svsel_f32(comparison, distances_0, min_distances);
                    min_indices =
                            svsel_u32(comparison, current_indices, min_indices);
                    current_indices = svadd_n_u32_x(
                            mask_0,
                            current_indices,
                            static_cast<uint32_t>(lanes));

                    // compare the new distances to the min distances
                    // for each of the second group of 4 ARM SIMD components.
                    comparison =
                            svcmpgt_f32(mask_1, min_distances, distances_1);

                    // update min distances and indices with closest vectors if
                    // needed.
                    min_distances =
                            svsel_f32(comparison, distances_1, min_distances);
                    min_indices =
                            svsel_u32(comparison, current_indices, min_indices);
                    current_indices = svadd_n_u32_x(
                            mask_1,
                            current_indices,
                            static_cast<uint32_t>(lanes));
                }

                // add missing x_norms[i]
                // negative values can occur for identical vectors
                //    due to roundoff errors.
                auto mask = svwhilelt_b32_u64(0, count);
                min_distances = svadd_n_f32_z(
                        svcmpge_n_f32(mask, min_distances, -x_norms[i]),
                        min_distances,
                        x_norms[i]);
                min_indices = svadd_n_u32_x(
                        mask, min_indices, static_cast<uint32_t>(j0));
                mask = svcmple_n_f32(mask, min_distances, res.dis_tab[i]);
                if (svcntp_b32(svptrue_b32(), mask) == 0)
                    res.add_result(i, res.dis_tab[i], res.ids_tab[i]);
                else {
                    const auto min_distance = svminv_f32(mask, min_distances);
                    const auto min_index = svminv_u32(
                            svcmpeq_n_f32(mask, min_distances, min_distance),
                            min_indices);
                    res.add_result(i, min_distance, min_index);
                }
            }
        }
        // Does nothing for SingleBestResultHandler, but
        // keeping the call for the consistency.
        res.end_multiple();
        InterruptCallback::check();
    }
    return true;
}

template <>
void exhaustive_L2sqr_blas_simd<SIMDLevel::ARM_SVE>(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        Top1BlockResultHandler<CMax<float, int64_t>>& res,
        const float* y_norms) {
    if (nx == 0 || ny == 0) {
        return;
    }

    if (exhaustive_L2sqr_fused_cmax_simdlib<SIMDLevel::ARM_SVE>(
                x, y, d, nx, ny, res, y_norms)) {
        return;
    }

    exhaustive_L2sqr_blas_simd<SIMDLevel::NONE>(x, y, d, nx, ny, res, y_norms);
}

} // namespace faiss
