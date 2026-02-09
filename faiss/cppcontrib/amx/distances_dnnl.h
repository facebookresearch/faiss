/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/* All distance functions for L2 and IP distances.
 * The actual functions are implemented in distances.cpp and distances_simd.cpp
 */

#include <faiss/cppcontrib/amx/onednn_utils.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/impl/platform_macros.h>
#include <omp.h>

#ifndef FINTEGER
#define FINTEGER long
#endif

namespace faiss {

// block sizes for oneDNN/AMX distance computations
FAISS_API int distance_compute_dnnl_query_bs = 10240;
FAISS_API int distance_compute_dnnl_database_bs = 10240;

/**
 * Find the nearest neighbors for nx queries in a set of ny vectors，
 * accelerated via oneDNN/AMX.
 */
template <class BlockResultHandler>
void exhaustive_inner_product_seq_dnnl(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        BlockResultHandler& res) {
    using SingleResultHandler =
            typename BlockResultHandler::SingleResultHandler;
    [[maybe_unused]] int nt = std::min(int(nx), omp_get_max_threads());

    std::unique_ptr<float[]> res_arr(new float[nx * ny]);

    comput_f32bf16f32_inner_product(
            nx,
            d,
            ny,
            d,
            const_cast<float*>(x),
            const_cast<float*>(y),
            res_arr.get());

#pragma omp parallel num_threads(nt)
    {
        SingleResultHandler resi(res);
#pragma omp for
        for (size_t i = 0; i < nx; i++) {
            resi.begin(i);
            for (size_t j = 0; j < ny; j++) {
                float ip = res_arr[i * ny + j];
                resi.add_result(ip, j);
            }
            resi.end();
        }
    }
}

/**
 * Find the nearest neighbors for nx queries in a set of ny vectors，
 * accelerated via oneDNN/AMX.
 */
template <class BlockResultHandler>
void exhaustive_inner_product_blas_dnnl(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        BlockResultHandler& res) {
    /* block sizes */
    const size_t bs_x = distance_compute_dnnl_query_bs;
    const size_t bs_y = distance_compute_dnnl_database_bs;
    std::unique_ptr<float[]> ip_block(new float[bs_x * bs_y]);

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
            FINTEGER nyi = j1 - j0, nxi = i1 - i0;
            comput_f32bf16f32_inner_product(
                    nxi,
                    d,
                    nyi,
                    d,
                    const_cast<float*>(x + i0 * d),
                    const_cast<float*>(y + j0 * d),
                    ip_block.get());

            res.add_results(j0, j1, ip_block.get());
        }
        res.end_multiple();
        InterruptCallback::check();
    }
}

} // namespace faiss
