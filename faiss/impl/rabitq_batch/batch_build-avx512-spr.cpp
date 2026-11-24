/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Compiled in the AVX512_SPR SIMD level, with -mavx512vbmi -mamx-tile
// -mamx-int8 added on top of the level flags. Only called after
// select_backend() checked the CPU.

#include <faiss/impl/rabitq_batch/batch_build.h>

#if FAISS_RABITQ_BATCH_BUILD

#include <faiss/impl/rabitq_batch/graph.h>
#include <faiss/impl/rabitq_batch/partition.h>

#include <omp.h>

#include <algorithm>

namespace faiss {
namespace rabitq_batch {

void build_avx512_spr(
        HNSW& h,
        const uint8_t* codes,
        const float* x,
        size_t n,
        size_t d,
        size_t code_size,
        double center_norm_sq,
        size_t k,
        size_t partition_threshold,
        bool vnni) {
    // at least about 1024 rows per thread, so that small databases are not
    // padded with many empty rows
    const size_t threads = std::min<size_t>(
            omp_get_max_threads(), std::max<size_t>(1, n / 1024));
    choose_entry(h, n);
    const size_t eff_threshold = partition_threshold > 0
            ? partition_threshold
            : default_partition_threshold;
    // Partitioning needs the float vectors (x) for k-means and enough points
    // for buckets to be meaningful; small batches always take the exact path.
    if (x != nullptr && n > eff_threshold && n >= 4096) {
        build_graph_partitioned(
                codes, x, n, d, code_size, center_norm_sq, h, k, vnni, threads);
        return;
    }
    Data data;
    data.m = Dims(d);
    data.n = n;
    data.threads = threads;
    data.codes = codes;
    data.code_size = code_size;
    build_graph(data, center_norm_sq, h, k, vnni);
}

} // namespace rabitq_batch
} // namespace faiss

#endif
