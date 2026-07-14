/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <algorithm>
#include <cstdint>
#include <string>
#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/utils/random.h>

#include "bench_cli_utils.h" // BENCH_DEFINE_DATASET_FILE_FLAGS, bench_init, int_list
#include "bench_dataset_utils.h" // DatasetSIFT1M, compute_recall_at, ...

// Shared helpers for the GPU benchmark suite (benchs/gpu). Kept separate from
// benchs/cpp/bench_dataset_utils.h (which this includes) because the GPU
// benches additionally need synthetic-data ground truth computed on the CPU so
// that recall can be reported even when no on-disk dataset is present.

namespace benchmarks {

/// A {train, base, query, groundtruth} working set for the GPU benches. It is
/// either loaded from an on-disk SIFT1M-style dataset (via DatasetSIFT1M) or
/// generated synthetically with exact CPU-computed ground truth, so every
/// bench can report recall regardless of whether a dataset is available.
struct GpuBenchData {
    size_t d = 0;
    size_t nb = 0;
    size_t nq = 0;
    size_t nt = 0;
    size_t gt_k = 0;

    std::vector<float> xb;   // base vectors   [nb x d]
    std::vector<float> xq;   // query vectors  [nq x d]
    std::vector<float> xt;   // train vectors  [nt x d]
    std::vector<int32_t> gt; // ground truth   [nq x gt_k]

    std::string tag = "synthetic"; // label prefix for benchmark names
    bool from_dataset = false;
};

/// Compute exact L2 ground truth for `data` using a CPU brute-force index.
/// Fills data.gt ([nq x gt_k], int32) so recall@k (k <= gt_k) can be measured.
inline void compute_synthetic_gt(GpuBenchData& data, int gt_k) {
    faiss::IndexFlatL2 index(data.d);
    index.add(data.nb, data.xb.data());

    std::vector<float> distances((size_t)data.nq * gt_k);
    std::vector<int64_t> labels((size_t)data.nq * gt_k);
    index.search(
            data.nq, data.xq.data(), gt_k, distances.data(), labels.data());

    data.gt_k = gt_k;
    data.gt.resize((size_t)data.nq * gt_k);
    for (size_t i = 0; i < (size_t)data.nq * gt_k; i++) {
        data.gt[i] = (int32_t)labels[i];
    }
}

/// Best-effort dataset acquisition shared by every GPU bench. If `data_dir`
/// holds a loadable SIFT1M-style dataset, it is used verbatim. Otherwise
/// synthetic float data
/// of the requested shape is generated and exact ground truth is computed on
/// the CPU. `gt_k` bounds the largest k / recall rank the caller can report.
inline GpuBenchData gpu_load_or_synth(
        const std::string& data_dir,
        const std::string& train_file,
        const std::string& base_file,
        const std::string& query_file,
        const std::string& gt_file,
        size_t d,
        size_t nb,
        size_t nq,
        size_t nt,
        size_t gt_k = 100) {
    GpuBenchData data;

    DatasetSIFT1M sift;
    if (dataset_available(data_dir) &&
        sift.load(data_dir, train_file, base_file, query_file, gt_file)) {
        data.d = sift.d;
        data.nb = sift.nb;
        data.nq = sift.nq;
        data.nt = sift.nt;
        data.gt_k = sift.gt_k;
        data.xb = std::move(sift.xb);
        data.xq = std::move(sift.xq);
        data.xt = std::move(sift.xt);
        data.gt = std::move(sift.gt);
        data.tag = dataset_label(base_file);
        data.from_dataset = true;
        return data;
    }

    // Synthetic fallback: reproducible pseudo-random float vectors.
    data.d = d;
    data.nb = nb;
    data.nq = nq;
    data.nt = nt;
    data.xt.resize((size_t)nt * d);
    data.xb.resize((size_t)nb * d);
    data.xq.resize((size_t)nq * d);
    faiss::float_rand(data.xt.data(), (size_t)nt * d, 12345);
    faiss::float_rand(data.xb.data(), (size_t)nb * d, 54321);
    faiss::float_rand(data.xq.data(), (size_t)nq * d, 67890);

    int eff_gt_k = std::min(gt_k, nb);
    compute_synthetic_gt(data, eff_gt_k);
    data.tag = "synthetic";
    data.from_dataset = false;
    return data;
}

} // namespace benchmarks
