/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "faiss_avx512_result_handler.h"

#include <faiss/IndexIVF.h>
#include <faiss/index_factory.h>
#include <faiss/utils/random.h>
#include <faiss/utils/utils.h>
#include <omp.h>

#include <cstdio>
#include <memory>
#include <vector>

using namespace faiss;

// Parameters
constexpr int d = 64;          // dimension
constexpr size_t nb = 1000000; // database size
constexpr size_t nt = 10000;   // training size
constexpr size_t nq = 100;     // number of queries
constexpr int nrun = 5;        // number of timing runs

int main() {
    // Use single OpenMP thread

    printf("Generating nt=%zu nb=%zu nq=%zu vectors of dimension %d\n",
           nt,
           nb,
           nq,
           d);
    std::vector<float> xt(nt * d), xb(nb * d), xq(nq * d);
    rand_smooth_vectors(nt, d, xt.data(), 1234);
    rand_smooth_vectors(nb, d, xb.data(), 4567);
    rand_smooth_vectors(nq, d, xq.data(), 7890);

    // Build IVF1024,Flat index
    printf("Building IVF1024,Flat index...\n");
    std::unique_ptr<Index> index(index_factory(d, "IVF1024,Flat", METRIC_L2));

    printf("Training index...\n");
    index->train(nt, xt.data());

    printf("Adding %zu vectors to index...\n", nb);
    index->add(nb, xb.data());

    // Set nprobe for IVF index
    IndexIVF* index_ivf = dynamic_cast<IndexIVF*>(index.get());
    if (index_ivf) {
    }
    omp_set_num_threads(1);

    // Test with varying k values
    std::vector<size_t> k_values = {1, 10, 20, 50, 100, 200, 500, 1000};
    std::vector<size_t> nprobe_values = {1, 2, 4, 8, 16, 64};

    printf("\nBenchmarking with %d OpenMP thread(s), %d runs per config\n",
           omp_get_max_threads(),
           nrun);
    printf("%-8s %15s %15s %10s\n",
           "k",
           "baseline(ms)",
           "avx512(ms)",
           "speedup");
    printf("------------------------------------------------------------\n");

    for (size_t nprobe : nprobe_values) {
        index_ivf->nprobe = nprobe;
        printf("============ nprobe=%zu ===========\n", nprobe);
        for (size_t k : k_values) {
            std::vector<float> D_ref(nq * k);
            std::vector<idx_t> I_ref(nq * k);
            std::vector<float> D_avx(nq * k);
            std::vector<idx_t> I_avx(nq * k);

            // Warmup
            index->search(nq, xq.data(), k, D_ref.data(), I_ref.data());

            // Benchmark baseline search
            double t0 = getmillisecs();
            for (int run = 0; run < nrun; run++) {
                for (size_t q = 0; q < nq; q++) {
                    index->search(
                            1,
                            xq.data() + q * d,
                            k,
                            D_ref.data() + q * k,
                            I_ref.data() + q * k);
                }
            }
            double baseline_time = (getmillisecs() - t0) / nrun;

            // Warmup AVX512 handler
            ReservoirResultHandlerAVX512 handler(k);
            for (size_t q = 0; q < nq; q++) {
                handler.begin();
                index->search1(xq.data() + q * d, handler);
                handler.end(D_avx.data() + q * k, I_avx.data() + q * k);
            }

            // Benchmark AVX512 result handler
            t0 = getmillisecs();
            for (int run = 0; run < nrun; run++) {
                for (size_t q = 0; q < nq; q++) {
                    handler.begin();
                    index->search1(xq.data() + q * d, handler);
                    handler.end(D_avx.data() + q * k, I_avx.data() + q * k);
                }
            }
            double avx512_time = (getmillisecs() - t0) / nrun;

            double speedup = baseline_time / avx512_time;
            printf("%-8zu %15.3f %15.3f %10.2fx\n",
                   k,
                   baseline_time,
                   avx512_time,
                   speedup);
        }
    }

    return 0;
}
