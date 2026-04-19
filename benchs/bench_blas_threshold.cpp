/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Benchmark to find the crossover point where BLAS becomes faster than
// sequential scan in IndexFlatL2, as a function of (nq, nb, d).

#include <faiss/IndexFlat.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/random.h>
#include <faiss/utils/utils.h>

#include <omp.h>

#include <cinttypes>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <vector>

void run_benchmark(int d, int nq, int nb, int k, int nrun) {
    std::vector<float> xb(static_cast<size_t>(nb) * d);
    faiss::rand_smooth_vectors(nb, d, xb.data(), 1234);

    std::vector<float> xq(static_cast<size_t>(nq) * d);
    faiss::rand_smooth_vectors(nq, d, xq.data(), 5678);

    std::vector<float> D(nq * k);
    std::vector<faiss::idx_t> I(nq * k);

    faiss::IndexFlatL2 index(d);
    index.add(nb, xb.data());

    // --- Force BLAS path ---
    faiss::distance_compute_blas_threshold = 1;
    index.search(nq, xq.data(), k, D.data(), I.data()); // warmup

    double blas_time = 0;
    for (int run = 0; run < nrun; run++) {
        double t0 = faiss::getmillisecs();
        index.search(nq, xq.data(), k, D.data(), I.data());
        blas_time += faiss::getmillisecs() - t0;
    }
    blas_time /= nrun;

    // --- Force Sequential path ---
    faiss::distance_compute_blas_threshold = INT64_MAX;
    index.search(nq, xq.data(), k, D.data(), I.data()); // warmup

    double seq_time = 0;
    for (int run = 0; run < nrun; run++) {
        double t0 = faiss::getmillisecs();
        index.search(nq, xq.data(), k, D.data(), I.data());
        seq_time += faiss::getmillisecs() - t0;
    }
    seq_time /= nrun;

    double speedup = seq_time / blas_time;
    int64_t product = static_cast<int64_t>(nq) * nb * d;
    const char* winner = (speedup > 1.05) ? "BLAS"
            : (speedup < 0.95)            ? "Seq"
                                          : "~tie";

    printf("%5d %5d %10d | %14" PRId64 " | %10.2f %10.2f | %6.2fx | %s\n",
           d,
           nq,
           nb,
           product,
           blas_time,
           seq_time,
           speedup,
           winner);
}

int main() {
    omp_set_num_threads(1);

    const int k = 10;
    const int nrun = 5;

    // Wide sweep to find the crossover
    std::vector<int> dims = {16, 32, 64, 128, 256, 512};
    std::vector<int> nqs = {1, 2, 5, 10, 20, 50, 100};
    std::vector<int> nbs = {100, 1000, 10000, 100000, 1000000};

    printf("Benchmark: BLAS vs Sequential crossover finder\n");
    printf("  k=%d, nrun=%d, single-threaded\n\n", k, nrun);

    for (int d : dims) {
        printf("\n=== d=%d ===\n", d);
        printf("%5s %5s %10s | %14s | %10s %10s | %7s | %s\n",
               "d",
               "nq",
               "nb",
               "nq*nb*d",
               "BLAS(ms)",
               "Seq(ms)",
               "Speedup",
               "Winner");
        printf("------|------|-----------|----------------|"
               "------------|------------|---------|-------\n");

        for (int nb : nbs) {
            for (int nq : nqs) {
                run_benchmark(d, nq, nb, k, nrun);
            }
        }
    }

    printf("\n'BLAS' = BLAS >5%% faster, 'Seq' = Sequential >5%% faster, "
           "'~tie' = within 5%%\n");

    return 0;
}
