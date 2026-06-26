/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Benchmark to find the BLAS vs Sequential crossover point for
// IndexFlat search as a function of (nq, nb, d, omp_threads).
// Prints a clean table with speedup ratios.

#include <faiss/IndexFlat.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/random.h>
#include <faiss/utils/utils.h>

#include <omp.h>

#include <cinttypes>
#include <climits>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

void run_config(int d, int nq, int nb, int k, int nrun) {
    std::vector<float> xb(static_cast<size_t>(nb) * d);
    faiss::float_rand(xb.data(), xb.size(), 123);

    std::vector<float> xq(static_cast<size_t>(nq) * d);
    faiss::float_rand(xq.data(), xq.size(), 456);

    std::vector<float> D(nq * k);
    std::vector<faiss::idx_t> I(nq * k);

    faiss::IndexFlatL2 index(d);
    index.add(nb, xb.data());

    // --- BLAS ---
    faiss::distance_compute_blas_threshold = 1;
    index.search(nq, xq.data(), k, D.data(), I.data()); // warmup

    double blas_time = 0;
    for (int run = 0; run < nrun; run++) {
        double t0 = faiss::getmillisecs();
        index.search(nq, xq.data(), k, D.data(), I.data());
        blas_time += faiss::getmillisecs() - t0;
    }
    blas_time /= nrun;

    // --- Sequential ---
    faiss::distance_compute_blas_threshold = INT64_MAX;
    index.search(nq, xq.data(), k, D.data(), I.data()); // warmup

    double seq_time = 0;
    for (int run = 0; run < nrun; run++) {
        double t0 = faiss::getmillisecs();
        index.search(nq, xq.data(), k, D.data(), I.data());
        seq_time += faiss::getmillisecs() - t0;
    }
    seq_time /= nrun;

    double ratio = seq_time / blas_time;
    int nthreads = omp_get_max_threads();
    int64_t nqd = static_cast<int64_t>(nq) * d;

    const char* winner = (ratio > 1.1) ? "BLAS"
            : (ratio < 0.9)            ? "Seq"
                                       : "~tie";

    printf("%3d %5d %5d %10d | %8" PRId64 " | %10.1f %10.1f | %6.2fx | %s\n",
           nthreads,
           d,
           nq,
           nb,
           nqd,
           blas_time,
           seq_time,
           ratio,
           winner);
    fflush(stdout);
}

int main(int argc, char** argv) {
    int k = 10;
    int nrun = 5;

    // Parse args: bench_blas_threshold [nb] [nthreads_list]
    // nthreads_list is comma-separated, e.g. "1,10,40,80,166"
    // If nthreads_list is "0" or omitted, uses default OMP threads only.
    int nb = 100000;
    std::vector<int> thread_counts;

    if (argc > 1)
        nb = atoi(argv[1]);

    if (argc > 2) {
        // Parse comma-separated thread counts
        char* str = argv[2];
        char* token = strtok(str, ",");
        while (token != nullptr) {
            int t = atoi(token);
            if (t > 0) {
                thread_counts.push_back(t);
            }
            token = strtok(nullptr, ",");
        }
    }

    // Default: just use current OMP setting
    if (thread_counts.empty()) {
        thread_counts.push_back(0); // 0 means "use default"
    }

    // Choose dimensions based on nb to avoid OOM
    std::vector<int> dims;
    if (nb <= 1000000) {
        dims = {64, 128, 256, 512, 1024};
    } else if (nb <= 5000000) {
        dims = {64, 128, 256, 512};
    } else if (nb <= 20000000) {
        dims = {64, 128, 256};
    } else {
        dims = {64, 128};
    }

    std::vector<int> nqs = {1, 5, 10, 20, 50, 100, 200, 500, 1000};

    int default_threads = omp_get_max_threads();

    for (int nthreads : thread_counts) {
        if (nthreads > 0) {
            omp_set_num_threads(nthreads);
        } else {
            omp_set_num_threads(default_threads);
            nthreads = default_threads;
        }

        printf("\n========================================\n");
        printf("BLAS vs Sequential benchmark\n");
        printf("  nb=%d, k=%d, nrun=%d, omp_threads=%d\n",
               nb,
               k,
               nrun,
               omp_get_max_threads());
        printf("========================================\n\n");
        printf("thr     d    nq         nb |     nq*d"
               " |   BLAS(ms)    Seq(ms) | Ratio  | Winner\n");
        printf("----|------|------|-----------|--------"
               "-|-----------|---------|--------|-------\n");

        for (int d : dims) {
            for (int nq : nqs) {
                run_config(d, nq, nb, k, nrun);
            }
            printf("\n");
        }
    }

    // Restore default
    omp_set_num_threads(default_threads);

    return 0;
}
