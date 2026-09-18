/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Standalone microbenchmark: four single estimates vs the batch-four API.
// Build target bench_rabitq_batch_estimates, then run with OMP_NUM_THREADS=1
// and OPENBLAS_NUM_THREADS=1. Query preparation is outside the timed region.
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <memory>
#include <random>
#include <vector>

#include <faiss/IndexRaBitQ.h>

int main() {
    constexpr int n = 4096, repetitions = 100;
    std::mt19937 rng(123);
    std::normal_distribution<float> normal;
    std::printf("dim,round,single_ns_per_code,batch_ns_per_code,speedup\n");
    for (int d : {128, 768, 960, 1536}) {
        std::vector<float> data(n * d), query(d);
        for (auto& v : data)
            v = normal(rng);
        for (auto& v : query)
            v = normal(rng);
        faiss::IndexRaBitQ index(d, faiss::METRIC_L2, 8);
        index.train(n, data.data());
        index.add(n, data.data());
        std::unique_ptr<faiss::FlatCodesDistanceComputer> owner(
                index.get_quantized_distance_computer(4, false));
        auto& dc = dynamic_cast<faiss::RaBitQDistanceComputer&>(*owner);
        dc.set_query(query.data());
        std::vector<const uint8_t*> codes(n);
        for (int i = 0; i < n; ++i) {
            codes[i] = index.codes.data() + i * index.code_size;
        }
        std::shuffle(codes.begin(), codes.end(), rng);
        auto measure = [&](bool batch) {
            double checksum = 0;
            const auto start = std::chrono::steady_clock::now();
            for (int r = 0; r < repetitions; ++r) {
                for (int i = 0; i < n; i += 4) {
                    float distances[4];
                    if (batch) {
                        dc.distance_to_code_1bit_batch_4(
                                codes.data() + i, distances);
                    } else {
                        for (int j = 0; j < 4; ++j) {
                            distances[j] =
                                    dc.distance_to_code_1bit(codes[i + j]);
                        }
                    }
                    for (float distance : distances)
                        checksum += distance;
                }
            }
            const double ns = std::chrono::duration<double, std::nano>(
                                      std::chrono::steady_clock::now() - start)
                                      .count();
            std::fprintf(
                    stderr,
                    "d=%d batch=%d checksum=%.9g\n",
                    d,
                    batch,
                    checksum);
            return ns / (n * repetitions);
        };
        measure(false);
        measure(true);
        for (int round = 0; round < 5; ++round) {
            double single, batch;
            if (round % 2) {
                batch = measure(true);
                single = measure(false);
            } else {
                single = measure(false);
                batch = measure(true);
            }
            std::printf(
                    "%d,%d,%.3f,%.3f,%.3f\n",
                    d,
                    round,
                    single,
                    batch,
                    single / batch);
        }
    }
}
