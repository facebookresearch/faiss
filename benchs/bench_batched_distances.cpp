/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Benchmark for batched SIMD threshold distance functions.
 *
 * Compares standard distance computation against batched versions that
 * support early abort when a threshold is exceeded. The batched functions
 * process dimensions in fixed-size batches using existing SIMD routines,
 * checking the threshold at batch boundaries.
 *
 * Usage: bench_batched_distances [num_vectors] [dimensions] [k] [num_queries]
 * Defaults: 100000 vectors, 2048 dimensions, k=100, 50 queries
 */

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include <faiss/utils/distances.h>
#include <faiss/utils/random.h>
#include <faiss/utils/utils.h>

using namespace faiss;

/// Run a k-NN scan using standard full distance computation.
/// Returns total time in milliseconds.
double bench_standard_L2(
        const float* queries,
        const float* database,
        size_t nq,
        size_t nb,
        size_t d,
        size_t k) {
    double t0 = getmillisecs();

    for (size_t qi = 0; qi < nq; qi++) {
        const float* q = queries + qi * d;

        // Simple max-heap to track top-k (threshold = heap top)
        std::vector<float> heap_dis(k, INFINITY);
        float threshold = INFINITY;

        for (size_t j = 0; j < nb; j++) {
            const float* y = database + j * d;
            float dis = fvec_L2sqr(q, y, d);
            if (dis < threshold) {
                // Find max in heap and replace
                size_t max_idx = 0;
                for (size_t h = 1; h < k; h++) {
                    if (heap_dis[h] > heap_dis[max_idx])
                        max_idx = h;
                }
                heap_dis[max_idx] = dis;
                // Update threshold
                threshold = heap_dis[0];
                for (size_t h = 1; h < k; h++) {
                    if (heap_dis[h] > threshold)
                        threshold = heap_dis[h];
                }
            }
        }
    }

    return getmillisecs() - t0;
}

/// Run a k-NN scan using batched distance with early abort.
/// Returns total time in milliseconds.
double bench_batched_L2(
        const float* queries,
        const float* database,
        size_t nq,
        size_t nb,
        size_t d,
        size_t k,
        size_t batch_size) {
    double t0 = getmillisecs();

    for (size_t qi = 0; qi < nq; qi++) {
        const float* q = queries + qi * d;

        std::vector<float> heap_dis(k, INFINITY);
        float threshold = INFINITY;

        for (size_t j = 0; j < nb; j++) {
            const float* y = database + j * d;
            float dis = fvec_L2sqr_batched(q, y, d, batch_size, threshold);
            if (dis < threshold) {
                size_t max_idx = 0;
                for (size_t h = 1; h < k; h++) {
                    if (heap_dis[h] > heap_dis[max_idx])
                        max_idx = h;
                }
                heap_dis[max_idx] = dis;
                threshold = heap_dis[0];
                for (size_t h = 1; h < k; h++) {
                    if (heap_dis[h] > threshold)
                        threshold = heap_dis[h];
                }
            }
        }
    }

    return getmillisecs() - t0;
}

/// Run a k-NN scan using standard inner product.
double bench_standard_IP(
        const float* queries,
        const float* database,
        size_t nq,
        size_t nb,
        size_t d,
        size_t k) {
    double t0 = getmillisecs();

    for (size_t qi = 0; qi < nq; qi++) {
        const float* q = queries + qi * d;

        std::vector<float> heap_sim(k, -INFINITY);
        float threshold = -INFINITY;

        for (size_t j = 0; j < nb; j++) {
            const float* y = database + j * d;
            float sim = fvec_inner_product(q, y, d);
            if (sim > threshold) {
                size_t min_idx = 0;
                for (size_t h = 1; h < k; h++) {
                    if (heap_sim[h] < heap_sim[min_idx])
                        min_idx = h;
                }
                heap_sim[min_idx] = sim;
                threshold = heap_sim[0];
                for (size_t h = 1; h < k; h++) {
                    if (heap_sim[h] < threshold)
                        threshold = heap_sim[h];
                }
            }
        }
    }

    return getmillisecs() - t0;
}

/// Run a k-NN scan using batched inner product with early abort.
double bench_batched_IP(
        const float* queries,
        const float* database,
        size_t nq,
        size_t nb,
        size_t d,
        size_t k,
        size_t batch_size) {
    double t0 = getmillisecs();

    for (size_t qi = 0; qi < nq; qi++) {
        const float* q = queries + qi * d;

        std::vector<float> heap_sim(k, -INFINITY);
        float threshold = -INFINITY;

        for (size_t j = 0; j < nb; j++) {
            const float* y = database + j * d;
            float sim = fvec_inner_product_batched(
                    q, y, d, batch_size, threshold);
            if (sim > threshold) {
                size_t min_idx = 0;
                for (size_t h = 1; h < k; h++) {
                    if (heap_sim[h] < heap_sim[min_idx])
                        min_idx = h;
                }
                heap_sim[min_idx] = sim;
                threshold = heap_sim[0];
                for (size_t h = 1; h < k; h++) {
                    if (heap_sim[h] < threshold)
                        threshold = heap_sim[h];
                }
            }
        }
    }

    return getmillisecs() - t0;
}

/// Normalize vectors in-place to unit length.
void normalize_vectors(float* x, size_t n, size_t d) {
    for (size_t i = 0; i < n; i++) {
        float* xi = x + i * d;
        float norm = 0;
        for (size_t j = 0; j < d; j++) {
            norm += xi[j] * xi[j];
        }
        norm = std::sqrt(norm);
        if (norm > 0) {
            for (size_t j = 0; j < d; j++) {
                xi[j] /= norm;
            }
        }
    }
}

int main(int argc, char** argv) {
    size_t nb = 100000;
    size_t d = 2048;
    size_t k = 100;
    size_t nq = 50;

    if (argc > 1) nb = atol(argv[1]);
    if (argc > 2) d = atol(argv[2]);
    if (argc > 3) k = atol(argv[3]);
    if (argc > 4) nq = atol(argv[4]);

    printf("Batched SIMD Threshold Distance Benchmark\n");
    printf("==========================================\n");
    printf("Database:   %zu vectors x %zu dimensions\n", nb, d);
    printf("Queries:    %zu\n", nq);
    printf("k:          %zu\n", k);
    printf("\n");

    // Generate random data
    printf("Generating random data...\n");
    std::vector<float> database(nb * d);
    std::vector<float> queries(nq * d);
    float_randn(database.data(), nb * d, 12345);
    float_randn(queries.data(), nq * d, 54321);

    // L2 Distance Benchmark
    printf("\n--- L2 Squared Distance (random data) ---\n");
    {
        double t_std = bench_standard_L2(
                queries.data(), database.data(), nq, nb, d, k);
        printf("Standard:     %.1f ms (%.2f ms/query)\n", t_std, t_std / nq);

        size_t batch_sizes[] = {4, 8, 16, 32, 64};
        for (size_t bs : batch_sizes) {
            double t_bat = bench_batched_L2(
                    queries.data(), database.data(), nq, nb, d, k, bs);
            double speedup = t_std / t_bat;
            printf("Batched(bs=%2zu): %.1f ms (%.2f ms/query)  speedup: %.2fx\n",
                   bs, t_bat, t_bat / nq, speedup);
        }
    }

    // Inner Product Benchmark (normalized vectors)
    printf("\n--- Inner Product (normalized, random data) ---\n");
    {
        // Normalize for inner product
        normalize_vectors(database.data(), nb, d);
        normalize_vectors(queries.data(), nq, d);

        double t_std = bench_standard_IP(
                queries.data(), database.data(), nq, nb, d, k);
        printf("Standard:     %.1f ms (%.2f ms/query)\n", t_std, t_std / nq);

        size_t batch_sizes[] = {4, 8, 16, 32, 64};
        for (size_t bs : batch_sizes) {
            double t_bat = bench_batched_IP(
                    queries.data(), database.data(), nq, nb, d, k, bs);
            double speedup = t_std / t_bat;
            printf("Batched(bs=%2zu): %.1f ms (%.2f ms/query)  speedup: %.2fx\n",
                   bs, t_bat, t_bat / nq, speedup);
        }
    }

    // Clustered data: create clusters with spread, queries near cluster
    // centers. This creates a more realistic scenario where many database
    // vectors are far from the query, enabling more early aborts.
    printf("\n--- L2 Squared Distance (clustered data) ---\n");
    {
        size_t nclusters = 10;
        // Generate cluster centers
        std::vector<float> centers(nclusters * d);
        float_randn(centers.data(), nclusters * d, 99999);
        // Scale centers apart
        for (size_t i = 0; i < nclusters * d; i++) {
            centers[i] *= 10.0f;
        }

        // Assign each database vector to a random cluster with small noise
        std::vector<float> db_clustered(nb * d);
        for (size_t i = 0; i < nb; i++) {
            size_t ci = i % nclusters;
            for (size_t j = 0; j < d; j++) {
                db_clustered[i * d + j] =
                        centers[ci * d + j] + database[i * d + j] * 0.1f;
            }
        }

        // Queries are near cluster 0
        std::vector<float> q_clustered(nq * d);
        for (size_t i = 0; i < nq; i++) {
            for (size_t j = 0; j < d; j++) {
                q_clustered[i * d + j] =
                        centers[j] + queries[i * d + j] * 0.1f;
            }
        }

        double t_std = bench_standard_L2(
                q_clustered.data(), db_clustered.data(), nq, nb, d, k);
        printf("Standard:     %.1f ms (%.2f ms/query)\n", t_std, t_std / nq);

        size_t batch_sizes[] = {4, 8, 16, 32, 64};
        for (size_t bs : batch_sizes) {
            double t_bat = bench_batched_L2(
                    q_clustered.data(), db_clustered.data(), nq, nb, d, k, bs);
            double speedup = t_std / t_bat;
            printf("Batched(bs=%2zu): %.1f ms (%.2f ms/query)  speedup: %.2fx\n",
                   bs, t_bat, t_bat / nq, speedup);
        }
    }

    printf("\nDone.\n");
    return 0;
}
