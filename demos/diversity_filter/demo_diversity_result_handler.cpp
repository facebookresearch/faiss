/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <limits>
#include <memory>
#include <set>
#include <unordered_map>

#include <faiss/IndexIVF.h>
#include <faiss/index_factory.h>

#include "diversity_result_handler.h"

// compare results, taking into account that ties may be ordered arbitrarily
bool compare_results(
        int test_no,
        int K,
        const std::vector<float>& reference_D,
        const std::vector<size_t>& reference_I,
        const std::vector<float>& new_D,
        const std::vector<int64_t>& new_I,
        const std::vector<int>& group_ids) {
    float current_dis = NAN;
    std::set<int64_t> ref_ids;
    std::set<int64_t> new_ids;

    for (int i = 0; i < K; i++) {
        float d = reference_D[i];
        if (new_D[i] != d) {
            printf("test_no=%d Distance mismatch at %d: %g vs %g\n",
                   test_no,
                   i,
                   new_D[i],
                   reference_D[i]);
            return false;
        }
        if (d != current_dis) {
            // check that the ids for the previous distance are the same
            if (ref_ids != new_ids) {
                // find the extra elements from each set
                std::set<int64_t> extra_ref_ids;
                std::set<int64_t> extra_new_ids;
                std::set_difference(
                        ref_ids.begin(),
                        ref_ids.end(),
                        new_ids.begin(),
                        new_ids.end(),
                        std::inserter(extra_ref_ids, extra_ref_ids.end()));
                std::set_difference(
                        new_ids.begin(),
                        new_ids.end(),
                        ref_ids.begin(),
                        ref_ids.end(),
                        std::inserter(extra_new_ids, extra_new_ids.end()));
                if (extra_ref_ids.size() == 1 && extra_new_ids.size() == 1) {
                    int64_t id1 = *extra_ref_ids.begin();
                    int64_t id2 = *extra_new_ids.begin();
                    if (group_ids[id1] == group_ids[id2]) {
                        // this is a valid case - same group, just different
                        // tie-breaking
                        goto ok;
                    }
                }

                printf("test_no=%d i=%d Id mismatch at distance %g\n",
                       test_no,
                       i,
                       current_dis);
                printf("  ref_ids: {");
                for (auto id : ref_ids) {
                    printf(" %ld", id);
                }
                printf(" }\n");
                printf("  new_ids: {");
                for (auto id : new_ids) {
                    printf(" %ld", id);
                }
                printf(" }\n");
                printf("\n=== Reference results (all K) ===\n");
                for (int j = 0; j < K; j++) {
                    printf("  [%d] id=%zu distance=%g group=%d\n",
                           j,
                           reference_I[j],
                           reference_D[j],
                           group_ids[reference_I[j]]);
                }
                printf("\n=== New handler results (all K) ===\n");
                for (int j = 0; j < K; j++) {
                    printf("  [%d] id=%ld distance=%g group=%d\n",
                           j,
                           new_I[j],
                           new_D[j],
                           group_ids[new_I[j]]);
                }
                return false;
            }
        ok:
            current_dis = d;
            ref_ids.clear();
            new_ids.clear();
        }

        ref_ids.insert(reference_I[i]);
        new_ids.insert(new_I[i]);
    }
    return true;
}

// test result handler with a table of random distances and group ids, without
// reference to an index
void test_result_handler_table() {
    int N = 100;
    int K = 10;
    int n_groups = 5;
    int max_per_group = 3;

    // Fixed seed for reproducibility
    srand(12345);

    for (int test_no = 0; test_no < 200; test_no++) {
        std::vector<int> group_ids(N);
        std::vector<float> distance(N);

        for (int i = 0; i < N; i++) {
            group_ids[i] = rand() % n_groups;
            // distance[i] = (float)rand() / (float)RAND_MAX;
            // increase likelihood of same-distances
            distance[i] = rand() % 500 / 500.0;
        }

        // reference implementation
        std::vector<size_t> reference_I(K);
        std::vector<float> reference_D(K);
        {
            std::vector<size_t> permutation(N);
            std::iota(permutation.begin(), permutation.end(), 0);
            std::sort(
                    permutation.begin(),
                    permutation.end(),
                    [&](size_t i, size_t j) {
                        return distance[i] < distance[j];
                    });

            std::unordered_map<int, size_t> group_sizes;
            size_t n_res = 0;
            size_t perm_idx = 0;
            while (n_res < K && perm_idx < N) {
                size_t idx = permutation[perm_idx];
                int group_id = group_ids[idx];
                perm_idx++;
                if (group_sizes[group_id] < max_per_group) {
                    group_sizes[group_id]++;
                    reference_I[n_res] = idx;
                    reference_D[n_res] = distance[idx];
                    n_res++;
                }
            }
        }

        // run result_handler
        std::vector<int64_t> new_I(K);
        std::vector<float> new_D(K);
        {
            DiversityResultHandlerBubble handler(
                    group_ids, max_per_group, K, new_D.data(), new_I.data());
            for (int i = 0; i < N; i++) {
                handler.add_result(distance[i], i);
            }
        }

        if (!compare_results(
                    test_no,
                    K,
                    reference_D,
                    reference_I,
                    new_D,
                    new_I,
                    group_ids)) {
            printf("test_no=%d FAILED\n", test_no);
            return;
        }
    }

    printf("All tests passed!\n");
}

void test_diversity_index(const char* factory_string) {
    int d = 64;    // dimension
    int N = 2000;  // database size
    int nt = 5000; // training set size
    int nq = 100;  // number of queries
    int K = 10;    // number of results to return
    int n_groups = 5;
    int max_per_group = 3;

    printf("Testing diversity index with factory: %s\n", factory_string);

    // Create index using factory
    std::unique_ptr<faiss::Index> index(
            faiss::index_factory(d, factory_string, faiss::METRIC_L2));

    // Generate training data
    std::vector<float> xt(nt * d);
    faiss::rand_smooth_vectors(nt, d, xt.data(), 1234);

    // Generate database vectors
    std::vector<float> xb(N * d);
    faiss::rand_smooth_vectors(N, d, xb.data(), 4567);

    // Generate query vectors
    std::vector<float> xq(nq * d);
    faiss::rand_smooth_vectors(nq, d, xq.data(), 7890);

    // Train the index
    index->train(nt, xt.data());

    // Add database vectors
    index->add(N, xb.data());

    // Set nprobe for IVF indexes
    if (auto* ivf = dynamic_cast<faiss::IndexIVF*>(index.get())) {
        ivf->nprobe = 4;
    }

    // Generate group ids for each database vector
    srand(54321);
    std::vector<int> group_ids(N);
    for (int i = 0; i < N; i++) {
        group_ids[i] = rand() % n_groups;
    }

    // Test each query
    for (int q = 0; q < nq; q++) {
        const float* query = xq.data() + q * d;

        // Reference: first get all results, then filter for diversity
        std::vector<float> all_D(N);
        std::vector<faiss::idx_t> all_I(N);

        // Get all distances using a standard search with large k
        // We'll use the index's search to get ordered results
        int large_k = std::min(N, 500); // use a reasonably large k
        std::vector<float> ref_D(large_k);
        std::vector<faiss::idx_t> ref_I(large_k);
        index->search(1, query, large_k, ref_D.data(), ref_I.data());

        // Reference implementation: filter for diversity
        std::vector<size_t> reference_I(K);
        std::vector<float> reference_D(K);
        {
            std::unordered_map<int, size_t> group_sizes;
            size_t n_res = 0;
            for (int i = 0; i < large_k && n_res < K; i++) {
                faiss::idx_t idx = ref_I[i];
                if (idx < 0) {
                    continue;
                }
                int group_id = group_ids[idx];
                if (group_sizes[group_id] < max_per_group) {
                    group_sizes[group_id]++;
                    reference_I[n_res] = idx;
                    reference_D[n_res] = ref_D[i];
                    n_res++;
                }
            }
        }

        // Now use the DiversityResultHandlerBubble with search1
        std::vector<float> new_D(K);
        std::vector<int64_t> new_I(K);
        {
            DiversityResultHandlerBubble handler(
                    group_ids, max_per_group, K, new_D.data(), new_I.data());
            handler.threshold = std::numeric_limits<float>::max();
            index->search1(query, handler);
        }

        // Compare results
        if (!compare_results(
                    q, K, reference_D, reference_I, new_D, new_I, group_ids)) {
            printf("test_diversity_index: query %d FAILED\n", q);
            return;
        }
    }

    printf("test_diversity_index: All %d queries passed!\n", nq);
}

int main() {
    test_result_handler_table();
    test_diversity_index("IVF32,Flat");
    return 0;
}
