/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <random>

#include <faiss/IndexFlat.h>
#include <faiss/IndexNNDescent.h>

using namespace std::chrono;

int main(void) {
    // dimension of the vectors to index
    int d = 64;
    int K = 64;

    // size of the database we plan to index
    size_t nb = 10000;

    std::mt19937 rng(12345);

    // make the index object and train it
    faiss::IndexNNDescentFlat index(d, K, faiss::METRIC_L2);
    index.nndescent.S = 10;
    index.nndescent.R = 32;
    index.nndescent.L = K;
    index.nndescent.iter = 10;
    index.verbose = true;

    // generate labels by IndexFlat
    faiss::IndexFlat bruteforce(d, faiss::METRIC_L2);

    std::vector<float> database(nb * d);
    for (size_t i = 0; i < nb * d; i++) {
        database[i] = rng() % 1024;
    }

    { // populating the database
        index.add(nb, database.data());
        bruteforce.add(nb, database.data());
    }

    size_t nq = 1000;

    { // searching the database
        printf("Searching ...\n");
        index.nndescent.search_L = 50;

        std::vector<float> queries(nq * d);
        for (size_t i = 0; i < nq * d; i++) {
            queries[i] = rng() % 1024;
        }

        int k = 5;
        std::vector<faiss::idx_t> nns(k * nq);
        std::vector<faiss::idx_t> gt_nns(k * nq);
        std::vector<float> dis(k * nq);

        auto start = high_resolution_clock::now();
        index.search(nq, queries.data(), k, dis.data(), nns.data());
        auto end = high_resolution_clock::now();

        // find exact kNNs by brute force search
        bruteforce.search(nq, queries.data(), k, dis.data(), gt_nns.data());

        int recalls = 0;
        for (size_t i = 0; i < nq; ++i) {
            for (int n = 0; n < k; n++) {
                for (int m = 0; m < k; m++) {
                    if (nns[i * k + n] == gt_nns[i * k + m]) {
                        recalls += 1;
                    }
                }
            }
        }
        float recall = 1.0f * recalls / (k * nq);
        auto t = duration_cast<microseconds>(end - start).count();
        int qps = nq * 1.0f * 1000 * 1000 / t;

        printf("Recall@%d: %f, QPS: %d\n", k, recall, qps);
    }
}
