/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdio>
#include <random>

#include <faiss/Index.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/MetricType.h>

// 64-bit int
using idx_t = faiss::idx_t;

int main() {
    int d = 64;  // dimension
    int nb = 10; // database size
    int nq = 1;  // nb of queries

    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;

    float* xb = new float[d * nb];
    float* xq = new float[d * nq];

    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < d; j++)
            xb[d * i + j] = distrib(rng);
        xb[d * i] += i / 1000.;
    }

    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < d; j++)
            xq[d * i + j] = distrib(rng);
        xq[d * i] += i / 1000.;
    }

    int k = 4;
    int m = 8;
    faiss::Index* index =
            new faiss::IndexHNSWFlat(d, m, faiss::MetricType::METRIC_L2);
    printf("is_trained = %s\n", index->is_trained ? "true" : "false");
    index->add(nb, xb); // add vectors to the index
    printf("ntotal = %zd\n", index->ntotal);

    { // sanity check: search 5 first vectors of xb
        idx_t* I = new idx_t[k * 5];
        float* D = new float[k * 5];

        index->search(5, xb, k, D, I);

        // print results
        printf("I=\n");
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < k; j++)
                printf("%5zd ", I[i * k + j]);
            printf("\n");
        }

        printf("D=\n");
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < k; j++)
                printf("%7g ", D[i * k + j]);
            printf("\n");
        }

        delete[] I;
        delete[] D;
    }

    { // search 5 first vectors of xb with deduper
        idx_t* I = new idx_t[k * 5];
        float* D = new float[k * 5];
        std::unordered_map<idx_t, idx_t> group;
        for (int i = 0; i < nb; i++) {
            group[i] = i % 2;
        }
        faiss::IDDeduperMap idDeduper(&group);
        auto pSearchParameters = new faiss::SearchParametersHNSW();
        pSearchParameters->dedup = &idDeduper;

        index->search(5, xb, k, D, I, pSearchParameters);

        // print results
        printf("I=\n");
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < k; j++)
                printf("%5zd ", I[i * k + j]);
            printf("\n");
        }

        printf("D=\n");
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < k; j++)
                printf("%7g ", D[i * k + j]);
            printf("\n");
        }

        delete[] I;
        delete[] D;
    }

    delete[] xb;
    delete[] xq;

    return 0;
}
