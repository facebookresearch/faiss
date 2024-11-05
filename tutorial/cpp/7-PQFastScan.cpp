/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cassert>
#include <cstdio>
#include <cstdlib>
#include <random>

#include <faiss/IndexPQFastScan.h>

using idx_t = faiss::idx_t;

int main() {
    int d = 64;      // dimension
    int nb = 100000; // database size
    int nq = 10000;  // nb of queries

    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;

    float* xb = new float[(int)(d * nb)];
    float* xq = new float[(int)(d * nq)];

    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < d; j++) {
            xb[d * i + j] = distrib(rng);
        }
        xb[d * i] += i / 1000.;
    }

    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < d; j++) {
            xq[d * i + j] = distrib(rng);
        }
        xq[d * i] += i / 1000.;
    }

    int m = 8;
    int n_bit = 4;

    faiss::IndexPQFastScan index(d, m, n_bit);
    printf("Index is trained? %s\n", index.is_trained ? "true" : "false");
    index.train(nb, xb);
    printf("Index is trained? %s\n", index.is_trained ? "true" : "false");
    index.add(nb, xb);

    int k = 4;

    { // search xq
        idx_t* I = new idx_t[(int)(k * nq)];
        float* D = new float[(int)(k * nq)];

        index.search(nq, xq, k, D, I);

        printf("I=\n");
        for (int i = nq - 5; i < nq; i++) {
            for (int j = 0; j < k; j++) {
                printf("%5zd ", I[i * k + j]);
            }
            printf("\n");
        }

        delete[] I;
        delete[] D;
    }

    delete[] xb;
    delete[] xq;

    return 0;
} // namespace facebook::detail
