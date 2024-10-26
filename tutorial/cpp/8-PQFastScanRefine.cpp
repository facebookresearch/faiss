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
#include <faiss/IndexRefine.h>

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
    faiss::IndexRefineFlat index_refine(&index);
    // refine index after PQFastScan

    printf("Index is trained? %s\n",
           index_refine.is_trained ? "true" : "false");
    index_refine.train(nb, xb);
    printf("Index is trained? %s\n",
           index_refine.is_trained ? "true" : "false");
    index_refine.add(nb, xb);

    int k = 4;
    { // search xq
        idx_t* I = new idx_t[(int)(k * nq)];
        float* D = new float[(int)(k * nq)];
        float k_factor = 3;
        faiss::IndexRefineSearchParameters* params =
                new faiss::IndexRefineSearchParameters();
        params->k_factor = k_factor;
        index_refine.search(nq, xq, k, D, I, params);

        printf("I=\n");
        for (int i = nq - 5; i < nq; i++) {
            for (int j = 0; j < k; j++) {
                printf("%5zd ", I[i * k + j]);
            }
            printf("\n");
        }

        delete[] I;
        delete[] D;
        delete params;
    }

    delete[] xb;
    delete[] xq;

    return 0;
}
