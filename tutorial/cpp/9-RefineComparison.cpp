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
#include <faiss/index_factory.h>
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

    // Constructing the refine PQ index with SQfp16 with index factory
    faiss::Index* index_fp16;
    index_fp16 = faiss::index_factory(
            d, "PQ32x4fs,Refine(SQfp16)", faiss::METRIC_L2);
    index_fp16->train(nb, xb);
    index_fp16->add(nb, xb);

    // Constructing the refine PQ index with SQ8
    faiss::Index* index_sq8;
    index_sq8 =
            faiss::index_factory(d, "PQ32x4fs,Refine(SQ8)", faiss::METRIC_L2);
    index_sq8->train(nb, xb);
    index_sq8->add(nb, xb);

    int k = 10;
    { // search xq
        idx_t* I_fp16 = new idx_t[(int)(k * nq)];
        float* D_fp16 = new float[(int)(k * nq)];
        idx_t* I_sq8 = new idx_t[(int)(k * nq)];
        float* D_sq8 = new float[(int)(k * nq)];

        // Parameterization on k factor while doing search for index refinement
        float k_factor = 3;
        faiss::IndexRefineSearchParameters* params =
                new faiss::IndexRefineSearchParameters();
        params->k_factor = k_factor;

        // Perform index search using different index refinement
        index_fp16->search(nq, xq, k, D_fp16, I_fp16, params);
        index_sq8->search(nq, xq, k, D_sq8, I_sq8, params);

        printf("I_fp16=\n");
        for (int i = nq - 5; i < nq; i++) {
            for (int j = 0; j < k; j++) {
                printf("%5zd ", I_fp16[i * k + j]);
            }
            printf("\n");
        }

        printf("I_sq8=\n");
        for (int i = nq - 5; i < nq; i++) {
            for (int j = 0; j < k; j++) {
                printf("%5zd ", I_sq8[i * k + j]);
            }
            printf("\n");
        }

        delete[] I_fp16;
        delete[] D_fp16;
        delete[] I_sq8;
        delete[] D_sq8;
        delete params;

        delete index_fp16;
        delete index_sq8;
    }

    delete[] xb;
    delete[] xq;

    return 0;
}
