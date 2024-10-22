/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c -*-

#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include "AutoTune_c.h"
#include "IndexFlat_c.h"
#include "Index_c.h"
#include "clone_index_c.h"
#include "error_c.h"
#include "impl/AuxIndexStructures_c.h"
#include "index_factory_c.h"
#include "index_io_c.h"

#define FAISS_TRY(C)                                       \
    {                                                      \
        if (C) {                                           \
            fprintf(stderr, "%s", faiss_get_last_error()); \
            exit(-1);                                      \
        }                                                  \
    }

double drand() {
    return (double)rand() / (double)RAND_MAX;
}

int main() {
    time_t seed = time(NULL);
    srand(seed);
    printf("Generating some data...\n");
    int d = 128;     // dimension
    int nb = 100000; // database size
    int nq = 10000;  // nb of queries
    float* xb = malloc(d * nb * sizeof(float));
    float* xq = malloc(d * nq * sizeof(float));

    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < d; j++)
            xb[d * i + j] = drand();
        xb[d * i] += i / 1000.;
    }
    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < d; j++)
            xq[d * i + j] = drand();
        xq[d * i] += i / 1000.;
    }

    printf("Building an index...\n");

    FaissIndex* index = NULL;
    FAISS_TRY(faiss_index_factory(
            &index, d, "Flat", METRIC_L2)); // use factory to create index
    printf("is_trained = %s\n",
           faiss_Index_is_trained(index) ? "true" : "false");
    FAISS_TRY(faiss_Index_add(index, nb, xb)); // add vectors to the index
    printf("ntotal = %lld\n", faiss_Index_ntotal(index));

    printf("Searching...\n");
    int k = 5;

    { // sanity check: search 5 first vectors of xb
        idx_t* I = malloc(k * 5 * sizeof(idx_t));
        float* D = malloc(k * 5 * sizeof(float));
        FAISS_TRY(faiss_Index_search(index, 5, xb, k, D, I));
        printf("I=\n");
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < k; j++)
                printf("%5lld (d=%2.3f)  ", I[i * k + j], D[i * k + j]);
            printf("\n");
        }
        free(I);
        free(D);
    }
    { // search xq
        idx_t* I = malloc(k * nq * sizeof(idx_t));
        float* D = malloc(k * nq * sizeof(float));
        FAISS_TRY(faiss_Index_search(index, nq, xq, k, D, I));
        printf("I=\n");
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < k; j++)
                printf("%5lld (d=%2.3f)  ", I[i * k + j], D[i * k + j]);
            printf("\n");
        }
        free(I);
        free(D);
    }
    { // search xb first 5 but search parameters of id range [50, 100]
        idx_t* I = malloc(k * nq * sizeof(idx_t));
        float* D = malloc(k * nq * sizeof(float));
        FaissIDSelectorRange* sel = NULL;
        FAISS_TRY(faiss_IDSelectorRange_new(&sel, 50, 100));
        FaissSearchParameters* params = NULL;
        FAISS_TRY(faiss_SearchParameters_new(&params, sel));
        FAISS_TRY(
                faiss_Index_search_with_params(index, nq, xq, k, params, D, I));
        printf("Searching w/ IDSelectorRange [50,100]\n");
        printf("I=\n");
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < k; j++)
                printf("%5lld (d=%2.3f)  ", I[i * k + j], D[i * k + j]);
            printf("\n");
        }
        free(I);
        free(D);
        faiss_SearchParameters_free(params);
        faiss_IDSelectorRange_free(sel);
    }

    { // search xb first 5 but search parameters of id range [20,40] OR
      // [45,60]
        idx_t* I = malloc(k * nq * sizeof(idx_t));
        float* D = malloc(k * nq * sizeof(float));
        FaissIDSelectorRange* lhs_sel = NULL;
        FAISS_TRY(faiss_IDSelectorRange_new(&lhs_sel, 20, 40));
        FaissIDSelectorRange* rhs_sel = NULL;
        FAISS_TRY(faiss_IDSelectorRange_new(&rhs_sel, 45, 60));
        FaissIDSelectorOr* sel = NULL;
        FAISS_TRY(faiss_IDSelectorOr_new(&sel, lhs_sel, rhs_sel));
        FaissSearchParameters* params = NULL;
        FAISS_TRY(faiss_SearchParameters_new(&params, sel));
        FAISS_TRY(
                faiss_Index_search_with_params(index, nq, xq, k, params, D, I));
        printf("Searching w/ IDSelectorRange [20,40] OR [45,60] \n");
        printf("I=\n");
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < k; j++)
                printf("%5lld (d=%2.3f)  ", I[i * k + j], D[i * k + j]);
            printf("\n");
        }
        free(I);
        free(D);
        faiss_SearchParameters_free(params);
        faiss_IDSelectorRange_free(lhs_sel);
        faiss_IDSelectorRange_free(rhs_sel);
        faiss_IDSelector_free(sel);
    }
    { // search xb first 5 but search parameters of id range [20,40] AND
      // [15,35] = [20,35]
        idx_t* I = malloc(k * nq * sizeof(idx_t));
        float* D = malloc(k * nq * sizeof(float));
        FaissIDSelectorRange* lhs_sel = NULL;
        FAISS_TRY(faiss_IDSelectorRange_new(&lhs_sel, 20, 40));
        FaissIDSelectorRange* rhs_sel = NULL;
        FAISS_TRY(faiss_IDSelectorRange_new(&rhs_sel, 15, 35));
        FaissIDSelectorAnd* sel = NULL;
        FAISS_TRY(faiss_IDSelectorAnd_new(&sel, lhs_sel, rhs_sel));
        FaissSearchParameters* params = NULL;
        FAISS_TRY(faiss_SearchParameters_new(&params, sel));
        FAISS_TRY(
                faiss_Index_search_with_params(index, nq, xq, k, params, D, I));
        printf("Searching w/ IDSelectorRange [20,40] AND [15,35] = [20,35]\n");
        printf("I=\n");
        for (int i = 0; i < 5; i++) {
            for (int j = 0; j < k; j++)
                printf("%5lld (d=%2.3f)  ", I[i * k + j], D[i * k + j]);
            printf("\n");
        }
        free(I);
        free(D);
        faiss_SearchParameters_free(params);
        faiss_IDSelectorRange_free(lhs_sel);
        faiss_IDSelectorRange_free(rhs_sel);
        faiss_IDSelector_free(sel);
    }

    printf("Saving index to disk...\n");
    FAISS_TRY(faiss_write_index_fname(index, "example.index"));

    printf("Freeing index...\n");
    faiss_Index_free(index);
    printf("Done.\n");

    return 0;
}
