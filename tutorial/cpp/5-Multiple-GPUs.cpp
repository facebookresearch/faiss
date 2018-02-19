/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdio>
#include <cstdlib>

#include <faiss/IndexFlat.h>
#include <faiss/gpu/GpuAutoTune.h>
#include <faiss/gpu/GpuIndexFlat.h>
#include <faiss/gpu/StandardGpuResources.h>
#include <faiss/gpu/utils/DeviceUtils.h>


int main() {
    int d = 64;                            // dimension
    int nb = 100000;                       // database size
    int nq = 10000;                        // nb of queries

    float *xb = new float[d * nb];
    float *xq = new float[d * nq];

    for(int i = 0; i < nb; i++) {
        for(int j = 0; j < d; j++)
            xb[d * i + j] = drand48();
        xb[d * i] += i / 1000.;
    }

    for(int i = 0; i < nq; i++) {
        for(int j = 0; j < d; j++)
            xq[d * i + j] = drand48();
        xq[d * i] += i / 1000.;
    }

    int ngpus = faiss::gpu::getNumDevices();

    std::vector<faiss::gpu::GpuResources*> res;
    std::vector<int> devs;
    for(int i = 0; i < ngpus; i++) {
        res.push_back(new faiss::gpu::StandardGpuResources);
        devs.push_back(i);
    }

    faiss::Index *index =
        faiss::gpu::index_cpu_to_gpu_multiple(  // call constructor
            res,
            devs,
            new faiss::IndexFlatL2(d)
        );

    printf("is_trained = %s\n", index->is_trained ? "true" : "false");
    index->add(nb, xb);                     // add vectors to the index
    printf("ntotal = %ld\n", index->ntotal);

    int k = 4;

    {       // sanity check: search 5 first vectors of xb
        long *I = new long[k * 5];
        float *D = new float[k * 5];

        index->search(5, xb, k, D, I);

        // print results
        printf("I=\n");
        for(int i = 0; i < 5; i++) {
            for(int j = 0; j < k; j++)
                printf("%5ld ", I[i * k + j]);
            printf("\n");
        }

        printf("D=\n");
        for(int i = 0; i < 5; i++) {
            for(int j = 0; j < k; j++)
                printf("%7g ", D[i * k + j]);
            printf("\n");
        }

        delete [] I;
        delete [] D;
    }


    {       // search xq
        long *I = new long[k * nq];
        float *D = new float[k * nq];

        index->search(nq, xq, k, D, I);

        // print results
        printf("I (5 first results)=\n");
        for(int i = 0; i < 5; i++) {
            for(int j = 0; j < k; j++)
                printf("%5ld ", I[i * k + j]);
            printf("\n");
        }

        printf("I (5 last results)=\n");
        for(int i = nq - 5; i < nq; i++) {
            for(int j = 0; j < k; j++)
                printf("%5ld ", I[i * k + j]);
            printf("\n");
        }

        delete [] I;
        delete [] D;
    }

    delete index;

    for(int i = 0; i < ngpus; i++) {
        delete res[i];
    }

    delete [] xb;
    delete [] xq;

    return 0;
}
