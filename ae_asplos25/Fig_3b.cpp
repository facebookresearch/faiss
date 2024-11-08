/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdio>
#include <cstdlib>
#include <random>
#include <chrono>
#include <iostream>
#include <omp.h>

#include <faiss/IndexFlat.h>

// 64-bit int
using idx_t = faiss::idx_t;


int main() {

    idx_t d = 768;      // dimension
    idx_t nb = 32552082;
	float raw_corpus_gb = (4.0* nb  * d) / 1000000000; //Use float32 for roofline. Done because no good float16 support at the moment.
	float adj_corpus_gb = (2.0* nb  * d) / 1000000000; //Use float16 for all comparisons

    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;

    float* xb = new float[d * nb];

    for (int i = 0; i < nb; i++) {
        for (int j = 0; j < d; j++)
            xb[d * i + j] = distrib(rng);
        xb[d * i] += i / 1000.;
    }


    faiss::IndexFlatIP index(d); 
    index.add(nb, xb); 
    printf("ntotal = %zd\n", index.ntotal);

    int k = 32;


	int cfgs[1] = {1};


	for (int run = 0; run < 3; run++){
		printf("------------------- Run %d -------------------\n", run);
		printf("nq = %d\n", cfgs[i]);
		int nq = cfgs[i];
		float gflop = (nb * nq * (2.0 * d - 1)) / 1000000000; 
    	float* xq = new float[d * nq];
		for (int i = 0; i < nq; i++) {
			for (int j = 0; j < d; j++)
				xq[d * i + j] = distrib(rng);
			xq[d * i] += i / 1000.;
		}

        idx_t* I = new idx_t[k * nq];
        float* D = new float[k * nq];

		auto start = std::chrono::high_resolution_clock::now();
        index.search(nq, xq, k, D, I);
		auto end = std::chrono::high_resolution_clock::now();


		std::chrono::duration<double> elapsed = end - start;
		printf("Elapsed time: %f s\n", elapsed.count());

		printf("Corpus size: %f GB\n", adj_corpus_gb);
		printf("GFLOP: %f\n", gflop);
		printf("GFLOP/BYTE: %f\n", gflop / raw_corpus_gb);
		printf("GFLOP/SEC: %f\n", gflop / elapsed.count());
		printf("BW: %f GB/s\n", raw_corpus_gb / elapsed.count());

		printf("");
        delete[] I;
        delete[] D;
		delete[] xq;
    }



    delete[] xb;

    return 0;
}
