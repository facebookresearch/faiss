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

	std::mt19937 rng;
	std::uniform_real_distribution<> distrib;


	int k = 32;


	int cfgs[2] = {1,16};
	idx_t nb_50 = 32552083;

	idx_t total_sizes[3] = {1,5,10, 50,200, 512, 1024, 2048};

	for(int batch_size_idx = 0; batch_size_idx < 2; ++batch_size_idx){

		int nq = cfgs[batch_size_idx];
		printf("------------------- %ld Queries -------------------\n", nq);
		faiss::IndexFlatIP index(d); 
		for(int size_idx = 0; size_idx< 8; ++size_idx){
			printf("|||||||||--------- %ld GB Corpus ---------|||||||||\n", total_sizes[size_idx]);
			idx_t nb = nb_50/50 * total_sizes[size_idx]- index.ntotal; //New vectors
			float* xb = new float[d * nb];

			for (int i = 0; i < nb; i++) {
				for (int j = 0; j < d; j++)
					xb[d * i + j] = distrib(rng);
				xb[d * i] += i / 1000.;
			}


			index.add(nb, xb); 
			delete[] xb;

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
			float adj_corpus_gb = (2.0* index.ntotal  * d) / 1000000000; //Use float16 for all comparisons
			printf("Corpus size: %f GB\n", adj_corpus_gb);
			printf("Elapsed time: %f s\n", elapsed.count());


			printf("");
			delete[] I;
			delete[] D;
			delete[] xq;
		}
	}




	return 0;
}
