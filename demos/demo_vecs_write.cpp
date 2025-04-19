/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/utils/vecs_storage.h>
#include <sys/types.h>
#include <iostream>
#include <random>
#include <string>

void fillWithRandom(
        std::mt19937& rng,
        std::uniform_real_distribution<>& distrib,
        float* array,
        int size) {
    for (int i = 0; i < size; i++) {
        array[i] = distrib(rng) * 1000.;
    }
}

int main(int argc, char** argv) {
    if (argc <= 4) {
        std::cout << "params: <fileName> <numVecs> <dim> <type>" << std::endl;
        return 1;
    }

    std::string fileName;
    size_t numVecs;
    int dim, type;

    fileName = argv[1];
    numVecs = std::stoul(argv[2]);
    dim = std::stoi(argv[3]);
    type = std::stoi(argv[4]);

    if (type == 0) {
        float* vecsToWrite = new float[numVecs * dim];
        for (size_t i = 0; i < numVecs; i++) {
            for (size_t j = 0; j < dim; j++) {
                vecsToWrite[i * dim + j] = j;
            }
        }
        faiss::bvecs_write(fileName.c_str(), numVecs, dim, vecsToWrite);
        delete vecsToWrite;
    } else if (type == 1) {
        int* vecsToWrite = new int[numVecs * dim];
        for (size_t i = 0; i < numVecs; i++) {
            for (size_t j = 0; j < dim; j++) {
                vecsToWrite[i * dim + j] = j;
            }
        }
        faiss::ivecs_write(fileName.c_str(), numVecs, dim, vecsToWrite);
        delete vecsToWrite;
    } else {
        std::mt19937 rng;
        std::uniform_real_distribution<> distrib;

        float* vecsToWrite = new float[numVecs * dim];
        fillWithRandom(rng, distrib, vecsToWrite, numVecs * dim);
        faiss::fvecs_write(fileName.c_str(), numVecs, dim, vecsToWrite);
        delete vecsToWrite;
    }

    return 0;
}
