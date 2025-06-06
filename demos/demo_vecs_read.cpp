/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/utils/vecs_storage.h>
#include <sys/types.h>
#include <iostream>
#include <string>

int main(int argc, char** argv) {
    if (argc <= 5) {
        std::cout << "params: <fileName> <numVecs> <numOffset> <dim> <type>"
                  << std::endl;
        return 1;
    }

    std::string fileName;
    size_t numVecs, numOffset;
    int dim, type;

    fileName = argv[1];
    numVecs = std::stoul(argv[2]);
    numOffset = std::stoi(argv[3]);
    dim = std::stoi(argv[4]);
    type = std::stoi(argv[5]);

    if (type == 0) {
        int readedDim;
        float* vecs = faiss::bvecs_read(
                fileName.c_str(), numVecs, numOffset, &readedDim);
        std::cout << readedDim << std::endl;
        for (size_t i = 0; i < numVecs; i++) {
            size_t j = 0;
            for (; j < dim - 1; j++) {
                std::cout << vecs[i * dim + j] << " ";
            }
            std::cout << vecs[i * dim + j] << std::endl;
        }
        delete vecs;
    } else if (type == 1) {
        int readedDim;
        int* vecs = faiss::ivecs_read(
                fileName.c_str(), numVecs, numOffset, &readedDim);
        std::cout << readedDim << std::endl;
        for (size_t i = 0; i < numVecs; i++) {
            size_t j = 0;
            for (; j < dim - 1; j++) {
                std::cout << vecs[i * dim + j] << " ";
            }
            std::cout << vecs[i * dim + j] << std::endl;
        }
        delete vecs;
    } else {
        int readedDim;
        float* vecs = faiss::fvecs_read(
                fileName.c_str(), numVecs, numOffset, &readedDim);
        std::cout << readedDim << std::endl;
        for (size_t i = 0; i < numVecs; i++) {
            size_t j = 0;
            for (; j < dim - 1; j++) {
                std::cout << vecs[i * dim + j] << " ";
            }
            std::cout << vecs[i * dim + j] << std::endl;
        }
        delete vecs;
    }

    return 0;
}
