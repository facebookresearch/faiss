/*
 * Example: IndexSVSUncompressed search on BigANN with QPS metric
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 * Licensed under the MIT license.
 */
/*
 * Copyright 2025 Intel Corporation
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <faiss/IndexSVSUncompressed.h>
#include <faiss/index_io.h>
#include <cassert>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <vector>

// Helper to read float32 fvecs file
void read_fvecs(
        const char* filename,
        std::vector<float>& data,
        size_t& d,
        size_t& n) {
    FILE* f = fopen(filename, "rb");
    assert(f);
    int dim;
    fread(&dim, sizeof(int), 1, f);
    fseek(f, 0, SEEK_SET);
    std::vector<int> buf(1);
    size_t filesize = 0;
    while (fread(buf.data(), sizeof(int), 1, f) == 1) {
        fseek(f, dim * sizeof(float), SEEK_CUR);
        filesize++;
    }
    n = filesize;
    d = dim;
    data.resize(n * d);
    fseek(f, 0, SEEK_SET);
    for (size_t i = 0; i < n; i++) {
        fread(buf.data(), sizeof(int), 1, f);
        fread(&data[i * d], sizeof(float), d, f);
    }
    fclose(f);
}

int main() {
    // Paths to deep-10m f32 dataset files
    const char* db_file = "/export/data/datasets/deep/deep_10m_f32.fvecs";
    const char* query_file = "/export/data/datasets/deep/deep_queries.fvecs";
    size_t d, nb, nq;
    std::vector<float> xb, xq;

    std::cout << "Loading database vectors..." << std::endl;
    read_fvecs(db_file, xb, d, nb);
    std::cout << "Loaded " << nb << " vectors of dimension " << d << std::endl;

    std::cout << "Loading query vectors..." << std::endl;
    size_t d2;
    read_fvecs(query_file, xq, d2, nq);
    assert(d == d2);
    std::cout << "Loaded " << nq << " query vectors." << std::endl;

    int k = 10; // number of nearest neighbors
    faiss::IndexSVSUncompressed index(d);
    index.num_threads = 72;
    std::cout << "Adding database vectors to index..." << std::endl;
    auto t_build0 = std::chrono::high_resolution_clock::now();
    index.add(nb, xb.data());
    auto t_build1 = std::chrono::high_resolution_clock::now();
    double build_time =
            std::chrono::duration_cast<std::chrono::duration<double>>(
                    t_build1 - t_build0)
                    .count();
    std::cout << "Index build time (add): " << build_time << " seconds"
              << std::endl;

    std::cout << "Searching queries and measuring QPS..." << std::endl;
    std::vector<faiss::idx_t> I(k * nq);
    std::vector<float> D(k * nq);
    auto t0 = std::chrono::high_resolution_clock::now();
    index.search(nq, xq.data(), k, D.data(), I.data());
    auto t1 = std::chrono::high_resolution_clock::now();
    double elapsed =
            std::chrono::duration_cast<std::chrono::duration<double>>(t1 - t0)
                    .count();
    double qps = nq / elapsed;
    std::cout << "Queries per second: " << qps << std::endl;

    // Print results for last 5 queries
    // std::cout << "Results for last 5 queries:" << std::endl;
    // for (size_t i = nq - 5; i < nq; i++) {
    //     std::cout << "Query " << i << ": ";
    //     for (int j = 0; j < k; j++) {
    //         std::cout << I[i * k + j] << " ";
    //     }
    //     std::cout << std::endl;
    // }

    return 0;
}
