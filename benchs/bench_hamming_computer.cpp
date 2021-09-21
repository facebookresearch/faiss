/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <omp.h>
#include <cstdio>
#include <vector>

#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/hamming.h>
#include <faiss/utils/random.h>
#include <faiss/utils/utils.h>

using namespace faiss;

template <class T>
void hamming_cpt_test(
        int code_size,
        uint8_t* data1,
        uint8_t* data2,
        int n,
        int* rst) {
    T computer(data1, code_size);
    for (int i = 0; i < n; i++) {
        rst[i] = computer.hamming(data2);
        data2 += code_size;
    }
}

int main() {
    size_t n = 4 * 1000 * 1000;

    std::vector<size_t> code_size = {128, 256, 512, 1000};

    std::vector<uint8_t> x(n * code_size.back());
    byte_rand(x.data(), n, 12345);

    int nrun = 100;
    for (size_t cs : code_size) {
        printf("benchmark with code_size=%zd n=%zd nrun=%d\n", cs, n, nrun);

        double tot_t1 = 0, tot_t2 = 0, tot_t3 = 0;
#pragma omp parallel reduction(+ : tot_t1, tot_t2, tot_t3)
        {
            std::vector<int> rst_m4(n);
            std::vector<int> rst_m8(n);
            std::vector<int> rst_default(n);

#pragma omp for
            for (int run = 0; run < nrun; run++) {
                double t0, t1, t2, t3;
                t0 = getmillisecs();

                // new implem from Zilliz
                hamming_cpt_test<HammingComputerDefault>(
                        cs, x.data(), x.data(), n, rst_default.data());
                t1 = getmillisecs();

                // M8
                hamming_cpt_test<HammingComputerM8>(
                        cs, x.data(), x.data(), n, rst_m8.data());
                t2 = getmillisecs();

                // M4
                hamming_cpt_test<HammingComputerM4>(
                        cs, x.data(), x.data(), n, rst_m4.data());
                t3 = getmillisecs();

                tot_t1 += t1 - t0;
                tot_t2 += t2 - t1;
                tot_t3 += t3 - t2;
            }

            for (int i = 0; i < n; i++) {
                FAISS_THROW_IF_NOT_FMT(
                        (rst_m4[i] == rst_m8[i] && rst_m4[i] == rst_default[i]),
                        "wrong result i=%d, m4 %d m8 %d default %d",
                        i,
                        rst_m4[i],
                        rst_m8[i],
                        rst_default[i]);
            }
        }

        printf("Hamming_Dft  implem: %.3f ms\n", tot_t1 / nrun);
        printf("Hamming_M8   implem: %.3f ms\n", tot_t2 / nrun);
        printf("Hamming_M4   implem: %.3f ms\n", tot_t3 / nrun);
    }
    return 0;
}
