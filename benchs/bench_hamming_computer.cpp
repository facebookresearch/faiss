/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdio>
#include <vector>

#include <cinttypes>

#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/hamming.h>
#include <faiss/utils/random.h>
#include <faiss/utils/utils.h>

using namespace faiss;

// These implementations are currently slower than HammingComputerDefault so
// they are not in the main faiss anymore.
struct HammingComputerM8 {
    const uint64_t* a;
    int n;

    HammingComputerM8() = default;

    HammingComputerM8(const uint8_t* a8, int code_size) {
        set(a8, code_size);
    }

    void set(const uint8_t* a8, int code_size) {
        assert(code_size % 8 == 0);
        a = (uint64_t*)a8;
        n = code_size / 8;
    }

    int hamming(const uint8_t* b8) const {
        const uint64_t* b = (uint64_t*)b8;
        int accu = 0;
        for (int i = 0; i < n; i++)
            accu += popcount64(a[i] ^ b[i]);
        return accu;
    }

    inline int get_code_size() const {
        return n * 8;
    }
};

struct HammingComputerM4 {
    const uint32_t* a;
    int n;

    HammingComputerM4() = default;

    HammingComputerM4(const uint8_t* a4, int code_size) {
        set(a4, code_size);
    }

    void set(const uint8_t* a4, int code_size) {
        assert(code_size % 4 == 0);
        a = (uint32_t*)a4;
        n = code_size / 4;
    }

    int hamming(const uint8_t* b8) const {
        const uint32_t* b = (uint32_t*)b8;
        int accu = 0;
        for (int i = 0; i < n; i++)
            accu += popcount64(a[i] ^ b[i]);
        return accu;
    }

    inline int get_code_size() const {
        return n * 4;
    }
};

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

template <int CODE_SIZE_IN_BITS>
void hamming_func_test(
        const uint8_t* const x1,
        const uint8_t* const x2,
        const size_t n1,
        const size_t n2,
        uint64_t& sumv,
        uint64_t& xorv) {
    constexpr size_t CODE_SIZE_IN_BYTES = CODE_SIZE_IN_BITS / 8;

    double t0 = faiss::getmillisecs();

    uint64_t sumx = 0;
    uint64_t xorx = 0;

    const size_t nruns = 10;
    for (size_t irun = 0; irun < 10; irun++) {
#pragma omp parallel reduction(+ : sumx, xorx)
        {
#pragma omp for
            for (size_t i = 0; i < n1; i++) {
                uint64_t local_sum = 0;
                uint64_t local_xor = 0;

                const uint64_t* data1_ptr =
                        (const uint64_t*)(x1 + i * CODE_SIZE_IN_BYTES);

                for (size_t j = 0; j < n2; j++) {
                    const uint64_t* data2_ptr =
                            (const uint64_t*)(x2 + j * CODE_SIZE_IN_BYTES);

                    uint64_t code = faiss::hamming<CODE_SIZE_IN_BITS>(
                            data1_ptr, data2_ptr);
                    local_sum += code;
                    local_xor ^= code;
                }

                sumx += local_sum;
                xorx ^= local_xor;
            }
        }
    }

    sumv = sumx;
    xorv = xorx;

    double t1 = faiss::getmillisecs();
    printf("hamming<%d>: %.3f msec, %" PRIX64 ", %" PRIX64 "\n",
           CODE_SIZE_IN_BITS,
           (t1 - t0) / nruns,
           sumx,
           xorx);
}

template <typename HammingComputerT, int CODE_SIZE_IN_BITS>
void hamming_computer_test(
        const uint8_t* const x1,
        const uint8_t* const x2,
        const size_t n1,
        const size_t n2,
        uint64_t& sumv,
        uint64_t& xorv) {
    constexpr size_t CODE_SIZE_IN_BYTES = CODE_SIZE_IN_BITS / 8;

    double t0 = faiss::getmillisecs();

    uint64_t sumx = 0;
    uint64_t xorx = 0;

    const size_t nruns = 10;
    for (size_t irun = 0; irun < nruns; irun++) {
        sumx = 0;
        xorx = 0;

#pragma omp parallel reduction(+ : sumx, xorx)
        {
#pragma omp for
            for (size_t i = 0; i < n1; i++) {
                uint64_t local_sum = 0;
                uint64_t local_xor = 0;

                const uint8_t* data1_ptr = x1 + i * CODE_SIZE_IN_BYTES;
                HammingComputerT hc(data1_ptr, CODE_SIZE_IN_BYTES);

                for (size_t j = 0; j < n2; j++) {
                    const uint8_t* data2_ptr = x2 + j * CODE_SIZE_IN_BYTES;
                    uint64_t code = hc.hamming(data2_ptr);
                    local_sum += code;
                    local_xor ^= code;
                }

                sumx += local_sum;
                xorx ^= local_xor;
            }
        }
    }

    sumv = sumx;
    xorv = xorx;

    double t1 = faiss::getmillisecs();
    printf("HammingComputer<%zd>: %.3f msec, %" PRIX64 ", %" PRIX64 "\n",
           CODE_SIZE_IN_BYTES,
           (t1 - t0) / nruns,
           sumx,
           xorx);
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

    // evaluate various hamming<>() function calls
    const size_t MAX_HAMMING_FUNC_CODE_SIZE = 512;

    const size_t n1 = 65536;
    const size_t n2 = 16384;

    std::vector<uint8_t> x1(n1 * MAX_HAMMING_FUNC_CODE_SIZE / 8);
    std::vector<uint8_t> x2(n2 * MAX_HAMMING_FUNC_CODE_SIZE / 8);
    byte_rand(x1.data(), x1.size(), 12345);
    byte_rand(x2.data(), x2.size(), 23456);

    // These two values serve as a kind of CRC.
    uint64_t sumx = 0;
    uint64_t xorx = 0;
    hamming_func_test<64>(x1.data(), x2.data(), n1, n2, sumx, xorx);
    hamming_func_test<128>(x1.data(), x2.data(), n1, n2, sumx, xorx);
    hamming_func_test<256>(x1.data(), x2.data(), n1, n2, sumx, xorx);
    hamming_func_test<384>(x1.data(), x2.data(), n1, n2, sumx, xorx);
    hamming_func_test<512>(x1.data(), x2.data(), n1, n2, sumx, xorx);

    // evaluate various HammingComputerXX
    hamming_computer_test<faiss::HammingComputer4, 32>(
            x1.data(), x2.data(), n1, n2, sumx, xorx);
    hamming_computer_test<faiss::HammingComputer8, 64>(
            x1.data(), x2.data(), n1, n2, sumx, xorx);
    hamming_computer_test<faiss::HammingComputer16, 128>(
            x1.data(), x2.data(), n1, n2, sumx, xorx);
    hamming_computer_test<faiss::HammingComputer20, 160>(
            x1.data(), x2.data(), n1, n2, sumx, xorx);
    hamming_computer_test<faiss::HammingComputer32, 256>(
            x1.data(), x2.data(), n1, n2, sumx, xorx);
    hamming_computer_test<faiss::HammingComputer64, 512>(
            x1.data(), x2.data(), n1, n2, sumx, xorx);

    // evaluate various GenHammingDistanceComputerXX
    hamming_computer_test<faiss::GenHammingComputer8, 64>(
            x1.data(), x2.data(), n1, n2, sumx, xorx);
    hamming_computer_test<faiss::GenHammingComputer16, 128>(
            x1.data(), x2.data(), n1, n2, sumx, xorx);
    hamming_computer_test<faiss::GenHammingComputer32, 256>(
            x1.data(), x2.data(), n1, n2, sumx, xorx);

    hamming_computer_test<faiss::GenHammingComputerM8, 64>(
            x1.data(), x2.data(), n1, n2, sumx, xorx);
    hamming_computer_test<faiss::GenHammingComputerM8, 128>(
            x1.data(), x2.data(), n1, n2, sumx, xorx);
    hamming_computer_test<faiss::GenHammingComputerM8, 256>(
            x1.data(), x2.data(), n1, n2, sumx, xorx);
    hamming_computer_test<faiss::GenHammingComputerM8, 512>(
            x1.data(), x2.data(), n1, n2, sumx, xorx);

    return 0;
}
