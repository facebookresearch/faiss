/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <omp.h>
#include <cstdio>

#include <faiss/impl/FaissAssert.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/random.h>
#include <faiss/utils/utils.h>

using namespace faiss;

void addn_default(
        size_t n,
        size_t k,
        const float* x,
        int64_t* heap_ids,
        float* heap_val) {
    for (size_t i = 0; i < k; i++) {
        minheap_push(i + 1, heap_val, heap_ids, x[i], i);
    }

    for (size_t i = k; i < n; i++) {
        if (x[i] > heap_val[0]) {
            minheap_pop(k, heap_val, heap_ids);
            minheap_push(k, heap_val, heap_ids, x[i], i);
        }
    }

    minheap_reorder(k, heap_val, heap_ids);
}

void addn_replace(
        size_t n,
        size_t k,
        const float* x,
        int64_t* heap_ids,
        float* heap_val) {
    for (size_t i = 0; i < k; i++) {
        minheap_push(i + 1, heap_val, heap_ids, x[i], i);
    }

    for (size_t i = k; i < n; i++) {
        if (x[i] > heap_val[0]) {
            minheap_replace_top(k, heap_val, heap_ids, x[i], i);
        }
    }

    minheap_reorder(k, heap_val, heap_ids);
}

void addn_func(
        size_t n,
        size_t k,
        const float* x,
        int64_t* heap_ids,
        float* heap_val) {
    minheap_heapify(k, heap_val, heap_ids);

    minheap_addn(k, heap_val, heap_ids, x, nullptr, n);

    minheap_reorder(k, heap_val, heap_ids);
}

int main() {
    size_t n = 10 * 1000 * 1000;

    std::vector<size_t> ks({20, 50, 100, 200, 500, 1000, 2000, 5000});

    std::vector<float> x(n);
    float_randn(x.data(), n, 12345);

    int nrun = 100;
    for (size_t k : ks) {
        printf("benchmark with k=%zd n=%zd nrun=%d\n", k, n, nrun);
        FAISS_THROW_IF_NOT(k < n);

        double tot_t1 = 0, tot_t2 = 0, tot_t3 = 0;
#pragma omp parallel reduction(+ : tot_t1, tot_t2, tot_t3)
        {
            std::vector<float> heap_dis(k);
            std::vector<float> heap_dis_2(k);
            std::vector<float> heap_dis_3(k);

            std::vector<int64_t> heap_ids(k);
            std::vector<int64_t> heap_ids_2(k);
            std::vector<int64_t> heap_ids_3(k);

#pragma omp for
            for (int run = 0; run < nrun; run++) {
                double t0, t1, t2, t3;

                t0 = getmillisecs();

                // default implem
                addn_default(n, k, x.data(), heap_ids.data(), heap_dis.data());
                t1 = getmillisecs();

                // new implem from Zilliz
                addn_replace(
                        n, k, x.data(), heap_ids_2.data(), heap_dis_2.data());
                t2 = getmillisecs();

                // with addn
                addn_func(n, k, x.data(), heap_ids_3.data(), heap_dis_3.data());
                t3 = getmillisecs();

                tot_t1 += t1 - t0;
                tot_t2 += t2 - t1;
                tot_t3 += t3 - t2;
            }

            for (size_t i = 0; i < k; i++) {
                FAISS_THROW_IF_NOT_FMT(
                        heap_ids[i] == heap_ids_2[i],
                        "i=%ld (%ld, %g) != (%ld, %g)",
                        i,
                        size_t(heap_ids[i]),
                        heap_dis[i],
                        size_t(heap_ids_2[i]),
                        heap_dis_2[i]);
                FAISS_THROW_IF_NOT(heap_dis[i] == heap_dis_2[i]);
            }

            for (size_t i = 0; i < k; i++) {
                FAISS_THROW_IF_NOT(heap_ids[i] == heap_ids_3[i]);
                FAISS_THROW_IF_NOT(heap_dis[i] == heap_dis_3[i]);
            }
        }
        printf("default implem: %.3f ms\n", tot_t1 / nrun);
        printf("replace implem: %.3f ms\n", tot_t2 / nrun);
        printf("addn    implem: %.3f ms\n", tot_t3 / nrun);
    }
    return 0;
}
