/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdio>
#include <memory>
#include <unordered_set>
#include <vector>

#include <gtest/gtest.h>

#include <faiss/AutoTune.h>
#include <faiss/IndexIVFPQFastScan.h>
#include <faiss/index_factory.h>
#include <faiss/utils/Heap.h>
#include <faiss/utils/random.h>

using namespace faiss;

namespace {

// Make sure that the InvertedListScanner::distance_to_code computes distances
// in the same way as what is used with the actual search function of an
// IndexIVF
void test_search_and_encode(const char* factory_string, MetricType metric) {
    int d = 64;
    int nb1 = 900;
    int nb2 = 100;
    int nq = 100;
    int k = 10;

    // generate smooth random data for training + database
    std::vector<float> xb((nb1 + nb2) * d);
    rand_smooth_vectors((nb1 + nb2), d, xb.data(), 1234);

    // build the index via factory
    std::unique_ptr<IndexIVF> index(
            dynamic_cast<IndexIVF*>(index_factory(d, factory_string, metric)));

    index->train(nb1, xb.data());
    index->add(nb1, xb.data());

    index->nprobe = 8;

    std::vector<float> xq(nq * d);
    rand_smooth_vectors(nq, d, xq.data(), 5678);

    // search
    std::vector<float> D0(nq * k);
    std::vector<idx_t> I0(nq * k);

    index->search(nq, xq.data(), k, D0.data(), I0.data());

    const float* xb2 = xb.data() + d * nb1;
    std::vector<idx_t> list_nos(nb2);
    std::vector<float> dis(nb2);
    index->quantizer->search(nb2, xb2, 1, dis.data(), list_nos.data());
    std::vector<uint8_t> codes(nb2 * index->code_size);
    index->encode_vectors(nb2, xb2, list_nos.data(), codes.data());

    // encode into flattened inverted lists
    std::vector<std::vector<uint8_t>> extra_codes(index->nlist);
    std::vector<std::vector<idx_t>> extra_ids(index->nlist);

    for (size_t i = 0; i < nb2; i++) {
        idx_t list_no = list_nos[i];
        extra_ids[list_no].push_back(i + nb1);
        extra_codes[list_no].insert(
                extra_codes[list_no].end(),
                codes.data() + i * index->code_size,
                codes.data() + (i + 1) * index->code_size);
    }

    // search these extra distances manually and combine with D0 and I0
    std::unique_ptr<InvertedListScanner> scanner(
            index->get_InvertedListScanner(false, nullptr, nullptr));

    // for each query, find its nprobe nearest lists
    int nprobe = index->nprobe;
    std::vector<float> coarse_dis(nq * nprobe);
    std::vector<idx_t> coarse_ids(nq * nprobe);
    index->quantizer->search(
            nq, xq.data(), nprobe, coarse_dis.data(), coarse_ids.data());

    bool is_similarity = is_similarity_metric(metric);

    for (idx_t q = 0; q < nq; q++) {
        float* D0q = D0.data() + k * q;
        idx_t* I0q = I0.data() + k * q;

        // build a heap from the sorted search results
        // L2: max-heap (worst = largest distance at top)
        // IP: min-heap (worst = smallest similarity at top)
        if (is_similarity) {
            minheap_heapify(k, D0q, I0q, D0q, I0q, k);
        } else {
            maxheap_heapify(k, D0q, I0q, D0q, I0q, k);
        }

        const float* query = xq.data() + q * d;
        scanner->set_query(query);

        // loop over probed lists
        for (int p = 0; p < nprobe; p++) {
            idx_t list_no = coarse_ids[q * nprobe + p];
            if (list_no < 0) {
                continue;
            }
            scanner->set_list(list_no, coarse_dis[q * nprobe + p]);

            // scan extra codes for this list
            size_t n_extra = extra_ids[list_no].size();
            for (size_t j = 0; j < n_extra; j++) {
                float dis = scanner->distance_to_code(
                        extra_codes[list_no].data() + j * index->code_size);

                idx_t id = extra_ids[list_no][j];
                if (is_similarity) {
                    if (dis > D0q[0]) {
                        minheap_replace_top(k, D0q, I0q, dis, id);
                    }
                } else {
                    if (dis < D0q[0]) {
                        maxheap_replace_top(k, D0q, I0q, dis, id);
                    }
                }
            }
        }

        // reorder from heap order to sorted order
        if (is_similarity) {
            minheap_reorder(k, D0q, I0q);
        } else {
            maxheap_reorder(k, D0q, I0q);
        }
    }

    // verification: add xb2 to the index, search, and compare
    index->add(nb2, xb2);

    std::vector<float> D1(nq * k);
    std::vector<idx_t> I1(nq * k);
    index->search(nq, xq.data(), k, D1.data(), I1.data());

    int ndiff_total = 0;
    for (int q = 0; q < nq; q++) {
        const idx_t* I0q = I0.data() + k * q;
        const idx_t* I1q = I1.data() + k * q;
        std::unordered_set<idx_t> S0(I0q, I0q + k);
        std::unordered_set<idx_t> S1(I1q, I1q + k);
        int ndiff = 0;
        for (idx_t id : S0) {
            if (S1.find(id) == S1.end()) {
                ndiff++;
            }
        }
        ndiff_total += ndiff;
    }
    // accept a small amount of discrepancy, because the FastScan look-up tables
    // are quantized to int8 so not as accurate.
    printf("total ndiff=%d / %d\n", ndiff_total, nq * k);
    EXPECT_LE(ndiff_total, nq * k / 200);
}

} // namespace

TEST(IVFPQFastScan, SearchAndEncodeIVFPQ_L2) {
    test_search_and_encode("IVF32,PQ16x4", METRIC_L2);
}

TEST(IVFPQFastScan, SearchAndEncodeIVFPQFastScan_L2) {
    test_search_and_encode("IVF32,PQ16x4fs", METRIC_L2);
}

TEST(IVFPQFastScan, SearchAndEncodeIVFPQ_IP) {
    test_search_and_encode("IVF32,PQ16x4", METRIC_INNER_PRODUCT);
}

TEST(IVFPQFastScan, SearchAndEncodeIVFPQFastScan_IP) {
    test_search_and_encode("IVF32,PQ16x4fs", METRIC_INNER_PRODUCT);
}
