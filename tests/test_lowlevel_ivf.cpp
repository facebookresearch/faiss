/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdio>
#include <cstdlib>

#include <memory>
#include <random>
#include <thread>
#include <vector>

#include <gtest/gtest.h>

#include <faiss/AutoTune.h>
#include <faiss/IVFlib.h>
#include <faiss/IndexBinaryIVF.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFPQFastScan.h>
#include <faiss/IndexIVFRaBitQFastScan.h>
#include <faiss/IndexPreTransform.h>
#include <faiss/index_factory.h>
#include <faiss/utils/distances.h>

using namespace faiss;

namespace {

int d = 32;
size_t nt = 5000;
size_t nb = 1000;
size_t nq = 200;

int k = 10;

std::mt19937 rng;

std::vector<float> make_data(size_t n) {
    std::vector<float> database(n * d);
    std::uniform_real_distribution<> distrib;
    for (size_t i = 0; i < n * d; i++) {
        database[i] = distrib(rng);
    }
    return database;
}

std::unique_ptr<Index> make_trained_index(
        const char* index_type,
        MetricType metric_type) {
    auto index =
            std::unique_ptr<Index>(index_factory(d, index_type, metric_type));
    auto xt = make_data(nt);
    index->train(nt, xt.data());
    ParameterSpace().set_index_parameter(index.get(), "nprobe", 4);
    return index;
}

std::pair<std::vector<idx_t>, std::vector<float>> search_index(
        Index* index,
        const float* xq) {
    std::vector<idx_t> I(k * nq);
    std::vector<float> D(k * nq);
    index->search(nq, xq, k, D.data(), I.data());
    return {I, D};
}

/*************************************************************
 * Test functions for a given index type
 *************************************************************/

void test_get_InvertedListScanner(
        IndexIVF* index_ivf,
        const IndexPreTransform* index_pt,
        std::vector<uint8_t> codes,
        std::unique_ptr<InvertedListScanner> scanner,
        std::vector<float> xq,
        std::vector<idx_t> ref_I,
        std::vector<float> ref_D,
        MetricType metric,
        bool distance_to_code_supported = true,
        float accuracy_requirement = 1.0);

void test_lowlevel_access(const char* index_key, MetricType metric) {
    std::unique_ptr<Index> index = make_trained_index(index_key, metric);

    auto xb = make_data(nb);
    index->add(nb, xb.data());

    const IndexPreTransform* index_pt =
            dynamic_cast<const IndexPreTransform*>(index.get());

    int dt = index->d;
    const float* xbt = xb.data();
    std::unique_ptr<float[]> del_xbt;

    if (index_pt) {
        dt = index_pt->index->d;
        xbt = index_pt->apply_chain(nb, xb.data());
        if (xbt != xb.data()) {
            del_xbt.reset((float*)xbt);
        }
    }

    IndexIVF* index_ivf = ivflib::extract_index_ivf(index.get());

    /** Test independent encoding
     *
     * Makes it possible to do additions on a custom inverted list
     * implementation. From a set of vectors, computes the inverted
     * list ids + the codes corresponding to each vector.
     */

    std::vector<idx_t> list_nos(nb);
    std::vector<uint8_t> codes(index_ivf->code_size * nb);
    index_ivf->quantizer->assign(nb, xbt, list_nos.data());
    index_ivf->encode_vectors(nb, xbt, list_nos.data(), codes.data());

    /** Test independent decoding
     *
     * Encode and decode twice on each vector and compare the first time decoded
     * vector and second time decoded vector (test decoding consistency only due
     * to data loss during encode and decode).
     */
    std::vector<float> decoded(nb * dt);
    std::vector<uint8_t> codes2(codes);
    std::vector<float> decoded2(nb * dt);
    index_ivf->decode_vectors(
            nb, codes.data(), list_nos.data(), decoded.data());
    index_ivf->encode_vectors(nb, xbt, list_nos.data(), codes2.data());
    index_ivf->decode_vectors(
            nb, codes2.data(), list_nos.data(), decoded2.data());
    EXPECT_LT(
            faiss::fvec_L2sqr(decoded.data(), decoded2.data(), nb * dt), 1e-5);

    // compare with normal IVF addition

    const InvertedLists* il = index_ivf->invlists;

    for (size_t list_no = 0; list_no < index_ivf->nlist; list_no++) {
        InvertedLists::ScopedCodes ivf_codes(il, list_no);
        InvertedLists::ScopedIds ivf_ids(il, list_no);
        size_t list_size = il->list_size(list_no);
        for (size_t i = 0; i < list_size; i++) {
            const uint8_t* ref_code = ivf_codes.get() + i * il->code_size;
            const uint8_t* new_code = codes.data() + ivf_ids[i] * il->code_size;
            EXPECT_EQ(memcmp(ref_code, new_code, il->code_size), 0);
        }
    }

    auto xq = make_data(nq);
    auto res = search_index(index.get(), xq.data());

    test_get_InvertedListScanner(
            index_ivf,
            index_pt,
            codes,
            nullptr,
            xq,
            res.first,
            res.second,
            metric);
}

void test_get_InvertedListScanner(
        IndexIVF* index_ivf,
        const IndexPreTransform* index_pt,
        std::vector<uint8_t> codes,
        std::unique_ptr<InvertedListScanner> scanner,
        std::vector<float> xq,
        std::vector<idx_t> ref_I,
        std::vector<float> ref_D,
        MetricType metric,
        bool distance_to_code_supported,
        float accuracy_requirement) {
    const float* xqt = xq.data();
    std::unique_ptr<float[]> del_xqt;

    if (index_pt) {
        xqt = index_pt->apply_chain(nq, xq.data());
        if (xqt != xq.data()) {
            del_xqt.reset((float*)xqt);
        }
    }

    int nprobe = index_ivf->nprobe;
    const InvertedLists* il = index_ivf->invlists;

    std::vector<idx_t> q_lists(nq * nprobe);
    std::vector<float> q_dis(nq * nprobe);

    index_ivf->quantizer->search(nq, xqt, nprobe, q_dis.data(), q_lists.data());

    if (scanner == nullptr) {
        scanner = std::unique_ptr<InvertedListScanner>(
                index_ivf->get_InvertedListScanner());
    }
    float recall = 0.0;
    for (size_t i = 0; i < nq; i++) {
        std::vector<idx_t> I(k, -1);
        float default_dis = metric == METRIC_L2
                ? std::numeric_limits<float>::max()
                : std::numeric_limits<float>::lowest();
        std::vector<float> D(k, default_dis);

        scanner->set_query(xqt + i * index_ivf->d);

        for (int j = 0; j < nprobe; j++) {
            int list_no = q_lists[i * nprobe + j];
            if (list_no < 0) {
                continue;
            }
            scanner->set_list(list_no, q_dis[i * nprobe + j]);

            // here we get the inverted lists from the InvertedLists
            // object but they could come from anywhere

            scanner->scan_codes(
                    il->list_size(list_no),
                    InvertedLists::ScopedCodes(il, list_no).get(),
                    InvertedLists::ScopedIds(il, list_no).get(),
                    D.data(),
                    I.data(),
                    k);

            if (distance_to_code_supported && j == 0) {
                // all results so far come from list_no, so let's check if
                // the distance function works
                for (int jj = 0; jj < k; jj++) {
                    int vno = I[jj];
                    if (vno < 0) {
                        break; // heap is not full yet
                    }

                    // we have the codes from the addition test
                    float computed_D = scanner->distance_to_code(
                            codes.data() + vno * il->code_size);

                    EXPECT_FLOAT_EQ(computed_D, D[jj]);
                }
            }
        }

        if (metric == METRIC_L2) {
            maxheap_reorder(k, D.data(), I.data());
        } else {
            minheap_reorder(k, D.data(), I.data());
        }

        for (int j = 0; j < k; j++) {
            if (I[j] != ref_I[i * k + j]) {
                if (accuracy_requirement == 1.0) {
                    EXPECT_EQ(D[j], ref_D[i * k + j]);
                }
                if (D[j] == ref_D[i * k + j]) {
                    recall += 1.0;
                }
            } else {
                recall += 1.0;
            }
        }
    }
    recall = recall / (k * nq);
    EXPECT_GE(recall, accuracy_requirement);
}

void test_fastscan_scanner(
        const char* factory_string,
        MetricType metric,
        float accuracy_requirement = 1.0,
        bool by_residual = false) {
    auto index =
            std::unique_ptr<Index>(index_factory(d, factory_string, metric));
    ParameterSpace().set_index_parameter(index.get(), "nprobe", 4);
    IndexIVFFastScan* index_ivf = static_cast<IndexIVFFastScan*>(
            ivflib::extract_index_ivf(index.get()));

    if (auto* pq_index = dynamic_cast<IndexIVFPQFastScan*>(index_ivf)) {
        pq_index->by_residual = by_residual;
    }

    // implem_10 also processes one query at a time, so compare with that.
    index_ivf->implem = 10;
    auto xt = make_data(nt);
    index->train(nt, xt.data());
    auto xb = make_data(nb);
    index->add(nb, xb.data());
    auto scanner = std::unique_ptr<InvertedListScanner>(
            index_ivf->get_InvertedListScanner());

    auto xq = make_data(nq);
    auto res = search_index(index.get(), xq.data());

    // distance_to_code_supported = false because codes are intermixed.
    test_get_InvertedListScanner(
            index_ivf,
            dynamic_cast<const IndexPreTransform*>(index.get()),
            {},
            std::move(scanner),
            xq,
            res.first,
            res.second,
            metric,
            false,
            accuracy_requirement);
}

} // anonymous namespace

/*************************************************************
 * Test entry points
 *************************************************************/

TEST(TestLowLevelIVF, IVFFlatL2) {
    test_lowlevel_access("IVF32,Flat", METRIC_L2);
}

TEST(TestLowLevelIVF, PCAIVFFlatL2) {
    test_lowlevel_access("PCAR16,IVF32,Flat", METRIC_L2);
}

TEST(TestLowLevelIVF, IVFFlatIP) {
    test_lowlevel_access("IVF32,Flat", METRIC_INNER_PRODUCT);
}

TEST(TestLowLevelIVF, IVFSQL2) {
    test_lowlevel_access("IVF32,SQ8", METRIC_L2);
}

TEST(TestLowLevelIVF, IVFSQIP) {
    test_lowlevel_access("IVF32,SQ8", METRIC_INNER_PRODUCT);
}

TEST(TestLowLevelIVF, IVFPQL2) {
    test_lowlevel_access("IVF32,PQ4np", METRIC_L2);
}

TEST(TestLowLevelIVF, IVFPQIP) {
    test_lowlevel_access("IVF32,PQ4np", METRIC_INNER_PRODUCT);
}

TEST(TestLowLevelIVF, IVFRaBitQ) {
    test_lowlevel_access("IVF32,RaBitQ", METRIC_L2);
}

TEST(TestLowLevelIVF, IVFRQ) {
    test_lowlevel_access("IVF32,RQ16x8", METRIC_L2);
}

TEST(TestLowLevelIVF, IVFPQFS_L2) {
    test_fastscan_scanner("IVF32,PQ4x4fs", METRIC_L2);
}

TEST(TestLowLevelIVF, IVFPQFS_IP) {
    test_fastscan_scanner("IVF32,PQ4x4fs", METRIC_INNER_PRODUCT);
}

TEST(TestLowLevelIVF, IVFPQFSr_L2) {
    test_fastscan_scanner("IVF32,PQ4x4fs", METRIC_L2, 0.9, true);
}

TEST(TestLowLevelIVF, IVFPQFSr_IP) {
    test_fastscan_scanner("IVF32,PQ4x4fs", METRIC_INNER_PRODUCT, 0.9, true);
}

TEST(TestLowLevelIVF, IVFRaBitQFS_L2) {
    test_fastscan_scanner("RR,IVF32,RaBitQfs", METRIC_L2, 0.90);
}

TEST(TestLowLevelIVF, IVFRaBitQFS_IP) {
    test_fastscan_scanner("RR,IVF32,RaBitQfs", METRIC_INNER_PRODUCT, 0.90);
}

TEST(TestLowLevelIVF, IVFRaBitQFS2_L2) {
    test_fastscan_scanner("RR,IVF32,RaBitQfs2", METRIC_L2, 0.90);
}

TEST(TestLowLevelIVF, IVFRaBitQFS2_IP) {
    test_fastscan_scanner("RR,IVF32,RaBitQfs2", METRIC_INNER_PRODUCT, 0.90);
}

TEST(TestLowLevelIVF, IVFRaBitQFS4_L2) {
    test_fastscan_scanner("RR,IVF32,RaBitQfs4", METRIC_L2, 0.90);
}

TEST(TestLowLevelIVF, IVFRaBitQFS4_IP) {
    test_fastscan_scanner("RR,IVF32,RaBitQfs4", METRIC_INNER_PRODUCT, 0.90);
}

/*************************************************************
 * Same for binary (a bit simpler)
 *************************************************************/

namespace {

int nbit = 256;

std::vector<uint8_t> make_data_binary(size_t n) {
    std::vector<uint8_t> database(n * nbit / 8);
    std::uniform_int_distribution<> distrib;
    for (size_t i = 0; i < n * d; i++) {
        database[i] = distrib(rng);
    }
    return database;
}

std::unique_ptr<IndexBinary> make_trained_index_binary(const char* index_type) {
    auto index = std::unique_ptr<IndexBinary>(
            index_binary_factory(nbit, index_type));
    auto xt = make_data_binary(nt);
    index->train(nt, xt.data());
    return index;
}

void test_lowlevel_access_binary(const char* index_key) {
    std::unique_ptr<IndexBinary> index = make_trained_index_binary(index_key);

    IndexBinaryIVF* index_ivf = dynamic_cast<IndexBinaryIVF*>(index.get());
    assert(index_ivf);

    index_ivf->nprobe = 4;

    auto xb = make_data_binary(nb);
    index->add(nb, xb.data());

    /* For binary there is no test for encoding because binary vectors
     * are copied verbatim to the inverted lists */

    const InvertedLists* il = index_ivf->invlists;

    auto xq = make_data_binary(nq);

    std::vector<idx_t> I_ref(k * nq);
    std::vector<int32_t> D_ref(k * nq);
    index->search(nq, xq.data(), k, D_ref.data(), I_ref.data());

    int nprobe = index_ivf->nprobe;

    std::vector<idx_t> q_lists(nq * nprobe);
    std::vector<int32_t> q_dis(nq * nprobe);

    index_ivf->quantizer->search(
            nq, xq.data(), nprobe, q_dis.data(), q_lists.data());

    std::unique_ptr<BinaryInvertedListScanner> scanner(
            index_ivf->get_InvertedListScanner());

    for (size_t i = 0; i < nq; i++) {
        std::vector<idx_t> I(k, -1);
        uint32_t default_dis = 1 << 30;
        std::vector<int32_t> D(k, default_dis);

        scanner->set_query(xq.data() + i * index_ivf->code_size);

        for (int j = 0; j < nprobe; j++) {
            int list_no = q_lists[i * nprobe + j];
            if (list_no < 0) {
                continue;
            }
            scanner->set_list(list_no, q_dis[i * nprobe + j]);

            scanner->scan_codes(
                    il->list_size(list_no),
                    InvertedLists::ScopedCodes(il, list_no).get(),
                    InvertedLists::ScopedIds(il, list_no).get(),
                    D.data(),
                    I.data(),
                    k);

            if (j == 0) {
                // all results so far come from list_no, so let's check if
                // the distance function works
                for (int jj = 0; jj < k; jj++) {
                    int vno = I[jj];
                    if (vno < 0) {
                        break; // heap is not full yet
                    }

                    float computed_D = scanner->distance_to_code(
                            xb.data() + vno * il->code_size);

                    EXPECT_EQ(computed_D, D[jj]);
                }
            }
        }

        heap_reorder<CMax<int32_t, idx_t>>(k, D.data(), I.data());

        for (int j = 0; j < k; j++) {
            // the order of ties is not guaranteed, so scan the ref results
            EXPECT_LE(D[j], D_ref[i * k + k - 1]);
            if (D[j] < D_ref[i * k + k - 1]) {
                int j2 = 0;
                while (j2 < k) {
                    if (I[j] == I_ref[i * k + j2]) {
                        break;
                    }
                    j2++;
                }
                EXPECT_LT(j2, k); // it was found
                if (j2 < k) {
                    EXPECT_EQ(D[j], D_ref[i * k + j2]);
                }
            }
        }
    }
}

} // anonymous namespace

TEST(TestLowLevelIVF, IVFBinary) {
    test_lowlevel_access_binary("BIVF32");
}

namespace {

void test_threaded_search(const char* index_key) {
    std::unique_ptr<Index> index = make_trained_index(index_key, METRIC_L2);

    auto xb = make_data(nb);
    index->add(nb, xb.data());

    IndexIVF* index_ivf = ivflib::extract_index_ivf(index.get());
    int dt = index->d;

    auto xq = make_data(nq);
    auto res = search_index(index.get(), xq.data());
    const float* xqt = xq.data();

    int nprobe = index_ivf->nprobe;

    std::vector<idx_t> q_lists(nq * nprobe);
    std::vector<float> q_dis(nq * nprobe);

    index_ivf->quantizer->search(nq, xqt, nprobe, q_dis.data(), q_lists.data());

    int nproc = 3;

    for (size_t i = 0; i < nq; i++) {
        std::vector<idx_t> I(k * nproc, -1);
        std::vector<float> D(k * nproc, HUGE_VAL);

        auto search_function = [index_ivf,
                                &I,
                                &D,
                                dt,
                                i,
                                nproc,
                                xqt,
                                nprobe,
                                &q_dis,
                                &q_lists](int rank) {
            const InvertedLists* il = index_ivf->invlists;

            std::unique_ptr<InvertedListScanner> scanner(
                    index_ivf->get_InvertedListScanner());

            idx_t* local_I = I.data() + rank * k;
            float* local_D = D.data() + rank * k;

            scanner->set_query(xqt + i * dt);

            for (int j = rank; j < nprobe; j += nproc) {
                int list_no = q_lists[i * nprobe + j];
                if (list_no < 0) {
                    continue;
                }
                scanner->set_list(list_no, q_dis[i * nprobe + j]);

                scanner->scan_codes(
                        il->list_size(list_no),
                        InvertedLists::ScopedCodes(il, list_no).get(),
                        InvertedLists::ScopedIds(il, list_no).get(),
                        local_D,
                        local_I,
                        k);
            }
        };

        // start the threads. Threads are numbered rank=0..nproc-1 (a la MPI)
        // thread rank takes care of inverted lists
        // rank, rank+nproc, rank+2*nproc,...
        std::vector<std::thread> threads;
        for (int rank = 0; rank < nproc; rank++) {
            threads.emplace_back(search_function, rank);
        }

        for (int rank = 0; rank < nproc; rank++) {
            threads[rank].join();
            if (rank == 0) {
                continue;
            }
            maxheap_addn(
                    k,
                    D.data(),
                    I.data(),
                    D.data() + rank * k,
                    I.data() + rank * k,
                    k);
        }

        maxheap_reorder(k, D.data(), I.data());

        for (int j = 0; j < k; j++) {
            EXPECT_EQ(I[j], res.first[i * k + j]);
        }
    }
}

} // namespace

TEST(TestLowLevelIVF, ThreadedSearch) {
    test_threaded_search("IVF32,Flat");
}
