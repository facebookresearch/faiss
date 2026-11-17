/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/approx_topk/approx_topk.h>
#include <faiss/utils/hamming.h>
#include <algorithm>
#include <cstdint>
#include <random>
#include <vector>

using namespace ::testing;

template <typename T>
std::string print_data(
        std::shared_ptr<std::vector<T>> data,
        const size_t divider) {
    std::string ret;
    for (size_t i = 0; i < data->size(); ++i) {
        if (i % divider) {
            ret += " ";
        } else {
            ret += "|";
        }
        ret += std::to_string((*data)[i]);
    }
    ret += "|";
    return ret;
}

std::stringstream get_correct_hamming_example(
        const size_t na, // number of queries
        const size_t nb, // number of candidates
        const size_t k,
        const size_t code_size,
        std::shared_ptr<std::vector<uint8_t>> a,
        std::shared_ptr<std::vector<uint8_t>> b,
        std::shared_ptr<std::vector<faiss::idx_t>> true_ids,
        // regular Hamming (bit-level distances)
        std::shared_ptr<std::vector<int>> true_bit_distances,
        // generalized Hamming (byte-level distances)
        std::shared_ptr<std::vector<int>> true_byte_distances) {
    assert(nb >= k);

    std::default_random_engine rng(123);
    std::uniform_int_distribution<int32_t> uniform(0, nb - 1);

    const size_t nresults = na * k;

    a->clear();
    a->resize(na * code_size, 1); // query vectors are all 1
    b->clear();
    b->resize(nb * code_size, 2); // database vectors are all 2
    true_ids->clear();
    true_ids->reserve(nresults);
    true_bit_distances->clear();
    true_bit_distances->reserve(nresults);
    true_byte_distances->clear();
    true_byte_distances->reserve(nresults);

    // define correct ids (must be unique)
    std::set<faiss::idx_t> correct_ids;
    do {
        correct_ids.insert(uniform(rng));
    } while (correct_ids.size() < k);

    // replace database vector at id with vector more similar to query
    // ordered, so earlier ids must be more similar
    for (size_t nmatches = k; nmatches > 0; --nmatches) {
        const size_t id = *correct_ids.begin();
        *correct_ids.erase(correct_ids.begin());

        true_ids->push_back(id);
        true_bit_distances->push_back(
                (code_size > nmatches ? code_size - nmatches : 0) *
                /* per-code distance between 1 and 2 (0b01 and 0b10) */
                2);
        true_byte_distances->push_back(
                (code_size > nmatches ? code_size - nmatches : 0));
        for (size_t i = 0; i < nmatches; ++i) {
            b->begin()[id * code_size + i] = 1; // query byte value
        }
    }

    // true_ids, true_bit_distances, true_byte_distances only contain results
    // for the first query.
    // Query vectors are identical (all 1s), so copy the first sets of k
    // distances na-1 times.
    for (size_t i = 1; i < na; ++i) {
        true_ids->insert(
                true_ids->end(), true_ids->begin(), true_ids->begin() + k);
        true_bit_distances->insert(
                true_bit_distances->end(),
                true_bit_distances->begin(),
                true_bit_distances->begin() + k);
        true_byte_distances->insert(
                true_byte_distances->end(),
                true_byte_distances->begin(),
                true_byte_distances->begin() + k);
    }

    // assemble string for debugging
    std::stringstream ret;
    ret << "na: " << na << std::endl
        << "nb: " << nb << std::endl
        << "k: " << k << std::endl
        << "code_size: " << code_size << std::endl
        << "a: " << print_data(a, code_size) << std::endl
        << "b: " << print_data(b, code_size) << std::endl
        << "true_ids: " << print_data(true_ids, k) << std::endl
        << "true_bit_distances: " << print_data(true_bit_distances, k)
        << std::endl
        << "true_byte_distances: " << print_data(true_byte_distances, k)
        << std::endl;
    return ret;
}

TEST(TestHamming, test_crosshamming_count_thres) {
    // 65 bytes hits the unsupported-size default case.
    const size_t n = 10;
    const size_t ncodes = 65;
    std::vector<uint8_t> dbs(n * ncodes);
    size_t count;
    ASSERT_THROW(
            faiss::crosshamming_count_thres(dbs.data(), n, 20, ncodes, &count),
            faiss::FaissException);
}

TEST(TestHamming, test_hamming_thres) {
    // 65 bytes hits the unsupported-size default case.
    const size_t n1 = 10;
    const size_t n2 = 15;
    const size_t ncodes = 65;
    const hamdis_t hamming_threshold = 100;
    std::vector<uint8_t> bs1(n1 * ncodes);
    std::vector<uint8_t> bs2(n2 * ncodes);
    std::vector<int64_t> idx(2 * n1 * n2);
    std::vector<hamdis_t> dis(n1 * n2);
    ASSERT_THROW(
            faiss::match_hamming_thres(
                    bs1.data(),
                    bs2.data(),
                    n1,
                    n2,
                    hamming_threshold,
                    ncodes,
                    idx.data(),
                    dis.data()),
            faiss::FaissException);
    ASSERT_THROW(
            faiss::hamming_count_thres(
                    bs1.data(),
                    bs2.data(),
                    n1,
                    n2,
                    hamming_threshold,
                    ncodes,
                    nullptr),
            faiss::FaissException);
}

TEST(TestHamming, test_hamming_knn) {
    // Initialize inputs
    const size_t na = 4;
    const size_t nb = 12; // number of candidates
    const size_t k = 6;

    auto a = std::make_shared<std::vector<uint8_t>>();
    auto b = std::make_shared<std::vector<uint8_t>>();
    auto true_ids = std::make_shared<std::vector<faiss::idx_t>>();
    auto true_bit_distances = std::make_shared<std::vector<int>>();
    auto true_byte_distances = std::make_shared<std::vector<int>>();

    // 8, 16, 32 are cases - 24 will hit default case
    // all should be multiples of 8
    for (auto code_size : {8, 16, 24, 32}) {
        std::stringstream assert_str = get_correct_hamming_example(
                na,
                nb,
                k,
                code_size,
                a,
                b,
                true_ids,
                true_bit_distances,
                true_byte_distances);

        std::vector<faiss::idx_t> ids_gen(na * k);
        std::vector<int> dist_gen(na * k);
        faiss::int_maxheap_array_t res = {
                na, k, ids_gen.data(), dist_gen.data()};
        faiss::generalized_hammings_knn_hc(
                &res, a->data(), b->data(), nb, code_size, true);
        ASSERT_EQ(ids_gen, *true_ids) << assert_str.str();
        ASSERT_EQ(dist_gen, *true_byte_distances) << assert_str.str();

        std::vector<faiss::idx_t> ids_ham_knn(na * k, 0);
        std::vector<int> dist_ham_knn(na * k, 0);
        res = {na, k, ids_ham_knn.data(), dist_ham_knn.data()};
        faiss::hammings_knn(&res, a->data(), b->data(), nb, code_size, true);
        ASSERT_EQ(ids_ham_knn, *true_ids) << assert_str.str();
        ASSERT_EQ(dist_ham_knn, *true_bit_distances) << assert_str.str();
    }
}

TEST(TestHamming, test_hammings_ragged_code_size) {
    std::default_random_engine rng(123);
    std::uniform_int_distribution<int32_t> uniform(0, 255);

    const size_t na = 3;
    const size_t nb = 7;

    for (auto code_size : {1, 2, 5, 12, 20, 33}) {
        std::vector<uint8_t> a(na * code_size);
        std::vector<uint8_t> b(nb * code_size);
        for (auto& v : a) {
            v = uniform(rng);
        }
        for (auto& v : b) {
            v = uniform(rng);
        }

        // Reference: byte-wise popcount of the XOR, independent of faiss.
        std::vector<hamdis_t> expected(na * nb);
        for (size_t i = 0; i < na; ++i) {
            for (size_t j = 0; j < nb; ++j) {
                int d = 0;
                for (int c = 0; c < code_size; ++c) {
                    uint8_t x = a[i * code_size + c] ^ b[j * code_size + c];
                    while (x) {
                        d += x & 1;
                        x >>= 1;
                    }
                }
                expected[i * nb + j] = d;
            }
        }

        std::vector<hamdis_t> dis(na * nb);
        faiss::hammings(a.data(), b.data(), na, nb, code_size, dis.data());
        EXPECT_EQ(dis, expected) << "code_size = " << code_size;
    }
}

namespace {

int reference_hamming_distance(const uint8_t* a, const uint8_t* b, size_t n) {
    int d = 0;
    for (size_t i = 0; i < n; i++) {
        uint8_t x = a[i] ^ b[i];
        while (x) {
            d += x & 1;
            x >>= 1;
        }
    }
    return d;
}

void fill_random(std::vector<uint8_t>& v, std::mt19937& rng) {
    for (auto& x : v) {
        x = static_cast<uint8_t>(rng());
    }
}

/* All pairwise distances of the random data set against a query. */
std::vector<int> reference_distances(
        const uint8_t* query,
        const std::vector<uint8_t>& b,
        size_t nb,
        size_t code_size) {
    std::vector<int> ref(nb);
    for (size_t j = 0; j < nb; j++) {
        ref[j] = reference_hamming_distance(
                query, b.data() + j * code_size, code_size);
    }
    return ref;
}

} // namespace

/* Differential tests for the batch distances against a brute-force
 * reference. They cover the exact top-k path (which uses the block-tile
 * kernels on RVV builds), the IDSelector specialization, the
 * multi-threshold top-k path and the range search. Code sizes include the
 * fixed 64/128/256/512-bit kernels plus a ragged size (24 bytes). */
TEST(TestHamming, test_hammings_knn_hc_reference) {
    std::mt19937 rng(12345);
    const size_t na = 5;
    const size_t nb = 10000; // > one 8192-code block-tile
    const size_t k = 10;
    const faiss::IDSelectorRange sel(1000, 6000);

    for (size_t code_size : {8, 16, 24, 32, 64}) {
        std::vector<uint8_t> a(na * code_size);
        std::vector<uint8_t> b(nb * code_size);
        fill_random(a, rng);
        fill_random(b, rng);

        std::vector<std::vector<int>> ref(na);
        for (size_t i = 0; i < na; i++) {
            ref[i] = reference_distances(
                    a.data() + i * code_size, b, nb, code_size);
        }

        auto check = [&](const faiss::IDSelector* s, const char* tag) {
            std::vector<faiss::idx_t> ids(na * k, -1);
            std::vector<int> dis(na * k, -1);
            faiss::int_maxheap_array_t res = {na, k, ids.data(), dis.data()};
            faiss::hammings_knn_hc(
                    &res,
                    a.data(),
                    b.data(),
                    nb,
                    code_size,
                    1,
                    ApproxTopK_mode_t::EXACT_TOPK,
                    s);

            for (size_t i = 0; i < na; i++) {
                std::vector<int> expected;
                for (size_t j = 0; j < nb; j++) {
                    if (s == nullptr || s->is_member(j)) {
                        expected.push_back(ref[i][j]);
                    }
                }
                std::sort(expected.begin(), expected.end());
                expected.resize(k);

                std::vector<int> got(
                        dis.begin() + i * k, dis.begin() + (i + 1) * k);
                std::sort(got.begin(), got.end());
                EXPECT_EQ(got, expected)
                        << "code_size=" << code_size << " " << tag;

                for (size_t t = 0; t < k; t++) {
                    const faiss::idx_t id = ids[i * k + t];
                    ASSERT_GE(id, 0) << "code_size=" << code_size << " " << tag;
                    ASSERT_LT(id, static_cast<faiss::idx_t>(nb))
                            << "code_size=" << code_size << " " << tag;
                    if (s != nullptr) {
                        ASSERT_TRUE(s->is_member(id));
                    }
                    EXPECT_EQ(dis[i * k + t], ref[i][id])
                            << "code_size=" << code_size << " " << tag;
                }
            }
        };

        check(nullptr, "no selector");
        check(&sel, "selector");
    }
}

TEST(TestHamming, test_hammings_knn_mc_reference) {
    std::mt19937 rng(54321);
    const size_t na = 3;
    const size_t nb = 5000;
    const size_t k = 10;

    for (size_t code_size : {8, 16, 32, 64}) {
        std::vector<uint8_t> a(na * code_size);
        std::vector<uint8_t> b(nb * code_size);
        fill_random(a, rng);
        fill_random(b, rng);

        std::vector<int32_t> dis(na * k, -1);
        std::vector<int64_t> ids(na * k, -1);
        faiss::hammings_knn_mc(
                a.data(),
                b.data(),
                na,
                nb,
                k,
                code_size,
                dis.data(),
                ids.data());

        for (size_t i = 0; i < na; i++) {
            std::vector<int> ref = reference_distances(
                    a.data() + i * code_size, b, nb, code_size);
            std::sort(ref.begin(), ref.end());
            ref.resize(k);

            std::vector<int> got(
                    dis.begin() + i * k, dis.begin() + (i + 1) * k);
            EXPECT_EQ(got, ref) << "code_size=" << code_size;
            for (size_t t = 0; t < k; t++) {
                const int64_t id = ids[i * k + t];
                ASSERT_GE(id, 0);
                ASSERT_LT(id, static_cast<int64_t>(nb));
                EXPECT_EQ(
                        dis[i * k + t],
                        reference_hamming_distance(
                                a.data() + i * code_size,
                                b.data() + id * code_size,
                                code_size))
                        << "code_size=" << code_size;
            }
        }
    }
}

TEST(TestHamming, test_hamming_range_search_reference) {
    std::mt19937 rng(999);
    const size_t na = 3;
    const size_t nb = 5000;

    for (size_t code_size : {8, 16, 32, 64}) {
        std::vector<uint8_t> a(na * code_size);
        std::vector<uint8_t> b(nb * code_size);
        fill_random(a, rng);
        fill_random(b, rng);

        const int radius = static_cast<int>(code_size * 8 * 42 / 100);
        faiss::RangeSearchResult res(na);
        faiss::hamming_range_search(
                a.data(), b.data(), na, nb, radius, code_size, &res);

        for (size_t i = 0; i < na; i++) {
            std::vector<int> ref = reference_distances(
                    a.data() + i * code_size, b, nb, code_size);
            size_t expected = 0;
            for (int d : ref) {
                expected += (d < radius);
            }
            EXPECT_EQ(res.lims[i + 1] - res.lims[i], expected)
                    << "code_size=" << code_size;

            for (size_t t = res.lims[i]; t < res.lims[i + 1]; t++) {
                const faiss::idx_t id = res.labels[t];
                ASSERT_GE(id, 0);
                ASSERT_LT(id, static_cast<faiss::idx_t>(nb));
                EXPECT_EQ(res.distances[t], ref[id]);
                EXPECT_LT(res.distances[t], radius);
            }
        }
    }
}
