/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cmath>
#include <memory>
#include <random>
#include <vector>

#include <faiss/IndexRaBitQ.h>
#include <faiss/impl/io.h>
#include <faiss/impl/simd_dispatch.h>
#include <faiss/index_io.h>
#include <faiss/utils/rabitq_simd.h>

namespace {

struct RestoreSIMD {
    faiss::SIMDLevel saved = faiss::SIMDConfig::get_level();
    ~RestoreSIMD() {
        faiss::SIMDConfig::set_level(saved);
    }
};

std::vector<faiss::SIMDLevel> available_levels() {
    std::vector<faiss::SIMDLevel> levels;
    for (auto level :
         {faiss::SIMDConfig::get_level(),
          faiss::SIMDLevel::NONE,
          faiss::SIMDLevel::AVX2,
          faiss::SIMDLevel::AVX512,
          faiss::SIMDLevel::AVX512_VPOPCNT}) {
        if (faiss::SIMDConfig::is_simd_level_available(level) &&
            std::find(levels.begin(), levels.end(), level) == levels.end()) {
            levels.push_back(level);
        }
    }
    return levels;
}

} // namespace

TEST(RaBitQBatchEstimates, MatchesSingleAcrossPrecisionsMetricsAndSIMD) {
    RestoreSIMD restore;
    std::mt19937 rng(12345);
    std::normal_distribution<float> normal;
    for (int d : {7, 64, 129, 769}) {
        std::vector<float> data(16 * d), queries(2 * d);
        for (auto& v : data) {
            v = normal(rng);
        }
        for (auto& v : queries) {
            v = normal(rng);
        }
        for (auto metric : {faiss::METRIC_L2, faiss::METRIC_INNER_PRODUCT}) {
            for (int bits = 1; bits <= 9; ++bits) {
                faiss::IndexRaBitQ index(d, metric, bits);
                index.train(16, data.data());
                index.add(16, data.data());
                const uint8_t* codes[] = {
                        index.codes.data() + 9 * index.code_size,
                        index.codes.data(),
                        index.codes.data() + 15 * index.code_size,
                        index.codes.data()};
                for (auto level : available_levels()) {
                    faiss::SIMDConfig::set_level(level);
                    for (int qb = 0; qb <= 8; ++qb) {
                        for (bool centered : {false, true}) {
                            SCOPED_TRACE(
                                    ::testing::Message()
                                    << d << '/' << bits << '/' << qb << '/'
                                    << centered << '/' << int(level));
                            std::unique_ptr<faiss::FlatCodesDistanceComputer>
                                    owner(index.get_quantized_distance_computer(
                                            qb, centered));
                            auto* dc = dynamic_cast<
                                    faiss::RaBitQDistanceComputer*>(
                                    owner.get());
                            ASSERT_NE(dc, nullptr);
                            for (int q = 0; q < 3; ++q) {
                                // Include the degenerate zero-residual query.
                                dc->set_query(
                                        q == 2 ? index.center.data()
                                               : queries.data() + q * d);
                                float distances[4];
                                dc->distance_to_code_1bit_batch_4(
                                        codes, distances);
                                for (int i = 0; i < 4; ++i) {
                                    ASSERT_TRUE(std::isfinite(distances[i]));
                                    EXPECT_FLOAT_EQ(
                                            distances[i],
                                            dc->distance_to_code_1bit(
                                                    codes[i]));
                                }
                                EXPECT_EQ(dc->stats.n_1bit, 0);
                                EXPECT_EQ(dc->stats.n_refine, 0);
                            }
                        }
                    }
                }
            }
        }
    }
}

TEST(RaBitQBatchEstimates, SerializationPreservesCodesAndEstimates) {
    constexpr int d = 65;
    std::vector<float> data(8 * d);
    for (size_t i = 0; i < data.size(); ++i) {
        data[i] = std::sin(float(i));
    }
    for (auto metric : {faiss::METRIC_L2, faiss::METRIC_INNER_PRODUCT}) {
        for (int bits : {1, 4, 8, 9}) {
            faiss::IndexRaBitQ index(d, metric, bits);
            index.train(8, data.data());
            index.add(8, data.data());
            faiss::VectorIOWriter writer;
            faiss::write_index(&index, &writer);
            faiss::VectorIOReader reader;
            reader.data = writer.data;
            std::unique_ptr<faiss::Index> loaded(faiss::read_index(&reader));
            auto* copy = dynamic_cast<faiss::IndexRaBitQ*>(loaded.get());
            ASSERT_NE(copy, nullptr);
            ASSERT_EQ(index.codes, copy->codes);
            std::unique_ptr<faiss::FlatCodesDistanceComputer> a(
                    index.get_FlatCodesDistanceComputer());
            std::unique_ptr<faiss::FlatCodesDistanceComputer> b(
                    copy->get_FlatCodesDistanceComputer());
            a->set_query(data.data());
            b->set_query(data.data());
            for (int i = 0; i < 8; ++i) {
                EXPECT_FLOAT_EQ((*a)(i), (*b)(i));
            }
        }
    }
}

TEST(RaBitQBatchEstimates, RBQ9ByteCodesMatchIndependentReference) {
    RestoreSIMD restore;
    std::mt19937 rng(2718);
    for (size_t d : {1, 7, 8, 9, 15, 16, 17, 63, 64, 65, 128, 257, 769}) {
        // Exactly d extra bytes after an intentionally unaligned start.
        std::vector<uint8_t> signs(1 + (d + 7) / 8), extra(1 + d);
        std::vector<float> query(d);
        for (auto& v : signs)
            v = rng();
        for (auto& v : extra)
            v = rng();
        for (auto& v : query)
            v = (int(rng() % 2001) - 1000) / 1000.f;
        double expected = 0, magnitude = 0;
        for (size_t i = 0; i < d; ++i) {
            const int sign = (signs[1 + i / 8] >> (i % 8)) & 1;
            const double term = query[i] * (extra[1 + i] + 256 * sign - 255.5);
            expected += term;
            magnitude += std::abs(term);
        }
        for (auto level : available_levels()) {
            faiss::SIMDConfig::set_level(level);
            const float actual = faiss::with_selected_simd_levels<
                    faiss::AVAILABLE_SIMD_LEVELS_BASE_WITH_VPOPCNT>(
                    [&]<faiss::SIMDLevel SL>() {
                        return faiss::rabitq::multibit::compute_inner_product<
                                SL>(
                                signs.data() + 1,
                                extra.data() + 1,
                                query.data(),
                                d,
                                8,
                                -255.5f);
                    });
            EXPECT_NEAR(actual, expected, 2e-6 * magnitude + 1e-5);
        }
    }
}

TEST(RaBitQBatchEstimates, Q4KernelUnalignedTailsMatchByteReference) {
    RestoreSIMD restore;
    std::mt19937 rng(42);
    for (size_t size :
         {0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 96, 97, 193}) {
        std::vector<uint8_t> query(1 + 4 * size), storage[4];
        for (auto& v : query)
            v = rng();
        const uint8_t* codes[4];
        for (int i = 0; i < 4; ++i) {
            storage[i].resize(i + 1 + size);
            for (auto& v : storage[i])
                v = rng();
            codes[i] = storage[i].data() + i + 1;
        }
        for (auto level : available_levels()) {
            faiss::SIMDConfig::set_level(level);
            faiss::rabitq::BitwiseAndDotProductResult results[4];
            faiss::with_selected_simd_levels<
                    faiss::AVAILABLE_SIMD_LEVELS_BASE_WITH_VPOPCNT>(
                    [&]<faiss::SIMDLevel SL>() {
                        faiss::rabitq::bitwise_q4_batch_4<SL>(
                                query.data() + 1, codes, size, results);
                    });
            for (int i = 0; i < 4; ++i) {
                uint64_t dot = 0, pop = 0;
                for (size_t j = 0; j < size; ++j) {
                    for (int bit = 0; bit < 8; ++bit) {
                        const int on = (codes[i][j] >> bit) & 1;
                        pop += on;
                        for (int plane = 0; plane < 4; ++plane) {
                            dot += (on & (query[1 + plane * size + j] >> bit)) *
                                    (1 << plane);
                        }
                    }
                }
                EXPECT_EQ(results[i].dot_product, dot);
                EXPECT_EQ(results[i].popcount, pop);
            }
        }
    }
}

TEST(RaBitQBatchEstimates, MultiBitKernelsMatchIndependentReference) {
    RestoreSIMD restore;
    std::mt19937 rng(31415);
    for (size_t d : {1, 7, 8, 9, 15, 16, 17, 65, 129}) {
        for (size_t ex_bits = 1; ex_bits <= 8; ++ex_bits) {
            // Existing sub-byte scorers may read a uint64 window into the
            // trailing factor area; reserve those bytes as in an encoded row.
            std::vector<uint8_t> signs(1 + (d + 7) / 8);
            std::vector<uint8_t> extra(1 + (d * ex_bits + 7) / 8 + 8);
            std::vector<float> query(d);
            for (auto& v : signs)
                v = rng();
            double expected = 0, magnitude = 0;
            const float cb = -(float(1u << ex_bits) - 0.5f);
            for (size_t i = 0; i < d; ++i) {
                query[i] = (int(rng() % 2001) - 1000) / 1000.f;
                const unsigned value = rng() % (1u << ex_bits);
                for (size_t bit = 0; bit < ex_bits; ++bit) {
                    const size_t position = i * ex_bits + bit;
                    extra[1 + position / 8] |= ((value >> bit) & 1)
                            << (position % 8);
                }
                const int sign = (signs[1 + i / 8] >> (i % 8)) & 1;
                const double term =
                        query[i] * (double(value) + (sign << ex_bits) + cb);
                expected += term;
                magnitude += std::abs(term);
            }
            for (auto level : available_levels()) {
                faiss::SIMDConfig::set_level(level);
                const float actual = faiss::with_selected_simd_levels<
                        faiss::AVAILABLE_SIMD_LEVELS_BASE_WITH_VPOPCNT>(
                        [&]<faiss::SIMDLevel SL>() {
                            return faiss::rabitq::multibit::
                                    compute_inner_product<SL>(
                                            signs.data() + 1,
                                            extra.data() + 1,
                                            query.data(),
                                            d,
                                            ex_bits,
                                            cb);
                        });
                EXPECT_NEAR(actual, expected, 2e-6 * magnitude + 1e-5);
            }
        }
    }
}
