/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/utils/rabitq_integer_adc.h>
#include <faiss/utils/simd_levels.h>
#include <gtest/gtest.h>

#include <algorithm>
#include <cstdlib>
#include <random>
#include <vector>

namespace {
namespace adc = faiss::rabitq_integer_adc;

bool available() {
#ifdef COMPILE_SIMD_AVX2
    return faiss::SIMDConfig::is_simd_level_available(faiss::SIMDLevel::AVX2);
#elif defined(COMPILE_SIMD_ARM_NEON)
    return true;
#else
    return false;
#endif
}

void product(
        const int8_t* query,
        const int8_t* const* levels,
        int count,
        size_t d,
        int64_t* dots) {
#ifdef COMPILE_SIMD_AVX2
    adc::dot_product_batch_avx2(query, levels, count, d, dots);
#elif defined(COMPILE_SIMD_ARM_NEON)
    if (count == 16) {
        adc::dot_product_batch_16_arm(query, levels, d, dots);
    } else if (count == 8) {
        adc::dot_product_batch_8_arm(query, levels, d, dots);
    } else if (count == 4) {
        adc::dot_product_batch_4_arm(
                query,
                levels[0],
                levels[1],
                levels[2],
                levels[3],
                d,
                dots[0],
                dots[1],
                dots[2],
                dots[3]);
    } else {
        adc::dot_product_batch_tail_arm(query, levels, count, d, dots);
    }
#endif
}

void put(uint8_t* bytes, size_t j, size_t bits, unsigned value) {
    for (size_t bit = 0; bit < bits; ++bit) {
        const size_t offset = j * bits + bit;
        bytes[offset / 8] |= ((value >> bit) & 1) << (offset % 8);
    }
}

} // namespace

TEST(RaBitQIntegerADC, SignedBytesAndUnalignedTails) {
    if (!available()) {
        GTEST_SKIP() << "Requires AVX2 or NEON";
    }
    std::mt19937 rng(77231);
    for (size_t d :
         {0,
          1,
          7,
          15,
          16,
          17,
          31,
          32,
          33,
          63,
          64,
          65,
          127,
          768,
          769,
          4095,
          4096,
          4097}) {
        std::vector<int8_t> query(d + 1);
        std::vector<std::vector<int8_t>> rows(16, std::vector<int8_t>(d + 1));
        const int8_t* levels[16];
        for (auto& value : query) {
            value = int(rng() % 256) - 128;
        }
        for (int row = 0; row < 16; ++row) {
            for (auto& value : rows[row]) {
                value = int(rng() % 256) - 128;
            }
            levels[row] = rows[row].data() + 1;
        }
        for (int count : {1, 2, 3, 4, 5, 6, 7, 8, 16}) {
            int64_t dots[16];
            product(query.data() + 1, levels, count, d, dots);
            for (int row = 0; row < count; ++row) {
                int64_t expected = 0;
                for (size_t j = 0; j < d; ++j) {
                    expected += int64_t(query[j + 1]) * levels[row][j];
                }
                ASSERT_EQ(dots[row], expected)
                        << "d=" << d << " count=" << count << " row=" << row;
            }
        }
    }
}

TEST(RaBitQIntegerADC, AccumulationBeyondInt32) {
    if (!available()) {
        GTEST_SKIP() << "Requires AVX2 or NEON";
    }
    constexpr size_t d = 262145;
    std::vector<int8_t> query(d, -128);
    const int8_t* levels[16];
    std::fill_n(levels, 16, query.data());
    int64_t dots[16];
    product(query.data(), levels, 16, d, dots);
    for (auto dot : dots) {
        EXPECT_EQ(dot, int64_t(d) * 128 * 128);
    }
}

TEST(RaBitQIntegerADC, CompactLayoutsMatchIndependentLevels) {
    if (!available()) {
        GTEST_SKIP() << "Requires AVX2 or NEON";
    }
    std::mt19937 rng(87341);
    for (size_t d :
         {0, 1, 15, 16, 17, 31, 32, 33, 63, 64, 65, 127, 768, 769, 4097}) {
        std::vector<int8_t> query(d + 1);
        for (auto& value : query) {
            value = int(rng() % 256) - 128;
        }
        for (int layout = 0; layout < 4; ++layout) {
            // Staged LUT4, nibble LUT4, LUT7, and two-bit prefix.
            const size_t prefix_bytes = (2 * d + 7) / 8;
            const size_t local_offset = prefix_bytes + 12;
            const size_t high_offset = local_offset + (4 * d + 7) / 8;
            const size_t row_bytes = layout == 1 ? (4 * d + 7) / 8 + 8
                    : layout == 2                ? high_offset + (d + 7) / 8
                                  : local_offset + (2 * d + 7) / 8;
            uint8_t lut[64];
            for (auto& value : lut) {
                value = rng() % (layout == 2 ? 64 : 8);
            }
            std::vector<std::vector<uint8_t>> rows(
                    16, std::vector<uint8_t>(row_bytes + 1));
            const uint8_t* codes[16];
            int64_t expected[16] = {};
            for (int row = 0; row < 16; ++row) {
                uint8_t* code = rows[row].data() + 1;
                codes[row] = code;
                for (size_t j = 0; j < d; ++j) {
                    const bool positive = rng() & 1;
                    const int coarse = rng() & 1;
                    const int prefix = (int(positive) << 1) |
                            (coarse ^ (positive ? 0 : 1));
                    const int local = rng() % (layout == 2 ? 32 : 4);
                    const int magnitude =
                            lut[coarse * (layout == 2 ? 32 : 4) + local];
                    if (layout == 1) {
                        put(code,
                            j,
                            4,
                            (int(positive) << 3) | (coarse << 2) | local);
                    } else {
                        put(code, j, 2, prefix);
                        if (layout == 2) {
                            put(code + local_offset, j, 4, local & 15);
                            put(code + high_offset, j, 1, local >> 4);
                        } else if (layout == 0) {
                            put(code + local_offset, j, 2, local);
                        }
                    }
                    const int level = layout == 3 ? prefix - 2
                            : positive            ? magnitude
                                                  : -1 - magnitude;
                    expected[row] += int64_t(query[j + 1]) * level;
                }
            }
            for (int count : {1, 2, 3, 4, 5, 6, 7, 8, 16}) {
                int64_t dots[16] = {};
#ifdef COMPILE_SIMD_AVX2
                if (layout < 2) {
                    adc::dot_product_nested_lut4_batch_avx2(
                            query.data() + 1,
                            codes,
                            count,
                            d,
                            local_offset,
                            lut,
                            layout == 1,
                            dots);
                } else if (layout == 2) {
                    adc::dot_product_nested_lut7_batch_avx2(
                            query.data() + 1,
                            codes,
                            count,
                            d,
                            local_offset,
                            high_offset,
                            lut,
                            dots);
                } else {
                    adc::dot_product_progressive_batch_avx2(
                            query.data() + 1,
                            codes,
                            count,
                            d,
                            0,
                            0,
                            true,
                            dots);
                }
#elif defined(COMPILE_SIMD_ARM_NEON)
                if (layout < 2) {
                    adc::dot_product_nested_lut4_batch_arm(
                            query.data() + 1,
                            codes,
                            count,
                            d,
                            local_offset,
                            lut,
                            layout == 1,
                            dots);
                } else if (layout == 2) {
                    adc::dot_product_nested_lut7_batch_arm(
                            query.data() + 1,
                            codes,
                            count,
                            d,
                            local_offset,
                            high_offset,
                            lut,
                            dots);
                } else {
                    adc::dot_product_progressive_prefix_batch_arm(
                            query.data() + 1, codes, count, d, dots);
                }
#endif
                for (int row = 0; row < count; ++row) {
                    ASSERT_EQ(dots[row], expected[row])
                            << "layout=" << layout << " d=" << d
                            << " count=" << count << " row=" << row;
                }
            }
        }
    }
}

TEST(RaBitQIntegerADC, PackedAndProgressiveLevels) {
    if (!available()) {
        GTEST_SKIP() << "Requires AVX2 or NEON";
    }
    std::mt19937 rng(19317);
    for (size_t d : {1, 15, 16, 31, 32, 33, 65, 768, 769, 4097}) {
        std::vector<int8_t> query(d);
        for (auto& value : query) {
            value = int(rng() % 256) - 128;
        }
        for (bool progressive : {false, true}) {
            for (size_t bits = 1; bits <= 7; ++bits) {
#ifndef COMPILE_SIMD_AVX2
                if (progressive ? (bits != 5 && bits != 6)
                                : (bits != 1 && bits != 3)) {
                    continue;
                }
#endif
                const size_t prefix_bytes =
                        progressive ? (2 * d + 7) / 8 : (d + 7) / 8;
                const size_t extra_offset = prefix_bytes + 12;
                const size_t row_bytes = extra_offset + (bits * d + 7) / 8;
                std::vector<std::vector<uint8_t>> rows(
                        16, std::vector<uint8_t>(row_bytes));
                const uint8_t* codes[16];
                int64_t expected[16] = {};
                for (int row = 0; row < 16; ++row) {
                    codes[row] = rows[row].data();
                    for (size_t j = 0; j < d; ++j) {
                        const bool positive = rng() & 1;
                        const int low = rng() % (1 << bits);
                        put(rows[row].data(),
                            j,
                            progressive ? 2 : 1,
                            progressive ? (int(positive) << 1) | (rng() & 1)
                                        : int(positive));
                        put(rows[row].data() + extra_offset, j, bits, low);
                        expected[row] += int64_t(query[j]) *
                                (low - (positive ? 0 : int(1 << bits)));
                    }
                }
                for (int count : {1, 2, 3, 4, 5, 6, 7, 8, 16}) {
                    int64_t dots[16];
#ifdef COMPILE_SIMD_AVX2
                    if (progressive) {
                        adc::dot_product_progressive_batch_avx2(
                                query.data(),
                                codes,
                                count,
                                d,
                                extra_offset,
                                bits,
                                false,
                                dots);
                    } else {
                        adc::dot_product_packed_batch_avx2(
                                query.data(),
                                codes,
                                count,
                                d,
                                extra_offset,
                                bits,
                                dots);
                    }
#elif defined(COMPILE_SIMD_ARM_NEON)
                    if (progressive) {
                        if (bits == 5) {
                            adc::dot_product_progressive_full6_batch_arm(
                                    query.data(),
                                    codes,
                                    count,
                                    d,
                                    extra_offset,
                                    dots);
                        } else {
                            adc::dot_product_progressive_full7_batch_arm(
                                    query.data(),
                                    codes,
                                    count,
                                    d,
                                    extra_offset,
                                    dots);
                        }
                    } else if (bits == 3) {
                        adc::dot_product_packed_4bit_batch_arm(
                                query.data(),
                                codes,
                                count,
                                d,
                                extra_offset,
                                dots);
                    } else if (count == 16) {
                        adc::dot_product_packed_2bit_batch_16_arm(
                                query.data(), codes, d, extra_offset, dots);
                    } else if (count == 8) {
                        adc::dot_product_packed_2bit_batch_8_arm(
                                query.data(), codes, d, extra_offset, dots);
                    } else {
                        adc::dot_product_packed_2bit_batch_tail_arm(
                                query.data(),
                                codes,
                                count,
                                d,
                                extra_offset,
                                dots);
                    }
#endif
                    for (int row = 0; row < count; ++row) {
                        ASSERT_EQ(dots[row], expected[row])
                                << "progressive=" << progressive
                                << " bits=" << bits << " d=" << d
                                << " count=" << count;
                    }
                }
            }
        }
    }
}

TEST(RaBitQIntegerADC, DotProductCapabilityMatchesTestCPU) {
#ifdef COMPILE_SIMD_ARM_NEON
    if (const char* expected = std::getenv("FAISS_TEST_EXPECT_DOTPROD")) {
        EXPECT_EQ(adc::arm_dotprod_supported(), std::atoi(expected) != 0);
    }
#endif
}
