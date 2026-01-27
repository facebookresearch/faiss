/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <faiss/utils/simdlib.h>

using namespace faiss;

template <typename T, typename U>
void TestCmpltAndBlendInplaceTemplate() {
    T lowestValues(0, 1, 2, 3, 4, 5, 6, 7);
    U lowestIndices(0, 1, 2, 3, 4, 5, 6, 7);

    T candidateValues0(5, 5, 5, 5, 5, 5, 5, 5);
    U candidateIndices0(10, 11, 12, 13, 14, 15, 16, 17);
    cmplt_and_blend_inplace(
            candidateValues0, candidateIndices0, lowestValues, lowestIndices);

    T candidateValues1(6, 6, 6, 6, 6, 6, 6, 6);
    U candidateIndices1(20, 21, 22, 23, 24, 25, 26, 27);
    cmplt_and_blend_inplace(
            candidateValues1, candidateIndices1, lowestValues, lowestIndices);

    T candidateValues2(0, 1, 2, 3, 4, 5, 5, 5);
    U candidateIndices2(30, 31, 32, 33, 34, 35, 36, 37);
    cmplt_and_blend_inplace(
            candidateValues2, candidateIndices2, lowestValues, lowestIndices);

    T expectedValues(0, 1, 2, 3, 4, 5, 5, 5);
    U expectedIndices(0, 1, 2, 3, 4, 5, 16, 17);
    ASSERT_TRUE(lowestValues.is_same_as(expectedValues));
    ASSERT_TRUE(lowestIndices.is_same_as(expectedIndices));
}

template <typename T, typename U>
void TestCmpltMinMaxFloatTemplate() {
    T minValues(0, 0, 0, 0, 0, 0, 0, 0);
    U minIndices(0, 0, 0, 0, 0, 0, 0, 0);
    T maxValues(0, 0, 0, 0, 0, 0, 0, 0);
    U maxIndices(0, 0, 0, 0, 0, 0, 0, 0);

    T candidateValues0(5, 5, 5, 5, 5, 5, 5, 5);
    U candidateIndices0(10, 11, 12, 13, 14, 15, 16, 17);
    T currentValues0(0, 1, 2, 3, 4, 5, 6, 7);
    U currentIndices0(0, 1, 2, 3, 4, 5, 6, 7);

    cmplt_min_max_fast(
            candidateValues0,
            candidateIndices0,
            currentValues0,
            currentIndices0,
            minValues,
            minIndices,
            maxValues,
            maxIndices);

    T expectedMinValues(0, 1, 2, 3, 4, 5, 5, 5);
    U expectedMinIndices(0, 1, 2, 3, 4, 5, 16, 17);
    ASSERT_TRUE(minValues.is_same_as(expectedMinValues));
    ASSERT_TRUE(minIndices.is_same_as(expectedMinIndices));

    T expectedMaxValues(5, 5, 5, 5, 5, 5, 6, 7);
    // the result is not 10,11,12,13,14,5,6,7 because it is _fast version
    U expectedMaxIndices(10, 11, 12, 13, 14, 15, 6, 7);
    ASSERT_TRUE(maxValues.is_same_as(expectedMaxValues));
    ASSERT_TRUE(maxIndices.is_same_as(expectedMaxIndices));
}

template <typename T, typename U>
void TestCmpltMinMaxIntTemplate() {
    T minValues(0, 0, 0, 0, 0, 0, 0, 0);
    T minIndices(0, 0, 0, 0, 0, 0, 0, 0);
    T maxValues(0, 0, 0, 0, 0, 0, 0, 0);
    T maxIndices(0, 0, 0, 0, 0, 0, 0, 0);

    T candidateValues0(5, 5, 5, 5, 5, 5, 5, 5);
    T candidateIndices0(10, 11, 12, 13, 14, 15, 16, 17);
    T currentValues0(0, 1, 2, 3, 4, 5, 6, 7);
    T currentIndices0(0, 1, 2, 3, 4, 5, 6, 7);

    cmplt_min_max_fast(
            candidateValues0,
            candidateIndices0,
            currentValues0,
            currentIndices0,
            minValues,
            minIndices,
            maxValues,
            maxIndices);

    T expectedMinValues(0, 1, 2, 3, 4, 5, 5, 5);
    T expectedMinIndices(0, 1, 2, 3, 4, 5, 16, 17);
    ASSERT_TRUE(minValues.is_same_as(expectedMinValues));
    ASSERT_TRUE(minIndices.is_same_as(expectedMinIndices));

    T expectedMaxValues(5, 5, 5, 5, 5, 5, 6, 7);
    // the result is not 10,11,12,13,14,5,6,7 because it is _fast version
    T expectedMaxIndices(10, 11, 12, 13, 14, 15, 6, 7);
    ASSERT_TRUE(maxValues.is_same_as(expectedMaxValues));
    ASSERT_TRUE(maxIndices.is_same_as(expectedMaxIndices));
}

template <typename T>
void TestCmpltMinMaxInt16Template() {
    T minValues(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    T minIndices(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    T maxValues(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    T maxIndices(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

    T candidateValues0(
            5,
            5,
            5,
            5,
            5,
            5,
            5,
            5,
            1005,
            1005,
            1005,
            1005,
            1005,
            1005,
            1005,
            1005);
    T candidateIndices0(
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            1010,
            1011,
            1012,
            1013,
            1014,
            1015,
            1016,
            1017);
    T currentValues0(
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            1000,
            1001,
            1002,
            1003,
            1004,
            1005,
            1006,
            1007);
    T currentIndices0(
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            1000,
            1001,
            1002,
            1003,
            1004,
            1005,
            1006,
            1007);

    cmplt_min_max_fast(
            candidateValues0,
            candidateIndices0,
            currentValues0,
            currentIndices0,
            minValues,
            minIndices,
            maxValues,
            maxIndices);

    T expectedMinValues(
            0,
            1,
            2,
            3,
            4,
            5,
            5,
            5,
            1000,
            1001,
            1002,
            1003,
            1004,
            1005,
            1005,
            1005);
    T expectedMinIndices(
            0,
            1,
            2,
            3,
            4,
            5,
            16,
            17,
            1000,
            1001,
            1002,
            1003,
            1004,
            1005,
            1016,
            1017);
    ASSERT_TRUE(minValues.is_same_as(expectedMinValues));
    ASSERT_TRUE(minIndices.is_same_as(expectedMinIndices));

    T expectedMaxValues(
            5,
            5,
            5,
            5,
            5,
            5,
            6,
            7,
            1005,
            1005,
            1005,
            1005,
            1005,
            1005,
            1006,
            1007);
    // the result is not 10,11,12,13,14,5,6,7 because it is _fast version
    T expectedMaxIndices(
            10,
            11,
            12,
            13,
            14,
            15,
            6,
            7,
            1010,
            1011,
            1012,
            1013,
            1014,
            1015,
            1006,
            1007);
    ASSERT_TRUE(maxValues.is_same_as(expectedMaxValues));
    ASSERT_TRUE(maxIndices.is_same_as(expectedMaxIndices));
}

#define TEST_SIMD_LEVEL(level)                                          \
    TEST(TestSIMDLib, TestCmpltAndBlendInplace##level) {                \
        TestCmpltAndBlendInplaceTemplate<                               \
                simd8float32<SIMDLevel::level>,                         \
                simd8uint32<SIMDLevel::level>>();                       \
    }                                                                   \
    TEST(TestSIMDLib, TestCmpltMinMaxFloatInplace##level) {             \
        TestCmpltMinMaxFloatTemplate<                                   \
                simd8float32<SIMDLevel::level>,                         \
                simd8uint32<SIMDLevel::level>>();                       \
    }                                                                   \
    TEST(TestSIMDLib, TestCmpltMinMaxIntInplace##level) {               \
        TestCmpltMinMaxIntTemplate<                                     \
                simd8uint32<SIMDLevel::level>,                          \
                simd8uint32<SIMDLevel::level>>();                       \
    }                                                                   \
    TEST(TestSIMDLib, TestCmpltMinMaxInt16Inplace##level) {             \
        TestCmpltMinMaxInt16Template<simd16uint16<SIMDLevel::level>>(); \
    }

#if defined(__x86_64__)
#if defined(__AVX512F__) && defined(__AVX512VL__) && defined(__AVX512BW__) && \
        defined(__AVX512DQ__) && defined(__AVX512CD__)
TEST_SIMD_LEVEL(AVX512)
#endif // __AVX512__

#if defined(__AVX2__)
TEST_SIMD_LEVEL(AVX2)
#endif // __AVX2__

#elif defined(__aarch64__)

#if defined(__ARM_NEON)
TEST_SIMD_LEVEL(ARM_NEON)
#endif // __ARM_NEON

#endif // __aarch64__

TEST_SIMD_LEVEL(NONE)
