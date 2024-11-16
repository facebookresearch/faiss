/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <faiss/utils/simdlib.h>

using namespace faiss;

TEST(TestSIMDLib, TestCmpltAndBlendInplace) {
    simd8float32 lowestValues(0, 1, 2, 3, 4, 5, 6, 7);
    simd8uint32 lowestIndices(0, 1, 2, 3, 4, 5, 6, 7);

    simd8float32 candidateValues0(5, 5, 5, 5, 5, 5, 5, 5);
    simd8uint32 candidateIndices0(10, 11, 12, 13, 14, 15, 16, 17);
    cmplt_and_blend_inplace(
            candidateValues0, candidateIndices0, lowestValues, lowestIndices);

    simd8float32 candidateValues1(6, 6, 6, 6, 6, 6, 6, 6);
    simd8uint32 candidateIndices1(20, 21, 22, 23, 24, 25, 26, 27);
    cmplt_and_blend_inplace(
            candidateValues1, candidateIndices1, lowestValues, lowestIndices);

    simd8float32 candidateValues2(0, 1, 2, 3, 4, 5, 5, 5);
    simd8uint32 candidateIndices2(30, 31, 32, 33, 34, 35, 36, 37);
    cmplt_and_blend_inplace(
            candidateValues2, candidateIndices2, lowestValues, lowestIndices);

    simd8float32 expectedValues(0, 1, 2, 3, 4, 5, 5, 5);
    simd8uint32 expectedIndices(0, 1, 2, 3, 4, 5, 16, 17);
    ASSERT_TRUE(lowestValues.is_same_as(expectedValues));
    ASSERT_TRUE(lowestIndices.is_same_as(expectedIndices));
}

TEST(TestSIMDLib, TestCmpltMinMaxFloat) {
    simd8float32 minValues(0, 0, 0, 0, 0, 0, 0, 0);
    simd8uint32 minIndices(0, 0, 0, 0, 0, 0, 0, 0);
    simd8float32 maxValues(0, 0, 0, 0, 0, 0, 0, 0);
    simd8uint32 maxIndices(0, 0, 0, 0, 0, 0, 0, 0);

    simd8float32 candidateValues0(5, 5, 5, 5, 5, 5, 5, 5);
    simd8uint32 candidateIndices0(10, 11, 12, 13, 14, 15, 16, 17);
    simd8float32 currentValues0(0, 1, 2, 3, 4, 5, 6, 7);
    simd8uint32 currentIndices0(0, 1, 2, 3, 4, 5, 6, 7);

    cmplt_min_max_fast(
            candidateValues0,
            candidateIndices0,
            currentValues0,
            currentIndices0,
            minValues,
            minIndices,
            maxValues,
            maxIndices);

    simd8float32 expectedMinValues(0, 1, 2, 3, 4, 5, 5, 5);
    simd8uint32 expectedMinIndices(0, 1, 2, 3, 4, 5, 16, 17);
    ASSERT_TRUE(minValues.is_same_as(expectedMinValues));
    ASSERT_TRUE(minIndices.is_same_as(expectedMinIndices));

    simd8float32 expectedMaxValues(5, 5, 5, 5, 5, 5, 6, 7);
    // the result is not 10,11,12,13,14,5,6,7 because it is _fast version
    simd8uint32 expectedMaxIndices(10, 11, 12, 13, 14, 15, 6, 7);
    ASSERT_TRUE(maxValues.is_same_as(expectedMaxValues));
    ASSERT_TRUE(maxIndices.is_same_as(expectedMaxIndices));
}

TEST(TestSIMDLib, TestCmpltMinMaxInt) {
    simd8uint32 minValues(0, 0, 0, 0, 0, 0, 0, 0);
    simd8uint32 minIndices(0, 0, 0, 0, 0, 0, 0, 0);
    simd8uint32 maxValues(0, 0, 0, 0, 0, 0, 0, 0);
    simd8uint32 maxIndices(0, 0, 0, 0, 0, 0, 0, 0);

    simd8uint32 candidateValues0(5, 5, 5, 5, 5, 5, 5, 5);
    simd8uint32 candidateIndices0(10, 11, 12, 13, 14, 15, 16, 17);
    simd8uint32 currentValues0(0, 1, 2, 3, 4, 5, 6, 7);
    simd8uint32 currentIndices0(0, 1, 2, 3, 4, 5, 6, 7);

    cmplt_min_max_fast(
            candidateValues0,
            candidateIndices0,
            currentValues0,
            currentIndices0,
            minValues,
            minIndices,
            maxValues,
            maxIndices);

    simd8uint32 expectedMinValues(0, 1, 2, 3, 4, 5, 5, 5);
    simd8uint32 expectedMinIndices(0, 1, 2, 3, 4, 5, 16, 17);
    ASSERT_TRUE(minValues.is_same_as(expectedMinValues));
    ASSERT_TRUE(minIndices.is_same_as(expectedMinIndices));

    simd8uint32 expectedMaxValues(5, 5, 5, 5, 5, 5, 6, 7);
    // the result is not 10,11,12,13,14,5,6,7 because it is _fast version
    simd8uint32 expectedMaxIndices(10, 11, 12, 13, 14, 15, 6, 7);
    ASSERT_TRUE(maxValues.is_same_as(expectedMaxValues));
    ASSERT_TRUE(maxIndices.is_same_as(expectedMaxIndices));
}

TEST(TestSIMDLib, TestCmpltMinMaxInt16) {
    simd16uint16 minValues(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    simd16uint16 minIndices(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    simd16uint16 maxValues(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);
    simd16uint16 maxIndices(0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0);

    simd16uint16 candidateValues0(
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
    simd16uint16 candidateIndices0(
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
    simd16uint16 currentValues0(
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
    simd16uint16 currentIndices0(
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

    simd16uint16 expectedMinValues(
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
    simd16uint16 expectedMinIndices(
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

    simd16uint16 expectedMaxValues(
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
    simd16uint16 expectedMaxIndices(
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
