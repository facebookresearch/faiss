#include <gtest/gtest.h>

#include <faiss/utils/simdlib.h>

using namespace faiss;

TEST(TEST_SIMDLIB, TestCmpltAndBlendInplace) {
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
    ASSERT_EQ(lowestValues, expectedValues);
    ASSERT_EQ(lowestIndices, expectedIndices);
}
