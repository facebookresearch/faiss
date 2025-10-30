/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cstddef>
#include <cstring>
#include <vector>

#include <faiss/IndexIVF.h>
#include <faiss/impl/pq4_fast_scan.h>
#include <faiss/utils/AlignedTable.h>
#include <faiss/utils/random.h>

/**
 * This validates we can run fast scan with unaligned codes and get the same
 * results as aligned codes. On modern CPUs, unaligned codes appear to have
 * similar performance characteristics to aligned codes.
 */

namespace {

std::vector<uint8_t> create_aligned_buffer(size_t size, size_t alignment) {
    // Allocate extra space for alignment adjustment
    std::vector<uint8_t> buffer(size + alignment);
    return buffer;
}

uint8_t* get_aligned_ptr(std::vector<uint8_t>& buffer, size_t alignment) {
    uint8_t* new_ptr;
    int ret = posix_memalign(
            (void**)&new_ptr, alignment, buffer.size() * sizeof(uint8_t));
    if (ret != 0) {
        throw std::bad_alloc();
    }
    return new_ptr;
}

uint8_t* get_unaligned_ptr(std::vector<uint8_t>& buffer, size_t alignment) {
    uint8_t* aligned = get_aligned_ptr(buffer, alignment);
    // Return pointer that is off by 1 byte from alignment
    return aligned + 1;
}

} // namespace

TEST(FastScanUnaligned, TestUnalignedCodesAccess) {
    constexpr size_t alignment = 32;
    constexpr int nsq = 8;
    size_t code_size;
#ifdef __AVX512F__
    code_size = 512;
#else
    code_size = 256;
#endif
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<uint8_t> dist(
            std::numeric_limits<uint8_t>::min(),
            std::numeric_limits<uint8_t>::max());

    std::vector<uint8_t> aligned_buffer =
            create_aligned_buffer(code_size, alignment);
    // unaligned_buffer itself is aligned but it's the buffer for the unaligned
    // pointer.
    std::vector<uint8_t> unaligned_buffer =
            create_aligned_buffer(code_size, alignment);

    uint8_t* codes_aligned = get_aligned_ptr(aligned_buffer, alignment);
    uint8_t* codes_unaligned = get_unaligned_ptr(unaligned_buffer, alignment);

    for (size_t i = 0; i < code_size; i++) {
        uint8_t val = static_cast<uint8_t>(dist(gen));
        codes_aligned[i] = val;
        codes_unaligned[i] = val;
    }

    ASSERT_TRUE(faiss::is_aligned_pointer(codes_aligned));
    ASSERT_FALSE(faiss::is_aligned_pointer(codes_unaligned));

    constexpr int nq = 2;
    std::vector<uint8_t> LUT_aligned_buffer =
            create_aligned_buffer(nq * nsq * 16, alignment);
    uint8_t* LUT = get_aligned_ptr(LUT_aligned_buffer, alignment);

    for (size_t i = 0; i < nq * nsq * 16; i++) {
        LUT[i] = static_cast<uint8_t>(dist(gen));
    }

    // result_size is a subset of code_size, to not run off the unaligned buffer
    constexpr size_t result_size = 64;
    std::vector<uint16_t> accu_aligned(nq * result_size);
    std::vector<uint16_t> accu_unaligned(nq * result_size);

    faiss::accumulate_to_mem(
            nq, result_size, nsq, codes_aligned, LUT, accu_aligned.data());

    faiss::accumulate_to_mem(
            nq, result_size, nsq, codes_unaligned, LUT, accu_unaligned.data());

    // They should be the same, because codes_unaligned gets set to the same
    // values as aligned, it's just not aligned.
    ASSERT_EQ(accu_aligned, accu_unaligned)
            << "Aligned and unaligned code paths should produce identical results";

    posix_memalign_free(codes_aligned);
    posix_memalign_free(codes_unaligned - 1);
    posix_memalign_free(LUT);
}
