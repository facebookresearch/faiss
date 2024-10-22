/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// This test was designed to be run using valgrind or ASAN to test the
// correctness of memory accesses.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>

#include <faiss/utils/hamming.h>

#include <faiss/cppcontrib/detail/UintReader.h>

template <intptr_t N_ELEMENTS, intptr_t CODE_BITS, intptr_t CPOS>
struct TestLoop {
    static void test(
            const uint8_t* const container,
            faiss::BitstringReader& br) {
        // validate
        const intptr_t uintreader_data = faiss::cppcontrib::detail::
                UintReaderRaw<N_ELEMENTS, CODE_BITS, CPOS>::get(container);
        const intptr_t bitstringreader_data = br.read(CODE_BITS);

        ASSERT_EQ(uintreader_data, bitstringreader_data)
                << "Mismatch between BitstringReader (" << bitstringreader_data
                << ") and UintReader (" << uintreader_data
                << ") for N_ELEMENTS=" << N_ELEMENTS
                << ", CODE_BITS=" << CODE_BITS << ", CPOS=" << CPOS;

        //
        TestLoop<N_ELEMENTS, CODE_BITS, CPOS + 1>::test(container, br);
    }
};

template <intptr_t N_ELEMENTS, intptr_t CODE_BITS>
struct TestLoop<N_ELEMENTS, CODE_BITS, N_ELEMENTS> {
    static void test(
            const uint8_t* const container,
            faiss::BitstringReader& br) {}
};

template <intptr_t N_ELEMENTS, intptr_t CODE_BITS>
void TestUintReader() {
    constexpr intptr_t CODE_BYTES = (CODE_BITS * N_ELEMENTS + 7) / 8;

    std::default_random_engine rng;
    std::uniform_int_distribution<uint64_t> u(0, 1 << CODE_BITS);

    // do several attempts
    for (size_t attempt = 0; attempt < 10; attempt++) {
        // allocate a buffer. This way, not std::vector
        std::unique_ptr<uint8_t[]> container(new uint8_t[CODE_BYTES]);
        // make it empty
        for (size_t i = 0; i < CODE_BYTES; i++) {
            container.get()[i] = 0;
        }

        // populate it
        faiss::BitstringWriter bw(container.get(), CODE_BYTES);
        for (size_t i = 0; i < N_ELEMENTS; i++) {
            bw.write(u(rng), CODE_BITS);
        }

        // read it back and verify against bitreader
        faiss::BitstringReader br(container.get(), CODE_BYTES);

        TestLoop<N_ELEMENTS, CODE_BITS, 0>::test(container.get(), br);
    }
}

template <intptr_t CODE_BITS>
void TestUintReaderBits() {
    TestUintReader<1, CODE_BITS>();
    TestUintReader<2, CODE_BITS>();
    TestUintReader<3, CODE_BITS>();
    TestUintReader<4, CODE_BITS>();
    TestUintReader<5, CODE_BITS>();
    TestUintReader<6, CODE_BITS>();
    TestUintReader<7, CODE_BITS>();
    TestUintReader<8, CODE_BITS>();
    TestUintReader<9, CODE_BITS>();
    TestUintReader<10, CODE_BITS>();
    TestUintReader<11, CODE_BITS>();
    TestUintReader<12, CODE_BITS>();
    TestUintReader<13, CODE_BITS>();
    TestUintReader<14, CODE_BITS>();
    TestUintReader<15, CODE_BITS>();
    TestUintReader<16, CODE_BITS>();
    TestUintReader<17, CODE_BITS>();
}

TEST(testCppcontribUintreader, Test8bit) {
    TestUintReaderBits<8>();
}

TEST(testCppcontribUintreader, Test10bit) {
    TestUintReaderBits<10>();
}

TEST(testCppcontribUintreader, Test12bit) {
    TestUintReaderBits<12>();
}

TEST(testCppcontribUintreader, Test16bit) {
    TestUintReaderBits<16>();
}
