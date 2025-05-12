/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <random>
#include <vector>

#include <faiss/IndexBinaryFlat.h>
#include <faiss/IndexFlat.h>
#include <faiss/impl/io.h>
#include <faiss/index_io.h>

namespace {

std::vector<float> make_data(const size_t n, const size_t d, size_t seed) {
    std::vector<float> database(n * d);
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> distrib;

    for (size_t i = 0; i < n * d; i++) {
        database[i] = distrib(rng);
    }
    return database;
}

std::vector<uint8_t> make_binary_data(
        const size_t n,
        const size_t d,
        size_t seed) {
    std::vector<uint8_t> database(n * d);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<uint8_t> distrib(0, 255);

    for (size_t i = 0; i < n * d; i++) {
        database[i] = distrib(rng);
    }
    return database;
}

} // namespace

// the logic is the following:
//   1. generate two flatcodes-based indices, Index1 and Index2
//   2. serialize both indices into std::vector<> buffers, Buf1 and Buf2
//   3. save Buf1 into a temporary file, File1
//   4. deserialize Index1 using mmap feature on File1 into Index1MM
//   5. ensure that Index1MM acts as Index2 if we write the data from Buf2
//      on top of the existing File1
//   6. ensure that Index1MM acts as Index1 if we write the data from Buf1
//      on top of the existing File1 again

TEST(TestMmap, mmap_flatcodes) {
#ifdef _AIX
    GTEST_SKIP() << "Skipping test on AIX.";
#endif
    // generate data
    const size_t nt = 1000;
    const size_t nq = 10;
    const size_t d = 32;
    const size_t k = 25;

    std::vector<float> xt1 = make_data(nt, d, 123);
    std::vector<float> xt2 = make_data(nt, d, 456);
    std::vector<float> xq = make_data(nq, d, 789);

    // ensure that the data is different
    ASSERT_NE(xt1, xt2);

    // make index1 and create reference results
    faiss::IndexFlatL2 index1(d);
    index1.train(nt, xt1.data());
    index1.add(nt, xt1.data());

    std::vector<float> ref_dis_1(k * nq);
    std::vector<faiss::idx_t> ref_ids_1(k * nq);
    index1.search(nq, xq.data(), k, ref_dis_1.data(), ref_ids_1.data());

    // make index2 and create reference results
    faiss::IndexFlatL2 index2(d);
    index2.train(nt, xt2.data());
    index2.add(nt, xt2.data());

    std::vector<float> ref_dis_2(k * nq);
    std::vector<faiss::idx_t> ref_ids_2(k * nq);
    index2.search(nq, xq.data(), k, ref_dis_2.data(), ref_ids_2.data());

    // ensure that the results are different
    ASSERT_NE(ref_dis_1, ref_dis_2);
    ASSERT_NE(ref_ids_1, ref_ids_2);

    // serialize both in a form of vectors
    faiss::VectorIOWriter wr1;
    faiss::write_index(&index1, &wr1);

    faiss::VectorIOWriter wr2;
    faiss::write_index(&index2, &wr2);

    // generate a temporary file and write index1 into it
    std::string tmpname = std::tmpnam(nullptr);

    {
        std::ofstream ofs(tmpname);
        ofs.write((const char*)wr1.data.data(), wr1.data.size());
    }

    // create a mmap index
    std::unique_ptr<faiss::Index> index1mm(
            faiss::read_index(tmpname.c_str(), faiss::IO_FLAG_MMAP_IFC));

    ASSERT_NE(index1mm, nullptr);

    // perform a search
    std::vector<float> cand_dis_1(k * nq);
    std::vector<faiss::idx_t> cand_ids_1(k * nq);
    index1mm->search(nq, xq.data(), k, cand_dis_1.data(), cand_ids_1.data());

    // match vs ref1
    ASSERT_EQ(ref_ids_1, cand_ids_1);
    ASSERT_EQ(ref_dis_1, cand_dis_1);

    // ok now, overwrite the internals of the file without recreating it
    {
        std::ofstream ofs(tmpname);
        ofs.seekp(0, std::ios::beg);

        ofs.write((const char*)wr2.data.data(), wr2.data.size());
    }

    // perform a search
    std::vector<float> cand_dis_2(k * nq);
    std::vector<faiss::idx_t> cand_ids_2(k * nq);
    index1mm->search(nq, xq.data(), k, cand_dis_2.data(), cand_ids_2.data());

    // match vs ref1
    ASSERT_EQ(ref_ids_2, cand_ids_2);
    ASSERT_EQ(ref_dis_2, cand_dis_2);

    // write back data1
    {
        std::ofstream ofs(tmpname);
        ofs.seekp(0, std::ios::beg);

        ofs.write((const char*)wr1.data.data(), wr1.data.size());
    }

    // perform a search
    std::vector<float> cand_dis_3(k * nq);
    std::vector<faiss::idx_t> cand_ids_3(k * nq);
    index1mm->search(nq, xq.data(), k, cand_dis_3.data(), cand_ids_3.data());

    // match vs ref1
    ASSERT_EQ(ref_ids_1, cand_ids_3);
    ASSERT_EQ(ref_dis_1, cand_dis_3);
}

TEST(TestMmap, mmap_binary_flatcodes) {
#ifdef _AIX
    GTEST_SKIP() << "Skipping test on AIX.";
#endif
    // generate data
    const size_t nt = 1000;
    const size_t nq = 10;
    // in bits
    const size_t d = 64;
    // in bytes
    const size_t d8 = (d + 7) / 8;
    const size_t k = 25;

    std::vector<uint8_t> xt1 = make_binary_data(nt, d8, 123);
    std::vector<uint8_t> xt2 = make_binary_data(nt, d8, 456);
    std::vector<uint8_t> xq = make_binary_data(nq, d8, 789);

    // ensure that the data is different
    ASSERT_NE(xt1, xt2);

    // make index1 and create reference results
    faiss::IndexBinaryFlat index1(d);
    index1.train(nt, xt1.data());
    index1.add(nt, xt1.data());

    std::vector<int32_t> ref_dis_1(k * nq);
    std::vector<faiss::idx_t> ref_ids_1(k * nq);
    index1.search(nq, xq.data(), k, ref_dis_1.data(), ref_ids_1.data());

    // make index2 and create reference results
    faiss::IndexBinaryFlat index2(d);
    index2.train(nt, xt2.data());
    index2.add(nt, xt2.data());

    std::vector<int32_t> ref_dis_2(k * nq);
    std::vector<faiss::idx_t> ref_ids_2(k * nq);
    index2.search(nq, xq.data(), k, ref_dis_2.data(), ref_ids_2.data());

    // ensure that the results are different
    ASSERT_NE(ref_dis_1, ref_dis_2);
    ASSERT_NE(ref_ids_1, ref_ids_2);

    // serialize both in a form of vectors
    faiss::VectorIOWriter wr1;
    faiss::write_index_binary(&index1, &wr1);

    faiss::VectorIOWriter wr2;
    faiss::write_index_binary(&index2, &wr2);

    // generate a temporary file and write index1 into it
    std::string tmpname = std::tmpnam(nullptr);

    {
        std::ofstream ofs(tmpname);
        ofs.write((const char*)wr1.data.data(), wr1.data.size());
    }

    // create a mmap index
    std::unique_ptr<faiss::IndexBinary> index1mm(
            faiss::read_index_binary(tmpname.c_str(), faiss::IO_FLAG_MMAP_IFC));

    ASSERT_NE(index1mm, nullptr);

    // perform a search
    std::vector<int32_t> cand_dis_1(k * nq);
    std::vector<faiss::idx_t> cand_ids_1(k * nq);
    index1mm->search(nq, xq.data(), k, cand_dis_1.data(), cand_ids_1.data());

    // match vs ref1
    ASSERT_EQ(ref_ids_1, cand_ids_1);
    ASSERT_EQ(ref_dis_1, cand_dis_1);

    // ok now, overwrite the internals of the file without recreating it
    {
        std::ofstream ofs(tmpname);
        ofs.seekp(0, std::ios::beg);

        ofs.write((const char*)wr2.data.data(), wr2.data.size());
    }

    // perform a search
    std::vector<int32_t> cand_dis_2(k * nq);
    std::vector<faiss::idx_t> cand_ids_2(k * nq);
    index1mm->search(nq, xq.data(), k, cand_dis_2.data(), cand_ids_2.data());

    // match vs ref1
    ASSERT_EQ(ref_ids_2, cand_ids_2);
    ASSERT_EQ(ref_dis_2, cand_dis_2);

    // write back data1
    {
        std::ofstream ofs(tmpname);
        ofs.seekp(0, std::ios::beg);

        ofs.write((const char*)wr1.data.data(), wr1.data.size());
    }

    // perform a search
    std::vector<int32_t> cand_dis_3(k * nq);
    std::vector<faiss::idx_t> cand_ids_3(k * nq);
    index1mm->search(nq, xq.data(), k, cand_dis_3.data(), cand_ids_3.data());

    // match vs ref1
    ASSERT_EQ(ref_ids_1, cand_ids_3);
    ASSERT_EQ(ref_dis_1, cand_dis_3);
}
