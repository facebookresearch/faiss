/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
#include <vector>

#include <faiss/IndexBinaryFlat.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexRaBitQ.h>
#include <faiss/impl/io.h>
#include <faiss/impl/zerocopy_io.h>
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
    std::uniform_int_distribution<int> distrib(0, 255);

    for (size_t i = 0; i < n * d; i++) {
        database[i] = distrib(rng);
    }
    return database;
}

std::unique_ptr<faiss::IndexRaBitQ> make_rabitq_index(
        const size_t nt,
        const size_t d,
        const uint8_t nb_bits,
        const size_t seed) {
    auto index =
            std::make_unique<faiss::IndexRaBitQ>(d, faiss::METRIC_L2, nb_bits);
    index->qb = 4;

    const std::vector<float> xt = make_data(nt, d, seed);
    index->train(nt, xt.data());
    index->add(nt, xt.data());
    return index;
}

// true if [begin, begin + size) lies inside the buffer
bool points_into(
        const uint8_t* begin,
        const size_t size,
        const std::vector<uint8_t>& buffer) {
    return begin >= buffer.data() &&
            begin + size <= buffer.data() + buffer.size();
}

} // namespace

// the logic is the following:
//   1. generate two flatcodes-based indices, Index1 and Index2
//   2. serialize both indices into std::vector<> buffers, Buf1 and Buf2
//   3. deserialize Index1 using zero-copy feature on Buf1 into Index1ZC
//   4. ensure that Index1ZC acts as Index2 if we write the data from Buf2
//      on top of the existing Buf1

TEST(TestZeroCopy, zerocopy_flatcodes) {
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

    ASSERT_EQ(wr1.data.size(), wr2.data.size());

    // clone a buffer
    std::vector<uint8_t> buffer = wr1.data;

    // create a zero-copy index
    faiss::ZeroCopyIOReader reader(buffer.data(), buffer.size());
    auto index1zc = faiss::read_index_up(&reader);

    ASSERT_NE(index1zc, nullptr);

    // perform a search
    std::vector<float> cand_dis_1(k * nq);
    std::vector<faiss::idx_t> cand_ids_1(k * nq);
    index1zc->search(nq, xq.data(), k, cand_dis_1.data(), cand_ids_1.data());

    // match vs ref1
    ASSERT_EQ(ref_ids_1, cand_ids_1);
    ASSERT_EQ(ref_dis_1, cand_dis_1);

    // overwrite buffer without moving it
    for (size_t i = 0; i < buffer.size(); i++) {
        buffer[i] = wr2.data[i];
    }

    // perform a search
    std::vector<float> cand_dis_2(k * nq);
    std::vector<faiss::idx_t> cand_ids_2(k * nq);
    index1zc->search(nq, xq.data(), k, cand_dis_2.data(), cand_ids_2.data());

    // match vs ref2
    ASSERT_EQ(ref_ids_2, cand_ids_2);
    ASSERT_EQ(ref_dis_2, cand_dis_2);

    // overwrite again
    for (size_t i = 0; i < buffer.size(); i++) {
        buffer[i] = wr1.data[i];
    }

    // perform a search
    std::vector<float> cand_dis_3(k * nq);
    std::vector<faiss::idx_t> cand_ids_3(k * nq);
    index1zc->search(nq, xq.data(), k, cand_dis_3.data(), cand_ids_3.data());

    // match vs ref1
    ASSERT_EQ(ref_ids_1, cand_ids_3);
    ASSERT_EQ(ref_dis_1, cand_dis_3);
}

TEST(TestZeroCopy, zerocopy_binary_flatcodes) {
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

    ASSERT_EQ(wr1.data.size(), wr2.data.size());

    // clone a buffer
    std::vector<uint8_t> buffer = wr1.data;

    // create a zero-copy index
    faiss::ZeroCopyIOReader reader(buffer.data(), buffer.size());
    auto index1zc = faiss::read_index_binary_up(&reader);

    ASSERT_NE(index1zc, nullptr);

    // perform a search
    std::vector<int32_t> cand_dis_1(k * nq);
    std::vector<faiss::idx_t> cand_ids_1(k * nq);
    index1zc->search(nq, xq.data(), k, cand_dis_1.data(), cand_ids_1.data());

    // match vs ref1
    ASSERT_EQ(ref_ids_1, cand_ids_1);
    ASSERT_EQ(ref_dis_1, cand_dis_1);

    // overwrite buffer without moving it
    for (size_t i = 0; i < buffer.size(); i++) {
        buffer[i] = wr2.data[i];
    }

    // perform a search
    std::vector<int32_t> cand_dis_2(k * nq);
    std::vector<faiss::idx_t> cand_ids_2(k * nq);
    index1zc->search(nq, xq.data(), k, cand_dis_2.data(), cand_ids_2.data());

    // match vs ref2
    ASSERT_EQ(ref_ids_2, cand_ids_2);
    ASSERT_EQ(ref_dis_2, cand_dis_2);

    // overwrite again
    for (size_t i = 0; i < buffer.size(); i++) {
        buffer[i] = wr1.data[i];
    }

    // perform a search
    std::vector<int32_t> cand_dis_3(k * nq);
    std::vector<faiss::idx_t> cand_ids_3(k * nq);
    index1zc->search(nq, xq.data(), k, cand_dis_3.data(), cand_ids_3.data());

    // match vs ref1
    ASSERT_EQ(ref_ids_1, cand_ids_3);
    ASSERT_EQ(ref_dis_1, cand_dis_3);
}

namespace {

// the logic is the following:
//   1. generate a rabitq index, Index1, and create reference results
//   2. serialize it into a std::vector<> buffer, Buf1
//   3. deserialize Buf1 twice: with a copying reader into Index1C, and with
//      the zero-copy feature into Index1ZC
//   4. ensure that Index1C owns its codes while Index1ZC views Buf1
//   5. ensure that both act as Index1
//   6. ensure that Index1ZC follows Buf1 if we overwrite a code byte in
//      place, while Index1C does not
void check_rabitq_zerocopy(const uint8_t nb_bits) {
    // generate data
    const size_t nt = 500;
    const size_t nq = 10;
    const size_t d = 32;
    const size_t k = 10;

    const auto index1 = make_rabitq_index(nt, d, nb_bits, 123);
    const std::vector<float> xq = make_data(nq, d, 789);

    // create reference results
    std::vector<float> ref_dis_1(k * nq);
    std::vector<faiss::idx_t> ref_ids_1(k * nq);
    index1->search(nq, xq.data(), k, ref_dis_1.data(), ref_ids_1.data());

    // serialize in a form of a vector
    faiss::VectorIOWriter wr1;
    faiss::write_index(index1.get(), &wr1);

    // clone a buffer
    std::vector<uint8_t> buffer = wr1.data;

    // create a copying index, it owns its codes
    faiss::VectorIOReader copy_reader;
    copy_reader.data = buffer;
    auto index1c = faiss::read_index_up(&copy_reader);
    auto* index1c_rq = dynamic_cast<faiss::IndexRaBitQ*>(index1c.get());
    ASSERT_NE(index1c_rq, nullptr);
    ASSERT_TRUE(index1c_rq->codes.is_owned);

    // create a zero-copy index, its codes view the buffer
    faiss::ZeroCopyIOReader reader(buffer.data(), buffer.size());
    auto index1zc = faiss::read_index_up(&reader);
    auto* index1zc_rq = dynamic_cast<faiss::IndexRaBitQ*>(index1zc.get());
    ASSERT_NE(index1zc_rq, nullptr);
    ASSERT_FALSE(index1zc_rq->codes.is_owned);
    ASSERT_TRUE(points_into(
            index1zc_rq->codes.data(), index1zc_rq->codes.size(), buffer));

    // match the remaining fields vs index1
    ASSERT_EQ(index1zc_rq->d, index1->d);
    ASSERT_EQ(index1zc_rq->ntotal, index1->ntotal);
    ASSERT_EQ(index1zc_rq->metric_type, index1->metric_type);
    ASSERT_EQ(index1zc_rq->code_size, index1->code_size);
    ASSERT_EQ(index1zc_rq->qb, index1->qb);
    ASSERT_EQ(index1zc_rq->rabitq.nb_bits, index1->rabitq.nb_bits);
    ASSERT_EQ(index1zc_rq->center, index1->center);
    ASSERT_EQ(index1zc_rq->codes.size(), index1->codes.size());
    ASSERT_TRUE(
            std::equal(
                    index1zc_rq->codes.data(),
                    index1zc_rq->codes.data() + index1zc_rq->codes.size(),
                    index1->codes.data()));

    // perform a search on both
    std::vector<float> cand_dis_zc(k * nq);
    std::vector<faiss::idx_t> cand_ids_zc(k * nq);
    index1zc->search(nq, xq.data(), k, cand_dis_zc.data(), cand_ids_zc.data());

    std::vector<float> cand_dis_c(k * nq);
    std::vector<faiss::idx_t> cand_ids_c(k * nq);
    index1c->search(nq, xq.data(), k, cand_dis_c.data(), cand_ids_c.data());

    // match vs ref1
    ASSERT_EQ(ref_ids_1, cand_ids_zc);
    ASSERT_EQ(ref_dis_1, cand_dis_zc);
    ASSERT_EQ(ref_ids_1, cand_ids_c);
    ASSERT_EQ(ref_dis_1, cand_dis_c);

    // overwrite a code byte without moving the buffer
    const size_t codes_offset = index1zc_rq->codes.data() - buffer.data();
    buffer[codes_offset] ^= 0xFF;

    // the zero-copy index follows the buffer, the copying one does not
    ASSERT_EQ(index1zc_rq->codes[0], buffer[codes_offset]);
    ASSERT_NE(index1c_rq->codes[0], index1zc_rq->codes[0]);
}

} // namespace

TEST(TestZeroCopy, zerocopy_rabitq_single_bit) {
    check_rabitq_zerocopy(1);
}

TEST(TestZeroCopy, zerocopy_rabitq_multi_bit) {
    check_rabitq_zerocopy(4);
}
