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

#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIDMap.h>
#include <faiss/IndexRaBitQ.h>
#include <faiss/impl/FaissException.h>
#include <faiss/impl/io.h>
#include <faiss/impl/zerocopy_io.h>
#include <faiss/index_factory.h>
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
    const size_t nt = 1000;
    const size_t nq = 10;
    const size_t d = 32;
    const size_t k = 25;

    std::vector<float> xt1 = make_data(nt, d, 123);
    std::vector<float> xt2 = make_data(nt, d, 456);
    std::vector<float> xq = make_data(nq, d, 789);

    faiss::IndexFlatL2 index1(d);
    index1.add(nt, xt1.data());

    std::vector<float> ref_dis_1(k * nq);
    std::vector<faiss::idx_t> ref_ids_1(k * nq);
    index1.search(nq, xq.data(), k, ref_dis_1.data(), ref_ids_1.data());

    faiss::IndexFlatL2 index2(d);
    index2.add(nt, xt2.data());

    std::vector<float> ref_dis_2(k * nq);
    std::vector<faiss::idx_t> ref_ids_2(k * nq);
    index2.search(nq, xq.data(), k, ref_dis_2.data(), ref_ids_2.data());

    ASSERT_NE(ref_dis_1, ref_dis_2);
    ASSERT_NE(ref_ids_1, ref_ids_2);

    faiss::VectorIOWriter wr1;
    faiss::write_index(&index1, &wr1);

    faiss::VectorIOWriter wr2;
    faiss::write_index(&index2, &wr2);

    ASSERT_EQ(wr1.data.size(), wr2.data.size());

    std::vector<uint8_t> buffer = wr1.data;

    faiss::ZeroCopyIOReader reader(buffer.data(), buffer.size());
    auto index1zc = faiss::read_index_up(&reader);

    ASSERT_NE(index1zc, nullptr);

    std::vector<float> cand_dis_1(k * nq);
    std::vector<faiss::idx_t> cand_ids_1(k * nq);
    index1zc->search(nq, xq.data(), k, cand_dis_1.data(), cand_ids_1.data());

    ASSERT_EQ(ref_ids_1, cand_ids_1);
    ASSERT_EQ(ref_dis_1, cand_dis_1);

    // overwrite buffer without moving it
    for (size_t i = 0; i < buffer.size(); i++) {
        buffer[i] = wr2.data[i];
    }

    std::vector<float> cand_dis_2(k * nq);
    std::vector<faiss::idx_t> cand_ids_2(k * nq);
    index1zc->search(nq, xq.data(), k, cand_dis_2.data(), cand_ids_2.data());

    ASSERT_EQ(ref_ids_2, cand_ids_2);
    ASSERT_EQ(ref_dis_2, cand_dis_2);
}

namespace {

// Deserialize a rabitq index with a copying reader (owns its codes) and with
// the zero-copy reader (codes view the buffer); both must search like the
// original, and the zero-copy one follows an in-place buffer overwrite.
void check_rabitq_zerocopy(const uint8_t nb_bits) {
    const size_t nt = 500;
    const size_t nq = 10;
    const size_t d = 32;
    const size_t k = 10;

    const auto index1 = make_rabitq_index(nt, d, nb_bits, 123);
    const std::vector<float> xq = make_data(nq, d, 789);

    std::vector<float> ref_dis_1(k * nq);
    std::vector<faiss::idx_t> ref_ids_1(k * nq);
    index1->search(nq, xq.data(), k, ref_dis_1.data(), ref_ids_1.data());

    faiss::VectorIOWriter wr1;
    faiss::write_index(index1.get(), &wr1);

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

    ASSERT_EQ(index1zc_rq->codes.size(), index1->codes.size());
    ASSERT_TRUE(
            std::equal(
                    index1zc_rq->codes.data(),
                    index1zc_rq->codes.data() + index1zc_rq->codes.size(),
                    index1->codes.data()));

    std::vector<float> cand_dis_zc(k * nq);
    std::vector<faiss::idx_t> cand_ids_zc(k * nq);
    index1zc->search(nq, xq.data(), k, cand_dis_zc.data(), cand_ids_zc.data());

    std::vector<float> cand_dis_c(k * nq);
    std::vector<faiss::idx_t> cand_ids_c(k * nq);
    index1c->search(nq, xq.data(), k, cand_dis_c.data(), cand_ids_c.data());

    ASSERT_EQ(ref_ids_1, cand_ids_zc);
    ASSERT_EQ(ref_dis_1, cand_dis_zc);
    ASSERT_EQ(ref_ids_1, cand_ids_c);
    ASSERT_EQ(ref_dis_1, cand_dis_c);

    // overwrite a code byte without moving the buffer
    const size_t codes_offset = index1zc_rq->codes.data() - buffer.data();
    buffer[codes_offset] ^= 0xFF;

    ASSERT_EQ(index1zc_rq->codes[0], buffer[codes_offset]);
}

} // namespace

TEST(TestZeroCopy, zerocopy_rabitq_single_bit) {
    check_rabitq_zerocopy(1);
}

TEST(TestZeroCopy, zerocopy_rabitq_multi_bit) {
    check_rabitq_zerocopy(4);
}

namespace {

template <typename T>
bool views_into(
        const faiss::MaybeOwnedVector<T>& vec,
        const std::vector<uint8_t>& buffer) {
    return !vec.is_owned &&
            points_into(
                    reinterpret_cast<const uint8_t*>(vec.data()),
                    vec.byte_size(),
                    buffer);
}

} // namespace

// The id map and the HNSW levels and offsets view the buffer, like the
// neighbors and codes do.
TEST(TestZeroCopy, zerocopy_idmap_hnsw) {
    const size_t nt = 1000;
    const size_t nq = 10;
    const size_t d = 32;
    const size_t k = 10;

    const std::vector<float> xt = make_data(nt, d, 123);
    const std::vector<float> xq = make_data(nq, d, 789);
    std::vector<faiss::idx_t> ids(nt);
    for (size_t i = 0; i < nt; i++) {
        ids[i] = 1000 + 7 * i;
    }

    std::unique_ptr<faiss::Index> index(
            faiss::index_factory(d, "IDMap,HNSW32,SQ8"));
    index->train(nt, xt.data());
    index->add_with_ids(nt, xt.data(), ids.data());

    std::vector<float> ref_dis(k * nq);
    std::vector<faiss::idx_t> ref_ids(k * nq);
    index->search(nq, xq.data(), k, ref_dis.data(), ref_ids.data());

    faiss::VectorIOWriter wr;
    faiss::write_index(index.get(), &wr);
    const std::vector<uint8_t> buffer = wr.data;

    faiss::ZeroCopyIOReader reader(buffer.data(), buffer.size());
    auto indexzc = faiss::read_index_up(&reader);
    auto* idmap = dynamic_cast<faiss::IndexIDMap*>(indexzc.get());
    ASSERT_NE(idmap, nullptr);
    auto* hnsw = dynamic_cast<faiss::IndexHNSW*>(idmap->index);
    ASSERT_NE(hnsw, nullptr);

    EXPECT_TRUE(views_into(idmap->id_map, buffer));
    EXPECT_TRUE(views_into(hnsw->hnsw.levels, buffer));
    EXPECT_TRUE(views_into(hnsw->hnsw.offsets, buffer));
    EXPECT_TRUE(views_into(hnsw->hnsw.neighbors, buffer));
    EXPECT_EQ(
            std::vector<faiss::idx_t>(
                    idmap->id_map.begin(), idmap->id_map.end()),
            ids);

    std::vector<float> cand_dis(k * nq);
    std::vector<faiss::idx_t> cand_ids(k * nq);
    indexzc->search(nq, xq.data(), k, cand_dis.data(), cand_ids.data());
    EXPECT_EQ(ref_ids, cand_ids);
    EXPECT_EQ(ref_dis, cand_dis);
}

// Operations that need to own the data throw on a view.
TEST(TestZeroCopy, view_mutation_throws) {
    std::vector<int> data(4);
    auto view = faiss::MaybeOwnedVector<int>::create_view(
            data.data(), data.size(), nullptr);
    EXPECT_THROW(view.push_back(0), faiss::FaissException);
    EXPECT_THROW(view.resize(8), faiss::FaissException);
    EXPECT_THROW(view.clear(), faiss::FaissException);
    EXPECT_EQ(view.size(), data.size());
}
