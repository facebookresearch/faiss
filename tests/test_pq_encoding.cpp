/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include <faiss/IndexPQFastScan.h>
#include <faiss/impl/ProductQuantizer.h>
#include <faiss/impl/fast_scan/fast_scan.h>

namespace {

const std::vector<uint64_t> random_vector(size_t s) {
    std::vector<uint64_t> v(s, 0);
    for (size_t i = 0; i < s; ++i) {
        v[i] = rand();
    }

    return v;
}

const std::vector<float> random_vector_float(size_t s) {
    std::vector<float> v(s, 0);
    for (size_t i = 0; i < s; ++i) {
        v[i] = rand();
    }

    return v;
}

} // namespace

TEST(PQEncoderGeneric, encode) {
    const int nsubcodes = 97;
    const int minbits = 1;
    const int maxbits = 24;
    const std::vector<uint64_t> values = random_vector(nsubcodes);

    for (int nbits = minbits; nbits <= maxbits; ++nbits) {
        SCOPED_TRACE(nbits);

        const uint64_t mask = (1ull << nbits) - 1;
        std::unique_ptr<uint8_t[]> codes(
                new uint8_t[(nsubcodes * maxbits + 7) / 8]);

        // Scope so the encoder destructor flushes the trailing bits.
        {
            faiss::PQEncoderGeneric encoder(codes.get(), nbits);
            for (const auto& v : values) {
                encoder.encode(v & mask);
            }
        }

        faiss::PQDecoderGeneric decoder(codes.get(), nbits);
        for (int i = 0; i < nsubcodes; ++i) {
            uint64_t v = decoder.decode();
            EXPECT_EQ(values[i] & mask, v);
        }
    }
}

TEST(PQFastScan, set_packed_element) {
    int d = 20, ntotal = 1000, M = 5, nbits = 4;
    const std::vector<float> ds = random_vector_float(ntotal * d);
    faiss::IndexPQFastScan index(d, M, nbits);
    index.train(ntotal, ds.data());
    index.add(ntotal, ds.data());

    for (int j = 0; j < 10; j++) {
        int vector_id = rand() % ntotal;
        std::vector<uint8_t> old(ntotal * M);
        std::vector<uint8_t> code(M);
        for (int i = 0; i < ntotal; i++) {
            for (int sq = 0; sq < M; sq++) {
                old[i * M + sq] = faiss::pq4_get_packed_element(
                        index.codes.data(), index.bbs, M, i, sq);
            }
        }
        for (int sq = 0; sq < M; sq++) {
            faiss::pq4_set_packed_element(
                    index.codes.data(),
                    ((old[vector_id * M + sq] + 3) % 16),
                    index.bbs,
                    M,
                    vector_id,
                    sq);
        }
        for (int i = 0; i < ntotal; i++) {
            for (int sq = 0; sq < M; sq++) {
                uint8_t newcode = faiss::pq4_get_packed_element(
                        index.codes.data(), index.bbs, M, i, sq);
                uint8_t oldcode = old[i * M + sq];
                if (i == vector_id) {
                    EXPECT_EQ(newcode, (oldcode + 3) % 16);
                } else {
                    EXPECT_EQ(newcode, oldcode);
                }
            }
        }
    }
}
