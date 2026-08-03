/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/IndexSQFastScan.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/impl/io.h>
#include <faiss/index_factory.h>
#include <faiss/index_io.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/random.h>

namespace {

using SQ = faiss::ScalarQuantizer;

// --- Helpers ---------------------------------------------------------------

std::vector<float> make_random_vectors(size_t n, size_t d, int seed = 1234) {
    std::vector<float> x(n * d);
    faiss::float_randn(x.data(), x.size(), seed);
    return x;
}

/// Recall@1: fraction of queries whose ground-truth NN is the top result.
float recall_at_1(
        const faiss::idx_t* I_gt,
        const faiss::idx_t* I_test,
        size_t nq,
        size_t k_gt,
        size_t k_test) {
    int hits = 0;
    for (size_t i = 0; i < nq; i++) {
        if (I_gt[i * k_gt] == I_test[i * k_test]) {
            hits++;
        }
    }
    return (float)hits / nq;
}

// Native 4-bit types (only supported types for IndexSQFastScan)
static const std::vector<SQ::QuantizerType> kNative4bit = {
        SQ::QT_4bit,
        SQ::QT_4bit_uniform,
};

} // namespace

// ---------------------------------------------------------------------------
// Construction
// ---------------------------------------------------------------------------

TEST(IndexSQFastScan, ConstructNative4bit) {
    const int d = 64;
    for (auto qtype : kNative4bit) {
        SCOPED_TRACE(qtype);
        faiss::IndexSQFastScan index(d, qtype);
        EXPECT_EQ(index.d, d);
        EXPECT_FALSE(index.is_trained);
        EXPECT_EQ(index.ntotal, 0);
    }
}

TEST(IndexSQFastScan, RejectsNon4bitTypes) {
    const int d = 64;
    EXPECT_THROW(faiss::IndexSQFastScan(d, SQ::QT_8bit), faiss::FaissException);
    EXPECT_THROW(faiss::IndexSQFastScan(d, SQ::QT_6bit), faiss::FaissException);
    EXPECT_THROW(faiss::IndexSQFastScan(d, SQ::QT_fp16), faiss::FaissException);
    EXPECT_THROW(faiss::IndexSQFastScan(d, SQ::QT_bf16), faiss::FaissException);
}

TEST(IndexSQFastScan, DefaultConstructor) {
    faiss::IndexSQFastScan index;
    EXPECT_EQ(index.d, 0);
    EXPECT_EQ(index.ntotal, 0);
}

// ---------------------------------------------------------------------------
// Train / Add / Search
// ---------------------------------------------------------------------------

TEST(IndexSQFastScan, TrainAddSearch) {
    const int d = 64;
    const int nb = 5000;
    const int nq = 50;
    const int k = 10;
    auto xb = make_random_vectors(nb, d, 42);
    auto xq = make_random_vectors(nq, d, 43);
    auto xt = make_random_vectors(2000, d, 44);

    for (auto qtype : kNative4bit) {
        SCOPED_TRACE(qtype);
        faiss::IndexSQFastScan index(d, qtype);
        index.train(xt.size() / d, xt.data());
        EXPECT_TRUE(index.is_trained);

        index.add(nb, xb.data());
        EXPECT_EQ(index.ntotal, nb);

        std::vector<float> D(nq * k);
        std::vector<faiss::idx_t> I(nq * k);
        index.search(nq, xq.data(), k, D.data(), I.data());

        for (int q = 0; q < nq; q++) {
            EXPECT_GE(I[q * k], 0);
        }

        for (int q = 0; q < nq; q++) {
            for (int j = 0; j < k; j++) {
                EXPECT_TRUE(std::isfinite(D[q * k + j]));
            }
            for (int j = 1; j < k; j++) {
                EXPECT_LE(D[q * k + j - 1], D[q * k + j] + 1e-5f);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Recall parity with IndexScalarQuantizer
// ---------------------------------------------------------------------------

TEST(IndexSQFastScan, RecallParityNative4bit) {
    const int d = 64;
    const int nb = 10000;
    const int nq = 100;
    const int k = 10;
    auto xb = make_random_vectors(nb, d, 42);
    auto xq = make_random_vectors(nq, d, 43);
    auto xt = make_random_vectors(2000, d, 44);

    faiss::IndexFlatL2 gt(d);
    gt.add(nb, xb.data());
    std::vector<float> D_gt(nq * k);
    std::vector<faiss::idx_t> I_gt(nq * k);
    gt.search(nq, xq.data(), k, D_gt.data(), I_gt.data());

    for (auto qtype : kNative4bit) {
        SCOPED_TRACE(qtype);
        faiss::IndexScalarQuantizer sq(d, qtype);
        sq.train(xt.size() / d, xt.data());
        sq.add(nb, xb.data());
        std::vector<float> D_sq(nq * k);
        std::vector<faiss::idx_t> I_sq(nq * k);
        sq.search(nq, xq.data(), k, D_sq.data(), I_sq.data());

        faiss::IndexSQFastScan fs(d, qtype);
        fs.train(xt.size() / d, xt.data());
        fs.add(nb, xb.data());
        std::vector<float> D_fs(nq * k);
        std::vector<faiss::idx_t> I_fs(nq * k);
        fs.search(nq, xq.data(), k, D_fs.data(), I_fs.data());

        float r_sq = recall_at_1(I_gt.data(), I_sq.data(), nq, k, k);
        float r_fs = recall_at_1(I_gt.data(), I_fs.data(), nq, k, k);
        EXPECT_NEAR(r_sq, r_fs, 0.05f);
    }
}

// ---------------------------------------------------------------------------
// Conversion constructor
// ---------------------------------------------------------------------------

TEST(IndexSQFastScan, ConversionConstructor) {
    const int d = 64;
    const int nb = 5000;
    const int nq = 50;
    const int k = 10;
    auto xb = make_random_vectors(nb, d, 42);
    auto xq = make_random_vectors(nq, d, 43);
    auto xt = make_random_vectors(2000, d, 44);

    for (auto qtype : kNative4bit) {
        SCOPED_TRACE(qtype);
        faiss::IndexScalarQuantizer sq(d, qtype);
        sq.train(xt.size() / d, xt.data());
        sq.add(nb, xb.data());

        faiss::IndexSQFastScan fs(sq);
        EXPECT_EQ(fs.ntotal, sq.ntotal);
        EXPECT_TRUE(fs.is_trained);

        std::vector<float> D(nq * k);
        std::vector<faiss::idx_t> I(nq * k);
        fs.search(nq, xq.data(), k, D.data(), I.data());

        for (int q = 0; q < nq; q++) {
            EXPECT_GE(I[q * k], 0);
            EXPECT_TRUE(std::isfinite(D[q * k]));
        }
    }
}

TEST(IndexSQFastScan, ConversionRejectsNon4bit) {
    const int d = 64;
    auto xt = make_random_vectors(1000, d, 44);

    faiss::IndexScalarQuantizer sq_idx(d, SQ::QT_8bit);
    sq_idx.train(xt.size() / d, xt.data());

    EXPECT_THROW({ faiss::IndexSQFastScan fs(sq_idx); }, faiss::FaissException);
}

// ---------------------------------------------------------------------------
// Reset
// ---------------------------------------------------------------------------

TEST(IndexSQFastScan, Reset) {
    const int d = 64;
    const int nb = 500;
    auto xb = make_random_vectors(nb, d, 42);
    auto xt = make_random_vectors(1000, d, 44);

    for (auto qtype : kNative4bit) {
        SCOPED_TRACE(qtype);
        faiss::IndexSQFastScan index(d, qtype);
        index.train(xt.size() / d, xt.data());
        index.add(nb, xb.data());
        EXPECT_EQ(index.ntotal, nb);

        index.reset();
        EXPECT_EQ(index.ntotal, 0);

        index.add(nb, xb.data());
        EXPECT_EQ(index.ntotal, nb);
    }
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

TEST(IndexSQFastScan, KEquals1) {
    const int d = 32;
    const int nb = 1000;
    const int nq = 10;
    auto xb = make_random_vectors(nb, d, 42);
    auto xq = make_random_vectors(nq, d, 43);

    for (auto qtype : kNative4bit) {
        SCOPED_TRACE(qtype);
        faiss::IndexSQFastScan index(d, qtype);
        index.train(nb, xb.data());
        index.add(nb, xb.data());

        std::vector<float> D(nq);
        std::vector<faiss::idx_t> I(nq);
        index.search(nq, xq.data(), 1, D.data(), I.data());

        for (int q = 0; q < nq; q++) {
            EXPECT_GE(I[q], 0);
            EXPECT_TRUE(std::isfinite(D[q]));
        }
    }
}

TEST(IndexSQFastScan, SingleVector) {
    const int d = 32;
    auto xb = make_random_vectors(1, d, 42);

    for (auto qtype : kNative4bit) {
        SCOPED_TRACE(qtype);
        faiss::IndexSQFastScan index(d, qtype);
        index.train(1, xb.data());
        index.add(1, xb.data());

        std::vector<float> D(1);
        std::vector<faiss::idx_t> I(1);
        index.search(1, xb.data(), 1, D.data(), I.data());

        EXPECT_EQ(I[0], 0);
        EXPECT_LT(D[0], 1.0f);
    }
}

TEST(IndexSQFastScan, OddDimensions) {
    const int nb = 500;
    const int nq = 5;
    const int k = 10;

    for (int d : {7, 9, 15, 17, 31, 33, 63, 65}) {
        SCOPED_TRACE(d);
        auto xb = make_random_vectors(nb, d, 42);
        auto xq = make_random_vectors(nq, d, 43);

        faiss::IndexSQFastScan index(d, SQ::QT_4bit);
        index.train(nb, xb.data());
        index.add(nb, xb.data());

        std::vector<float> D(nq * k);
        std::vector<faiss::idx_t> I(nq * k);
        index.search(nq, xq.data(), k, D.data(), I.data());

        for (int q = 0; q < nq; q++) {
            EXPECT_GE(I[q * k], 0);
        }
    }
}

TEST(IndexSQFastScan, ZeroVectors) {
    const int d = 32;
    const int nb = 100;
    std::vector<float> xb(nb * d, 0.0f);
    std::vector<float> xq(d, 0.0f);

    for (auto qtype : kNative4bit) {
        SCOPED_TRACE(qtype);
        faiss::IndexSQFastScan index(d, qtype);
        index.train(nb, xb.data());
        index.add(nb, xb.data());

        std::vector<float> D(10);
        std::vector<faiss::idx_t> I(10);
        index.search(1, xq.data(), 10, D.data(), I.data());

        for (int j = 0; j < 10; j++) {
            EXPECT_TRUE(std::isfinite(D[j]));
        }
    }
}

// ---------------------------------------------------------------------------
// Inner product
// ---------------------------------------------------------------------------

TEST(IndexSQFastScan, InnerProduct) {
    const int d = 64;
    const int nb = 5000;
    const int nq = 50;
    const int k = 10;
    auto xb = make_random_vectors(nb, d, 42);
    auto xq = make_random_vectors(nq, d, 43);
    faiss::fvec_renorm_L2(d, nb, xb.data());
    faiss::fvec_renorm_L2(d, nq, xq.data());

    for (auto qtype : kNative4bit) {
        SCOPED_TRACE(qtype);
        faiss::IndexSQFastScan index(d, qtype, faiss::METRIC_INNER_PRODUCT);
        index.train(nb, xb.data());
        index.add(nb, xb.data());

        std::vector<float> D(nq * k);
        std::vector<faiss::idx_t> I(nq * k);
        index.search(nq, xq.data(), k, D.data(), I.data());

        for (int q = 0; q < nq; q++) {
            EXPECT_GE(I[q * k], 0);
            EXPECT_TRUE(std::isfinite(D[q * k]));
            for (int j = 1; j < k; j++) {
                EXPECT_GE(D[q * k + j - 1] + 1e-5f, D[q * k + j]);
            }
        }
    }
}

// ---------------------------------------------------------------------------
// Incremental add
// ---------------------------------------------------------------------------

TEST(IndexSQFastScan, IncrementalAdd) {
    const int d = 64;
    const int nb = 1000;
    const int nq = 20;
    const int k = 10;
    auto xb = make_random_vectors(nb, d, 42);
    auto xq = make_random_vectors(nq, d, 43);
    auto xt = make_random_vectors(1000, d, 44);

    for (auto qtype : kNative4bit) {
        SCOPED_TRACE(qtype);

        faiss::IndexSQFastScan idx1(d, qtype);
        idx1.train(xt.size() / d, xt.data());
        idx1.add(nb, xb.data());

        faiss::IndexSQFastScan idx2(d, qtype);
        idx2.train(xt.size() / d, xt.data());
        idx2.add(nb / 2, xb.data());
        idx2.add(nb / 2, xb.data() + (nb / 2) * d);

        EXPECT_EQ(idx1.ntotal, idx2.ntotal);

        std::vector<float> D1(nq * k), D2(nq * k);
        std::vector<faiss::idx_t> I1(nq * k), I2(nq * k);
        idx1.search(nq, xq.data(), k, D1.data(), I1.data());
        idx2.search(nq, xq.data(), k, D2.data(), I2.data());

        for (int i = 0; i < nq * k; i++) {
            EXPECT_EQ(I1[i], I2[i]);
            EXPECT_NEAR(D1[i], D2[i], 1e-5f);
        }
    }
}

// ---------------------------------------------------------------------------
// Reconstruct
// ---------------------------------------------------------------------------

TEST(IndexSQFastScan, Reconstruct) {
    const int d = 64;
    const int nb = 100;
    auto xb = make_random_vectors(nb, d, 42);
    auto xt = make_random_vectors(1000, d, 44);

    for (auto qtype : kNative4bit) {
        SCOPED_TRACE(qtype);
        faiss::IndexSQFastScan index(d, qtype);
        index.train(xt.size() / d, xt.data());
        index.add(nb, xb.data());

        std::vector<float> xr(nb * d);
        index.reconstruct_n(0, nb, xr.data());

        for (size_t i = 0; i < xr.size(); i++) {
            EXPECT_TRUE(std::isfinite(xr[i]));
        }
    }
}

// ---------------------------------------------------------------------------
// sa_encode / sa_decode round-trip
// ---------------------------------------------------------------------------

TEST(IndexSQFastScan, SaEncodeDecodeRoundTrip) {
    const int d = 64;
    const int nb = 100;
    auto xb = make_random_vectors(nb, d, 42);
    auto xt = make_random_vectors(1000, d, 44);

    for (auto qtype : kNative4bit) {
        SCOPED_TRACE(qtype);
        faiss::IndexSQFastScan fs(d, qtype);
        fs.train(xt.size() / d, xt.data());

        size_t cs = fs.sa_code_size();
        std::vector<uint8_t> encoded(nb * cs);
        fs.sa_encode(nb, xb.data(), encoded.data());

        std::vector<float> decoded(nb * d);
        fs.sa_decode(nb, encoded.data(), decoded.data());

        faiss::IndexScalarQuantizer sq(d, qtype);
        sq.train(xt.size() / d, xt.data());

        std::vector<uint8_t> sq_encoded(nb * cs);
        sq.sa_encode(nb, xb.data(), sq_encoded.data());
        EXPECT_EQ(encoded, sq_encoded);

        std::vector<float> sq_decoded(nb * d);
        sq.sa_decode(nb, sq_encoded.data(), sq_decoded.data());
        for (size_t i = 0; i < decoded.size(); i++) {
            EXPECT_NEAR(decoded[i], sq_decoded[i], 1e-6f);
        }
    }
}

// ---------------------------------------------------------------------------
// add_sa_codes
// ---------------------------------------------------------------------------

TEST(IndexSQFastScan, AddSaCodes) {
    const int d = 64;
    const int nb = 500;
    const int nq = 20;
    const int k = 10;
    auto xb = make_random_vectors(nb, d, 42);
    auto xq = make_random_vectors(nq, d, 43);
    auto xt = make_random_vectors(1000, d, 44);

    for (auto qtype : kNative4bit) {
        SCOPED_TRACE(qtype);

        faiss::IndexSQFastScan ref(d, qtype);
        ref.train(xt.size() / d, xt.data());
        ref.add(nb, xb.data());

        faiss::IndexSQFastScan test(d, qtype);
        test.train(xt.size() / d, xt.data());
        size_t cs = test.sa_code_size();
        std::vector<uint8_t> encoded(nb * cs);
        test.sa_encode(nb, xb.data(), encoded.data());
        test.add_sa_codes(nb, encoded.data(), nullptr);

        EXPECT_EQ(ref.ntotal, test.ntotal);

        std::vector<float> D_ref(nq * k), D_test(nq * k);
        std::vector<faiss::idx_t> I_ref(nq * k), I_test(nq * k);
        ref.search(nq, xq.data(), k, D_ref.data(), I_ref.data());
        test.search(nq, xq.data(), k, D_test.data(), I_test.data());

        for (int q = 0; q < nq; q++) {
            EXPECT_GE(I_ref[q * k], 0);
            EXPECT_GE(I_test[q * k], 0);
        }
    }
}

// ---------------------------------------------------------------------------
// get_distance_computer
// ---------------------------------------------------------------------------

TEST(IndexSQFastScan, DistanceComputer) {
    const int d = 64;
    const int nb = 1000;
    auto xb = make_random_vectors(nb, d, 42);
    auto xt = make_random_vectors(1000, d, 44);

    for (auto qtype : kNative4bit) {
        SCOPED_TRACE(qtype);
        faiss::IndexSQFastScan index(d, qtype);
        index.train(xt.size() / d, xt.data());
        index.add(nb, xb.data());

        std::unique_ptr<faiss::DistanceComputer> dc(
                index.get_distance_computer());
        dc->set_query(xb.data());

        float self_dist = (*dc)(0);
        EXPECT_TRUE(std::isfinite(self_dist));
        // 4-bit quantization has only 16 levels per dimension,
        // so distance-to-self can be non-trivial
        EXPECT_LT(self_dist, 5.0f);

        float other_dist = (*dc)(nb / 2);
        EXPECT_TRUE(std::isfinite(other_dist));
        EXPECT_GE(other_dist, 0.0f);
    }
}

// ---------------------------------------------------------------------------
// range_search
// ---------------------------------------------------------------------------

TEST(IndexSQFastScan, RangeSearch) {
    const int d = 64;
    const int nb = 1000;
    const int nq = 10;
    auto xb = make_random_vectors(nb, d, 42);
    auto xq = make_random_vectors(nq, d, 43);
    auto xt = make_random_vectors(1000, d, 44);

    for (auto qtype : kNative4bit) {
        SCOPED_TRACE(qtype);

        faiss::IndexScalarQuantizer sq(d, qtype);
        sq.train(xt.size() / d, xt.data());
        sq.add(nb, xb.data());

        faiss::IndexSQFastScan fs(d, qtype);
        fs.train(xt.size() / d, xt.data());
        fs.add(nb, xb.data());

        // Pick a reasonable radius
        std::vector<float> D(nb);
        std::vector<faiss::idx_t> I(nb);
        sq.search(1, xq.data(), nb, D.data(), I.data());
        float radius = D[nb / 2];

        faiss::RangeSearchResult rr_sq(nq), rr_fs(nq);
        sq.range_search(nq, xq.data(), radius, &rr_sq);
        fs.range_search(nq, xq.data(), radius, &rr_fs);

        size_t total_sq = rr_sq.lims[nq];
        size_t total_fs = rr_fs.lims[nq];
        EXPECT_GT(total_sq, 0);
        EXPECT_GT(total_fs, 0);
        EXPECT_NEAR(total_sq, total_fs, 2);
    }
}

// ---------------------------------------------------------------------------
// I/O round-trip
// ---------------------------------------------------------------------------

TEST(IndexSQFastScan, IOReadWrite) {
    const int d = 64;
    const int nb = 500;
    const int nq = 10;
    const int k = 10;
    auto xb = make_random_vectors(nb, d, 42);
    auto xq = make_random_vectors(nq, d, 43);
    auto xt = make_random_vectors(1000, d, 44);

    for (auto qtype : kNative4bit) {
        SCOPED_TRACE(qtype);

        faiss::IndexSQFastScan index(d, qtype);
        index.train(xt.size() / d, xt.data());
        index.add(nb, xb.data());

        std::vector<float> D1(nq * k);
        std::vector<faiss::idx_t> I1(nq * k);
        index.search(nq, xq.data(), k, D1.data(), I1.data());

        // Write to buffer
        faiss::VectorIOWriter writer;
        faiss::write_index(&index, &writer);

        // Read back
        faiss::VectorIOReader reader;
        reader.data = std::move(writer.data);
        std::unique_ptr<faiss::Index> loaded(faiss::read_index(&reader));

        ASSERT_NE(loaded, nullptr);
        auto* loaded_fs = dynamic_cast<faiss::IndexSQFastScan*>(loaded.get());
        ASSERT_NE(loaded_fs, nullptr);
        EXPECT_EQ(loaded_fs->ntotal, nb);
        EXPECT_EQ(loaded_fs->d, d);
        EXPECT_EQ(loaded_fs->sq.qtype, qtype);

        std::vector<float> D2(nq * k);
        std::vector<faiss::idx_t> I2(nq * k);
        loaded_fs->search(nq, xq.data(), k, D2.data(), I2.data());

        for (int i = 0; i < nq * k; i++) {
            EXPECT_EQ(I1[i], I2[i]);
            EXPECT_NEAR(D1[i], D2[i], 1e-5f);
        }
    }
}

// ---------------------------------------------------------------------------
// Factory string
// ---------------------------------------------------------------------------

TEST(IndexSQFastScan, Factory) {
    const int d = 64;

    std::unique_ptr<faiss::Index> idx(faiss::index_factory(d, "SQ4fs"));
    ASSERT_NE(idx, nullptr);
    auto* fs = dynamic_cast<faiss::IndexSQFastScan*>(idx.get());
    ASSERT_NE(fs, nullptr);
    EXPECT_EQ(fs->d, d);
    EXPECT_EQ(fs->sq.qtype, SQ::QT_4bit);
}

TEST(IndexSQFastScan, FactoryWithBbs) {
    const int d = 64;
    std::unique_ptr<faiss::Index> idx(faiss::index_factory(d, "SQ4fs_64"));
    auto* fs = dynamic_cast<faiss::IndexSQFastScan*>(idx.get());
    ASSERT_NE(fs, nullptr);
    EXPECT_EQ(fs->bbs, 64);
}

// ---------------------------------------------------------------------------
// permute_entries
// ---------------------------------------------------------------------------

TEST(IndexSQFastScan, PermuteEntries) {
    const int d = 64;
    const int nb = 100;
    auto xb = make_random_vectors(nb, d, 42);
    auto xt = make_random_vectors(1000, d, 44);

    for (auto qtype : kNative4bit) {
        SCOPED_TRACE(qtype);
        faiss::IndexSQFastScan index(d, qtype);
        index.train(xt.size() / d, xt.data());
        index.add(nb, xb.data());

        std::vector<float> before(nb * d);
        index.reconstruct_n(0, nb, before.data());

        std::vector<faiss::idx_t> perm(nb);
        for (int i = 0; i < nb; i++) {
            perm[i] = nb - 1 - i;
        }
        index.permute_entries(perm.data());

        std::vector<float> after(nb * d);
        index.reconstruct_n(0, nb, after.data());

        for (int i = 0; i < nb; i++) {
            for (int j = 0; j < d; j++) {
                EXPECT_NEAR(after[i * d + j], before[perm[i] * d + j], 1e-5f)
                        << "i=" << i << " j=" << j;
            }
        }
    }
}
