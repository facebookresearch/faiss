/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <array>
#include <cmath>
#include <cstring>
#include <memory>
#include <random>
#include <vector>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/impl/scalar_quantizer/distance_computers.h>
#include <faiss/impl/scalar_quantizer/quantizers.h>
#include <faiss/impl/scalar_quantizer/scanners.h>
#include <faiss/impl/scalar_quantizer/similarities.h>
#include <faiss/index_factory.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/random.h>
#include <faiss/utils/simd_levels.h>

namespace {

std::vector<float> make_normalized_vectors(size_t n, size_t d) {
    std::vector<float> x(n * d);
    faiss::float_randn(x.data(), x.size(), 1234);
    faiss::fvec_renorm_L2(d, n, x.data());
    return x;
}

float mean_squared_error(
        const std::vector<float>& x,
        const std::vector<float>& y) {
    EXPECT_EQ(x.size(), y.size());
    return faiss::fvec_L2sqr(x.data(), y.data(), x.size()) / x.size();
}

template <int NBits>
using ScalarTurboQuantQuantizer = faiss::scalar_quantizer::
        QuantizerTurboQuantMSE<NBits, faiss::SIMDLevel::NONE>;

template <int NBits>
using ScalarTurboQuantL2DistanceComputer = faiss::scalar_quantizer::DCTemplate<
        ScalarTurboQuantQuantizer<NBits>,
        faiss::scalar_quantizer::SimilarityL2<faiss::SIMDLevel::NONE>,
        faiss::SIMDLevel::NONE>;

template <int NBits>
using ScalarTurboQuantL2Scanner = faiss::scalar_quantizer::IVFSQScannerL2<
        ScalarTurboQuantL2DistanceComputer<NBits>>;

faiss::ScalarQuantizer make_trained_tqmse_sq(
        size_t d,
        faiss::ScalarQuantizer::QuantizerType qtype) {
    const size_t n = 128;
    std::vector<float> x = make_normalized_vectors(n, d);
    faiss::ScalarQuantizer sq(d, qtype);
    sq.train(n, x.data());
    return sq;
}

struct ScopedSIMDLevel {
    faiss::SIMDLevel original_level;

    explicit ScopedSIMDLevel(faiss::SIMDLevel level)
            : original_level(faiss::SIMDConfig::get_level()) {
        faiss::SIMDConfig::set_level(level);
    }

    ~ScopedSIMDLevel() {
        faiss::SIMDConfig::set_level(original_level);
    }
};

std::vector<faiss::SIMDLevel> available_tqmse_simd_levels() {
    std::vector<faiss::SIMDLevel> levels;
    for (faiss::SIMDLevel level :
         {faiss::SIMDLevel::AVX512,
          faiss::SIMDLevel::AVX2,
          faiss::SIMDLevel::ARM_NEON}) {
        if (faiss::SIMDConfig::is_simd_level_available(level)) {
            levels.push_back(level);
        }
    }
    return levels;
}

template <int NBits>
void expect_tqmse_simd_dispatch_for_compatible_dim(
        faiss::SIMDLevel level,
        faiss::ScalarQuantizer::QuantizerType qtype) {
    ScopedSIMDLevel scoped(level);
    const size_t d = 32;
    faiss::ScalarQuantizer sq = make_trained_tqmse_sq(d, qtype);

    std::unique_ptr<faiss::ScalarQuantizer::SQuantizer> quantizer(
            sq.select_quantizer());
    ASSERT_NE(quantizer, nullptr);

    std::unique_ptr<faiss::ScalarQuantizer::SQDistanceComputer> dc(
            sq.get_distance_computer(faiss::METRIC_L2));
    ASSERT_NE(dc, nullptr);

    std::unique_ptr<faiss::InvertedListScanner> scanner(
            sq.select_InvertedListScanner(
                    faiss::METRIC_L2, nullptr, false, nullptr, false));
    ASSERT_NE(scanner, nullptr);

    auto* quantizer_raw = quantizer.get();
    auto* dc_raw = dc.get();
    auto* scanner_raw = scanner.get();

    EXPECT_NE(typeid(*quantizer_raw), typeid(ScalarTurboQuantQuantizer<NBits>));
    EXPECT_NE(
            typeid(*dc_raw), typeid(ScalarTurboQuantL2DistanceComputer<NBits>));
    EXPECT_NE(typeid(*scanner_raw), typeid(ScalarTurboQuantL2Scanner<NBits>));
}

template <int NBits>
void expect_tqmse_simd_dispatch_fallback_for_incompatible_dim(
        faiss::SIMDLevel level,
        size_t d,
        faiss::ScalarQuantizer::QuantizerType qtype) {
    ScopedSIMDLevel scoped(level);
    faiss::ScalarQuantizer sq = make_trained_tqmse_sq(d, qtype);

    std::unique_ptr<faiss::ScalarQuantizer::SQuantizer> quantizer(
            sq.select_quantizer());
    ASSERT_NE(quantizer, nullptr);

    std::unique_ptr<faiss::ScalarQuantizer::SQDistanceComputer> dc(
            sq.get_distance_computer(faiss::METRIC_L2));
    ASSERT_NE(dc, nullptr);

    std::unique_ptr<faiss::InvertedListScanner> scanner(
            sq.select_InvertedListScanner(
                    faiss::METRIC_L2, nullptr, false, nullptr, false));
    ASSERT_NE(scanner, nullptr);

    auto* quantizer_raw = quantizer.get();
    auto* dc_raw = dc.get();
    auto* scanner_raw = scanner.get();

    EXPECT_EQ(typeid(*quantizer_raw), typeid(ScalarTurboQuantQuantizer<NBits>));
    EXPECT_EQ(
            typeid(*dc_raw), typeid(ScalarTurboQuantL2DistanceComputer<NBits>));
    EXPECT_EQ(typeid(*scanner_raw), typeid(ScalarTurboQuantL2Scanner<NBits>));
}

template <int NBits>
void check_tqmse_distance_path_parity(
        faiss::SIMDLevel level,
        faiss::ScalarQuantizer::QuantizerType qtype) {
    ScopedSIMDLevel scoped(level);
    const size_t d = 32;
    const size_t n = 128;
    std::vector<float> xb = make_normalized_vectors(n, d);
    std::vector<float> xq = make_normalized_vectors(1, d);

    faiss::ScalarQuantizer sq(d, qtype);
    sq.train(n, xb.data());

    std::vector<uint8_t> codes(sq.code_size * n, 0);
    sq.compute_codes(xb.data(), codes.data(), n);

    std::unique_ptr<faiss::ScalarQuantizer::SQDistanceComputer> scalar_dc(
            faiss::scalar_quantizer::sq_select_distance_computer<
                    faiss::SIMDLevel::NONE>(
                    faiss::METRIC_L2, qtype, d, sq.trained));
    std::unique_ptr<faiss::ScalarQuantizer::SQDistanceComputer> simd_dc(
            sq.get_distance_computer(faiss::METRIC_L2));
    ASSERT_NE(scalar_dc, nullptr);
    ASSERT_NE(simd_dc, nullptr);

    scalar_dc->set_query(xq.data());
    simd_dc->set_query(xq.data());

    std::vector<uint8_t> zero_code(sq.code_size, 0);
    std::vector<uint8_t> max_code(sq.code_size, 0xff);
    const std::array<const uint8_t*, 6> query_codes = {
            codes.data(),
            codes.data() + sq.code_size,
            codes.data() + 2 * sq.code_size,
            codes.data() + 3 * sq.code_size,
            zero_code.data(),
            max_code.data()};

    for (const uint8_t* code : query_codes) {
        EXPECT_NEAR(
                scalar_dc->query_to_code(code),
                simd_dc->query_to_code(code),
                1e-5);
    }

    std::vector<uint8_t> bundle(4 * sq.code_size, 0);
    std::memcpy(bundle.data(), zero_code.data(), sq.code_size);
    std::memcpy(bundle.data() + sq.code_size, max_code.data(), sq.code_size);
    std::memcpy(bundle.data() + 2 * sq.code_size, codes.data(), sq.code_size);
    std::memcpy(
            bundle.data() + 3 * sq.code_size,
            codes.data() + sq.code_size,
            sq.code_size);

    scalar_dc->codes = bundle.data();
    scalar_dc->code_size = sq.code_size;
    simd_dc->codes = bundle.data();
    simd_dc->code_size = sq.code_size;

    const std::array<std::pair<faiss::idx_t, faiss::idx_t>, 4> pairs = {{
            {0, 1},
            {0, 2},
            {1, 3},
            {2, 3},
    }};
    for (const auto& [lhs, rhs] : pairs) {
        EXPECT_NEAR(
                scalar_dc->symmetric_dis(lhs, rhs),
                simd_dc->symmetric_dis(lhs, rhs),
                1e-5);
    }

    std::unique_ptr<faiss::InvertedListScanner> scalar_scanner(
            faiss::scalar_quantizer::sq_select_InvertedListScanner<
                    faiss::SIMDLevel::NONE>(
                    qtype,
                    faiss::METRIC_L2,
                    d,
                    sq.code_size,
                    sq.trained,
                    nullptr,
                    false,
                    nullptr,
                    false));
    std::unique_ptr<faiss::InvertedListScanner> simd_scanner(
            sq.select_InvertedListScanner(
                    faiss::METRIC_L2, nullptr, false, nullptr, false));
    ASSERT_NE(scalar_scanner, nullptr);
    ASSERT_NE(simd_scanner, nullptr);

    scalar_scanner->set_query(xq.data());
    simd_scanner->set_query(xq.data());
    scalar_scanner->set_list(0, 0.0f);
    simd_scanner->set_list(0, 0.0f);

    for (const uint8_t* code : query_codes) {
        EXPECT_NEAR(
                scalar_scanner->distance_to_code(code),
                simd_scanner->distance_to_code(code),
                1e-5);
    }
}

void check_tqmse_roundtrip(
        size_t d,
        faiss::ScalarQuantizer::QuantizerType qtype) {
    const size_t n = 128;
    std::vector<float> x = make_normalized_vectors(n, d);
    faiss::ScalarQuantizer sq(d, qtype);

    sq.train(n, x.data());

    std::vector<uint8_t> codes(sq.code_size * n, 0);
    sq.compute_codes(x.data(), codes.data(), n);

    std::vector<float> decoded(n * d);
    sq.decode(codes.data(), decoded.data(), n);

    for (float v : decoded) {
        EXPECT_TRUE(std::isfinite(v));
        EXPECT_LE(v, 1.0f);
        EXPECT_GE(v, -1.0f);
    }
}

} // namespace

TEST(ScalarQuantizer, RSQuantilesClamping) {
    int d = 8;
    int n = 100;

    std::vector<float> x(d * n);
    for (size_t i = 0; i < x.size(); i++) {
        x[i] = static_cast<float>(i % 100);
    }

    faiss::ScalarQuantizer sq(d, faiss::ScalarQuantizer::QT_8bit);
    sq.rangestat = faiss::ScalarQuantizer::RS_quantiles;

    sq.rangestat_arg = 0.05f;
    ASSERT_NO_THROW(sq.train(n, x.data()));

    sq.rangestat_arg = 0.0f;
    ASSERT_NO_THROW(sq.train(n, x.data()));

    sq.rangestat_arg = -0.1f;
    ASSERT_NO_THROW(sq.train(n, x.data()));

    sq.rangestat_arg = 0.8f;
    ASSERT_NO_THROW(sq.train(n, x.data()));

    sq.rangestat_arg = 0.5f;
    ASSERT_NO_THROW(sq.train(n, x.data()));
}

TEST(ScalarQuantizer, RSQuantilesOddSize) {
    int d = 4;
    int n = 5;

    std::vector<float> x(d * n);
    for (size_t i = 0; i < x.size(); i++) {
        x[i] = static_cast<float>(i);
    }

    faiss::ScalarQuantizer sq(d, faiss::ScalarQuantizer::QT_8bit);
    sq.rangestat = faiss::ScalarQuantizer::RS_quantiles;

    sq.rangestat_arg = 0.4f;
    ASSERT_NO_THROW(sq.train(n, x.data()));

    sq.rangestat_arg = 0.5f;
    ASSERT_NO_THROW(sq.train(n, x.data()));

    sq.rangestat_arg = 0.6f;
    ASSERT_NO_THROW(sq.train(n, x.data()));
}

TEST(ScalarQuantizer, RSQuantilesValidRange) {
    int d = 8;
    int n = 100;

    std::vector<float> x(d * n);
    for (size_t i = 0; i < x.size(); i++) {
        x[i] = static_cast<float>(i);
    }

    faiss::ScalarQuantizer sq(d, faiss::ScalarQuantizer::QT_8bit);
    sq.rangestat = faiss::ScalarQuantizer::RS_quantiles;
    sq.rangestat_arg = 0.1f;

    sq.train(n, x.data());

    std::vector<uint8_t> codes(sq.code_size * n);
    ASSERT_NO_THROW(sq.compute_codes(x.data(), codes.data(), n));

    std::vector<float> decoded(d * n);
    ASSERT_NO_THROW(sq.decode(codes.data(), decoded.data(), n));
}

TEST(ScalarQuantizer, RSQuantilesSmallDataset) {
    int d = 2;
    int n = 2;

    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f};

    faiss::ScalarQuantizer sq(d, faiss::ScalarQuantizer::QT_8bit);
    sq.rangestat = faiss::ScalarQuantizer::RS_quantiles;
    sq.rangestat_arg = 0.1f;

    ASSERT_NO_THROW(sq.train(n, x.data()));
}

TEST(TestSQ0bit, CoarseOnlySearch) {
    // Test QT_0bit: centroid-only distance
    int d = 64;
    int nlist = 8;
    int nb = 1000;
    int nq = 10;
    int k = 5;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> distrib(0.0f, 1.0f);
    std::vector<float> xb(nb * d), xq(nq * d);
    for (int i = 0; i < nb * d; i++) {
        xb[i] = distrib(rng);
    }
    for (int i = 0; i < nq * d; i++) {
        xq[i] = distrib(rng);
    }

    faiss::IndexFlatL2 quantizer(d);
    faiss::IndexIVFScalarQuantizer index(
            &quantizer,
            d,
            nlist,
            faiss::ScalarQuantizer::QT_0bit,
            faiss::METRIC_L2,
            false);
    EXPECT_EQ(index.code_size, 0);
    EXPECT_FALSE(index.by_residual);

    index.train(nb, xb.data());
    index.add(nb, xb.data());
    EXPECT_EQ(index.ntotal, nb);

    index.nprobe = nlist;
    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    index.search(nq, xq.data(), k, distances.data(), labels.data());

    // Verify we got results
    for (int q = 0; q < nq; q++) {
        EXPECT_GE(labels[q * k], 0);
    }

    // Compare with direct quantizer search - distances should match
    std::vector<float> coarse_dis(nq * nlist);
    std::vector<faiss::idx_t> coarse_ids(nq * nlist);
    quantizer.search(
            nq, xq.data(), nlist, coarse_dis.data(), coarse_ids.data());

    for (int q = 0; q < nq; q++) {
        float ivf_dis = distances[q * k];
        bool found = false;
        for (int j = 0; j < nlist; j++) {
            if (std::abs(ivf_dis - coarse_dis[q * nlist + j]) < 1e-5) {
                found = true;
                break;
            }
        }
        EXPECT_TRUE(found) << "IVF distance " << ivf_dis
                           << " not found in coarse distances for query " << q;
    }
}

TEST(TestSQ0bit, IndexFactory) {
    int d = 32;
    std::unique_ptr<faiss::Index> index(faiss::index_factory(d, "IVF8,SQ0"));
    EXPECT_NE(index, nullptr);
    auto* ivfsq = dynamic_cast<faiss::IndexIVFScalarQuantizer*>(index.get());
    EXPECT_NE(ivfsq, nullptr);
    EXPECT_EQ(ivfsq->sq.qtype, faiss::ScalarQuantizer::QT_0bit);
    EXPECT_EQ(ivfsq->code_size, 0);
}

TEST(TestSQ0bit, InnerProduct) {
    int d = 64;
    int nlist = 4;
    int nb = 500;
    int nq = 5;
    int k = 3;

    std::mt19937 rng2(43);
    std::uniform_real_distribution<float> distrib2(0.0f, 1.0f);
    std::vector<float> xb(nb * d), xq(nq * d);
    for (int i = 0; i < nb * d; i++) {
        xb[i] = distrib2(rng2);
    }
    for (int i = 0; i < nq * d; i++) {
        xq[i] = distrib2(rng2);
    }

    faiss::IndexFlatIP quantizer(d);
    faiss::IndexIVFScalarQuantizer index(
            &quantizer,
            d,
            nlist,
            faiss::ScalarQuantizer::QT_0bit,
            faiss::METRIC_INNER_PRODUCT,
            false);

    index.train(nb, xb.data());
    index.add(nb, xb.data());

    index.nprobe = nlist;
    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    index.search(nq, xq.data(), k, distances.data(), labels.data());

    for (int q = 0; q < nq; q++) {
        EXPECT_GE(labels[q * k], 0);
    }
}

TEST(ScalarQuantizer, TQMSEEncodeDecode) {
    check_tqmse_roundtrip(32, faiss::ScalarQuantizer::QT_1bit_tqmse);
    check_tqmse_roundtrip(32, faiss::ScalarQuantizer::QT_2bit_tqmse);
    check_tqmse_roundtrip(32, faiss::ScalarQuantizer::QT_3bit_tqmse);
    check_tqmse_roundtrip(32, faiss::ScalarQuantizer::QT_4bit_tqmse);
    check_tqmse_roundtrip(32, faiss::ScalarQuantizer::QT_8bit_tqmse);
}

TEST(ScalarQuantizer, TQMSEAccuracyOrdering) {
    const size_t d = 32;
    const size_t n = 256;
    std::vector<float> x = make_normalized_vectors(n, d);
    const std::array<faiss::ScalarQuantizer::QuantizerType, 5> qtypes = {
            faiss::ScalarQuantizer::QT_1bit_tqmse,
            faiss::ScalarQuantizer::QT_2bit_tqmse,
            faiss::ScalarQuantizer::QT_3bit_tqmse,
            faiss::ScalarQuantizer::QT_4bit_tqmse,
            faiss::ScalarQuantizer::QT_8bit_tqmse};

    std::vector<float> mses;
    mses.reserve(qtypes.size());

    for (auto qtype : qtypes) {
        faiss::ScalarQuantizer sq(d, qtype);
        sq.train(n, x.data());

        std::vector<uint8_t> codes(sq.code_size * n, 0);
        sq.compute_codes(x.data(), codes.data(), n);

        std::vector<float> decoded(n * d);
        sq.decode(codes.data(), decoded.data(), n);
        mses.push_back(mean_squared_error(x, decoded));
    }

    for (size_t i = 0; i + 1 < mses.size(); ++i) {
        EXPECT_GE(mses[i], mses[i + 1]);
    }
}

TEST(ScalarQuantizer, TQMSENonSimdDims) {
    check_tqmse_roundtrip(7, faiss::ScalarQuantizer::QT_1bit_tqmse);
    check_tqmse_roundtrip(9, faiss::ScalarQuantizer::QT_2bit_tqmse);
    check_tqmse_roundtrip(11, faiss::ScalarQuantizer::QT_3bit_tqmse);
    check_tqmse_roundtrip(13, faiss::ScalarQuantizer::QT_4bit_tqmse);
    check_tqmse_roundtrip(33, faiss::ScalarQuantizer::QT_8bit_tqmse);
}

TEST(ScalarQuantizer, TQMSESimdDispatchSelection) {
    const std::vector<faiss::SIMDLevel> levels = available_tqmse_simd_levels();
    if (levels.empty()) {
        GTEST_SKIP() << "No SIMD level available for TurboQuant dispatch tests";
    }

    for (faiss::SIMDLevel level : levels) {
        SCOPED_TRACE(faiss::to_string(level));
        expect_tqmse_simd_dispatch_for_compatible_dim<1>(
                level, faiss::ScalarQuantizer::QT_1bit_tqmse);
        expect_tqmse_simd_dispatch_for_compatible_dim<2>(
                level, faiss::ScalarQuantizer::QT_2bit_tqmse);
        expect_tqmse_simd_dispatch_for_compatible_dim<3>(
                level, faiss::ScalarQuantizer::QT_3bit_tqmse);
        expect_tqmse_simd_dispatch_for_compatible_dim<4>(
                level, faiss::ScalarQuantizer::QT_4bit_tqmse);
        expect_tqmse_simd_dispatch_for_compatible_dim<8>(
                level, faiss::ScalarQuantizer::QT_8bit_tqmse);

        expect_tqmse_simd_dispatch_fallback_for_incompatible_dim<1>(
                level, 7, faiss::ScalarQuantizer::QT_1bit_tqmse);
        expect_tqmse_simd_dispatch_fallback_for_incompatible_dim<2>(
                level, 9, faiss::ScalarQuantizer::QT_2bit_tqmse);
        expect_tqmse_simd_dispatch_fallback_for_incompatible_dim<3>(
                level, 11, faiss::ScalarQuantizer::QT_3bit_tqmse);
        expect_tqmse_simd_dispatch_fallback_for_incompatible_dim<4>(
                level, 13, faiss::ScalarQuantizer::QT_4bit_tqmse);
        expect_tqmse_simd_dispatch_fallback_for_incompatible_dim<8>(
                level, 33, faiss::ScalarQuantizer::QT_8bit_tqmse);
    }
}

TEST(ScalarQuantizer, TQMSESimdDistancePathParity) {
    const std::vector<faiss::SIMDLevel> levels = available_tqmse_simd_levels();
    if (levels.empty()) {
        GTEST_SKIP() << "No SIMD level available for TurboQuant parity tests";
    }

    for (faiss::SIMDLevel level : levels) {
        SCOPED_TRACE(faiss::to_string(level));
        check_tqmse_distance_path_parity<1>(
                level, faiss::ScalarQuantizer::QT_1bit_tqmse);
        check_tqmse_distance_path_parity<2>(
                level, faiss::ScalarQuantizer::QT_2bit_tqmse);
        check_tqmse_distance_path_parity<3>(
                level, faiss::ScalarQuantizer::QT_3bit_tqmse);
        check_tqmse_distance_path_parity<4>(
                level, faiss::ScalarQuantizer::QT_4bit_tqmse);
        check_tqmse_distance_path_parity<8>(
                level, faiss::ScalarQuantizer::QT_8bit_tqmse);
    }
}
