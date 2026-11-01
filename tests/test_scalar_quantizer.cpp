/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <algorithm>
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

template <int NBits>
using ScalarLloydMaxQuantizer = faiss::scalar_quantizer::
        QuantizerLloydMax<NBits, faiss::SIMDLevel::NONE>;

template <int NBits>
using ScalarLloydMaxL2DistanceComputer = faiss::scalar_quantizer::DCTemplate<
        ScalarLloydMaxQuantizer<NBits>,
        faiss::scalar_quantizer::SimilarityL2<faiss::SIMDLevel::NONE>,
        faiss::SIMDLevel::NONE>;

template <int NBits>
using ScalarLloydMaxL2Scanner = faiss::scalar_quantizer::IVFSQScannerL2<
        ScalarLloydMaxL2DistanceComputer<NBits>>;

faiss::ScalarQuantizer make_trained_lloyd_max_sq(
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

std::vector<faiss::SIMDLevel> available_lloyd_max_simd_levels() {
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

// SIMD levels with scalar-quantizer kernels that can be checked for parity
// against the scalar (NONE) path. Unlike dispatch tests, parity tests do not
// require dimension restrictions, so RISCV_RVV (no alignment constraints)
// is included here.
std::vector<faiss::SIMDLevel> available_sq_parity_simd_levels() {
    std::vector<faiss::SIMDLevel> levels;
    for (faiss::SIMDLevel level :
         {faiss::SIMDLevel::AVX512,
          faiss::SIMDLevel::AVX2,
          faiss::SIMDLevel::ARM_NEON,
          faiss::SIMDLevel::RISCV_RVV}) {
        if (faiss::SIMDConfig::is_simd_level_available(level)) {
            levels.push_back(level);
        }
    }
    return levels;
}

template <int NBits>
void expect_lloyd_max_simd_dispatch_for_compatible_dim(
        faiss::SIMDLevel level,
        faiss::ScalarQuantizer::QuantizerType qtype) {
    ScopedSIMDLevel scoped(level);
    const size_t d = 32;
    faiss::ScalarQuantizer sq = make_trained_lloyd_max_sq(d, qtype);

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

    EXPECT_NE(typeid(*quantizer_raw), typeid(ScalarLloydMaxQuantizer<NBits>));
    EXPECT_NE(typeid(*dc_raw), typeid(ScalarLloydMaxL2DistanceComputer<NBits>));
    EXPECT_NE(typeid(*scanner_raw), typeid(ScalarLloydMaxL2Scanner<NBits>));
}

template <int NBits>
void expect_lloyd_max_simd_dispatch_fallback_for_incompatible_dim(
        faiss::SIMDLevel level,
        size_t d,
        faiss::ScalarQuantizer::QuantizerType qtype) {
    ScopedSIMDLevel scoped(level);
    faiss::ScalarQuantizer sq = make_trained_lloyd_max_sq(d, qtype);

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

    EXPECT_EQ(typeid(*quantizer_raw), typeid(ScalarLloydMaxQuantizer<NBits>));
    EXPECT_EQ(typeid(*dc_raw), typeid(ScalarLloydMaxL2DistanceComputer<NBits>));
    EXPECT_EQ(typeid(*scanner_raw), typeid(ScalarLloydMaxL2Scanner<NBits>));
}

// Random data never lands exactly on a boundary, so the distance-path parity
// checks cannot catch a SIMD encoder whose tie-break differs from the scalar
// upper_bound (x == b encodes to the upper bin). Encode the boundaries
// themselves and require byte-identical codes.
void check_lloyd_max_boundary_encode_parity(
        faiss::SIMDLevel level,
        faiss::ScalarQuantizer::QuantizerType qtype) {
    ScopedSIMDLevel scoped(level);
    const size_t d = 32;
    faiss::ScalarQuantizer sq = make_trained_lloyd_max_sq(d, qtype);

    const size_t k = (sq.trained.size() + 1) / 2;
    const float* boundaries = sq.trained.data() + k;
    std::vector<float> x(d);
    for (size_t i = 0; i < d; i++) {
        x[i] = boundaries[i % (k - 1)];
    }
    x[d - 2] = 0.0f;
    x[d - 1] = -0.0f;

    std::unique_ptr<faiss::ScalarQuantizer::SQuantizer> scalar_quant(
            faiss::scalar_quantizer::sq_select_quantizer<
                    faiss::SIMDLevel::NONE>(qtype, d, sq.trained));
    std::unique_ptr<faiss::ScalarQuantizer::SQuantizer> simd_quant(
            sq.select_quantizer());
    ASSERT_NE(scalar_quant, nullptr);
    ASSERT_NE(simd_quant, nullptr);

    std::vector<uint8_t> ref(sq.code_size, 0), out(sq.code_size, 0);
    scalar_quant->encode_vector(x.data(), ref.data());
    simd_quant->encode_vector(x.data(), out.data());
    EXPECT_EQ(ref, out);
}

// Parity between a SIMD level and the scalar (NONE) path for one quantizer
// type, metric and dimension.
void check_sq_distance_path_parity(
        faiss::SIMDLevel level,
        faiss::ScalarQuantizer::QuantizerType qtype,
        faiss::MetricType metric,
        size_t d) {
    ScopedSIMDLevel scoped(level);
    const size_t n = 64;
    std::vector<float> xb = make_normalized_vectors(n, d);
    std::vector<float> xq = make_normalized_vectors(1, d);

    faiss::ScalarQuantizer sq(d, qtype);
    sq.train(n, xb.data());

    std::vector<uint8_t> codes(sq.code_size * n, 0);
    sq.compute_codes(xb.data(), codes.data(), n);

    // fp16/bf16 codes are raw IEEE bit patterns: arbitrary bytes can decode
    // to NaN, so only codes produced by compute_codes are checked for those.
    const bool arbitrary_bytes_ok = qtype != faiss::ScalarQuantizer::QT_fp16 &&
            qtype != faiss::ScalarQuantizer::QT_bf16;

    std::vector<uint8_t> zero_code(sq.code_size, 0);
    std::vector<uint8_t> max_code(sq.code_size, 0xff);
    std::vector<const uint8_t*> query_codes = {
            codes.data(),
            codes.data() + sq.code_size,
            codes.data() + 2 * sq.code_size,
            codes.data() + 3 * sq.code_size};
    if (arbitrary_bytes_ok) {
        query_codes.push_back(zero_code.data());
        query_codes.push_back(max_code.data());
    }

    std::unique_ptr<faiss::ScalarQuantizer::SQuantizer> scalar_quant(
            faiss::scalar_quantizer::sq_select_quantizer<
                    faiss::SIMDLevel::NONE>(qtype, d, sq.trained));
    std::unique_ptr<faiss::ScalarQuantizer::SQuantizer> simd_quant(
            sq.select_quantizer());
    ASSERT_NE(scalar_quant, nullptr);
    ASSERT_NE(simd_quant, nullptr);

    for (const uint8_t* code : query_codes) {
        std::vector<float> ref(d), out(d);
        scalar_quant->decode_vector(code, ref.data());
        simd_quant->decode_vector(code, out.data());
        for (size_t j = 0; j < d; j++) {
            EXPECT_NEAR(
                    ref[j], out[j], 1e-5 * std::max(1.0f, std::fabs(ref[j])));
        }
    }

    auto expect_distance_near = [](float ref, float out) {
        EXPECT_NEAR(ref, out, 5e-3 * std::max(1.0f, std::fabs(ref)));
    };

    std::unique_ptr<faiss::ScalarQuantizer::SQDistanceComputer> scalar_dc(
            faiss::scalar_quantizer::sq_select_distance_computer<
                    faiss::SIMDLevel::NONE>(metric, qtype, d, sq.trained));
    std::unique_ptr<faiss::ScalarQuantizer::SQDistanceComputer> simd_dc(
            sq.get_distance_computer(metric));
    ASSERT_NE(scalar_dc, nullptr);
    ASSERT_NE(simd_dc, nullptr);

    scalar_dc->set_query(xq.data());
    simd_dc->set_query(xq.data());

    for (const uint8_t* code : query_codes) {
        expect_distance_near(
                scalar_dc->query_to_code(code), simd_dc->query_to_code(code));
    }

    float scalar_dis[4], simd_dis[4];
    scalar_dc->query_to_codes_batch_4(
            query_codes[0],
            query_codes[1],
            query_codes[2],
            query_codes[3],
            scalar_dis[0],
            scalar_dis[1],
            scalar_dis[2],
            scalar_dis[3]);
    simd_dc->query_to_codes_batch_4(
            query_codes[0],
            query_codes[1],
            query_codes[2],
            query_codes[3],
            simd_dis[0],
            simd_dis[1],
            simd_dis[2],
            simd_dis[3]);
    for (int k = 0; k < 4; k++) {
        expect_distance_near(scalar_dis[k], simd_dis[k]);
    }

    std::vector<uint8_t> bundle(4 * sq.code_size, 0);
    for (int k = 0; k < 4; k++) {
        std::memcpy(
                bundle.data() + k * sq.code_size, query_codes[k], sq.code_size);
    }
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
        expect_distance_near(
                scalar_dc->symmetric_dis(lhs, rhs),
                simd_dc->symmetric_dis(lhs, rhs));
    }

    std::unique_ptr<faiss::InvertedListScanner> scalar_scanner(
            faiss::scalar_quantizer::sq_select_InvertedListScanner<
                    faiss::SIMDLevel::NONE>(
                    qtype,
                    metric,
                    d,
                    sq.code_size,
                    sq.trained,
                    nullptr,
                    false,
                    nullptr,
                    false));
    std::unique_ptr<faiss::InvertedListScanner> simd_scanner(
            sq.select_InvertedListScanner(
                    metric, nullptr, false, nullptr, false));
    ASSERT_NE(scalar_scanner, nullptr);
    ASSERT_NE(simd_scanner, nullptr);

    scalar_scanner->set_query(xq.data());
    simd_scanner->set_query(xq.data());
    scalar_scanner->set_list(0, 0.0f);
    simd_scanner->set_list(0, 0.0f);

    for (const uint8_t* code : query_codes) {
        expect_distance_near(
                scalar_scanner->distance_to_code(code),
                simd_scanner->distance_to_code(code));
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

void check_eden_roundtrip(
        size_t d,
        faiss::ScalarQuantizer::QuantizerType qtype) {
    const size_t n = 128;
    std::vector<float> x = make_normalized_vectors(n, d);
    faiss::ScalarQuantizer sq(d, qtype);

    sq.train(0, nullptr);

    std::vector<uint8_t> codes(sq.code_size * n, 0);
    sq.compute_codes(x.data(), codes.data(), n);

    std::vector<float> decoded(n * d);
    sq.decode(codes.data(), decoded.data(), n);

    for (float v : decoded) {
        EXPECT_TRUE(std::isfinite(v));
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

TEST(ScalarQuantizer, MinimalTraining) {
    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f};
    using SQ = faiss::ScalarQuantizer;
    for (auto qt : {
                 SQ::QT_4bit,
                 SQ::QT_4bit_uniform,
                 SQ::QT_6bit,
                 SQ::QT_8bit,
                 SQ::QT_8bit_uniform,
         }) {
        SCOPED_TRACE(qt);
        for (size_t d : {1, 2}) {
            faiss::IndexScalarQuantizer index(d, qt);
            index.sq.rangestat = SQ::RS_minmax;
            EXPECT_THROW(index.train(0, x.data()), faiss::FaissException);
            EXPECT_THROW(index.train(1, nullptr), faiss::FaissException);
            EXPECT_NO_THROW(index.train(1, x.data()));
        }
    }
}

TEST(ScalarQuantizer, RVVDirect8bitEncodingMatchesTruncation) {
    if (!faiss::SIMDConfig::is_simd_level_available(
                faiss::SIMDLevel::RISCV_RVV)) {
        GTEST_SKIP() << "RVV is not available";
    }
    ScopedSIMDLevel scoped(faiss::SIMDLevel::RISCV_RVV);
    // Negative fractions greater than -1 truncate to the valid code zero.
    // Inputs outside (-1, 256) have no portable scalar conversion contract.
    const std::vector<float> values = {
            std::nextafter(-1.0f, 0.0f),
            -0.75f,
            -0.0f,
            std::nextafter(0.0f, -1.0f),
            0.0f,
            std::nextafter(0.0f, 1.0f),
            std::nextafter(1.0f, 0.0f),
            1.0f,
            std::nextafter(1.0f, 2.0f),
            127.75f,
            128.5f,
            254.999f,
            255.0f,
            std::nextafter(256.0f, 0.0f)};
    for (size_t d : {0, 1, 15, 16, 17, 31, 32, 33, 65, 769}) {
        SCOPED_TRACE(d);
        const size_t n = 3;
        std::vector<float> input(d * n + 1);
        std::vector<uint8_t> expected(d * n + 32, 0xa5);
        for (size_t i = 0; i < d * n; ++i) {
            input[i] = values[i % values.size()];
            expected[16 + i] = static_cast<uint8_t>(input[i]);
        }
        std::vector<uint8_t> actual(d * n + 32, 0xa5);
        faiss::ScalarQuantizer sq(d, faiss::ScalarQuantizer::QT_8bit_direct);
        sq.compute_codes(input.data(), actual.data() + 16, n);
        EXPECT_EQ(actual, expected);
    }
}

TEST(TestSQ0bit, CoarseOnlySearch) {
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

    index.train(nb, xb.data());
    index.add(nb, xb.data());

    index.nprobe = nlist;
    std::vector<float> distances(nq * k);
    std::vector<faiss::idx_t> labels(nq * k);
    index.search(nq, xq.data(), k, distances.data(), labels.data());

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

TEST(ScalarQuantizer, EDENEncodeDecode) {
    check_eden_roundtrip(32, faiss::ScalarQuantizer::QT_1bit_eden);
    check_eden_roundtrip(32, faiss::ScalarQuantizer::QT_2bit_eden);
    check_eden_roundtrip(32, faiss::ScalarQuantizer::QT_3bit_eden);
    check_eden_roundtrip(32, faiss::ScalarQuantizer::QT_4bit_eden);
    check_eden_roundtrip(32, faiss::ScalarQuantizer::QT_5bit_eden);
    check_eden_roundtrip(32, faiss::ScalarQuantizer::QT_6bit_eden);
    check_eden_roundtrip(32, faiss::ScalarQuantizer::QT_7bit_eden);
    check_eden_roundtrip(32, faiss::ScalarQuantizer::QT_8bit_eden);
}

TEST(ScalarQuantizer, TQMSENonSimdDims) {
    check_tqmse_roundtrip(7, faiss::ScalarQuantizer::QT_1bit_tqmse);
    check_tqmse_roundtrip(9, faiss::ScalarQuantizer::QT_2bit_tqmse);
    check_tqmse_roundtrip(11, faiss::ScalarQuantizer::QT_3bit_tqmse);
    check_tqmse_roundtrip(13, faiss::ScalarQuantizer::QT_4bit_tqmse);
    check_tqmse_roundtrip(33, faiss::ScalarQuantizer::QT_8bit_tqmse);
}

TEST(ScalarQuantizer, LloydMaxSimdDispatchSelection) {
    const std::vector<faiss::SIMDLevel> levels =
            available_lloyd_max_simd_levels();
    if (levels.empty()) {
        GTEST_SKIP() << "No SIMD level available for Lloyd-Max dispatch tests";
    }

    using SQ = faiss::ScalarQuantizer;
    for (faiss::SIMDLevel level : levels) {
        SCOPED_TRACE(faiss::to_string(level));
        expect_lloyd_max_simd_dispatch_for_compatible_dim<1>(
                level, SQ::QT_1bit_tqmse);
        expect_lloyd_max_simd_dispatch_for_compatible_dim<1>(
                level, SQ::QT_1bit_eden);
        expect_lloyd_max_simd_dispatch_for_compatible_dim<2>(
                level, SQ::QT_2bit_tqmse);
        expect_lloyd_max_simd_dispatch_for_compatible_dim<2>(
                level, SQ::QT_2bit_eden);
        expect_lloyd_max_simd_dispatch_for_compatible_dim<3>(
                level, SQ::QT_3bit_tqmse);
        expect_lloyd_max_simd_dispatch_for_compatible_dim<3>(
                level, SQ::QT_3bit_eden);
        expect_lloyd_max_simd_dispatch_for_compatible_dim<4>(
                level, SQ::QT_4bit_tqmse);
        expect_lloyd_max_simd_dispatch_for_compatible_dim<4>(
                level, SQ::QT_4bit_eden);
        expect_lloyd_max_simd_dispatch_for_compatible_dim<8>(
                level, SQ::QT_8bit_tqmse);
        expect_lloyd_max_simd_dispatch_for_compatible_dim<8>(
                level, SQ::QT_8bit_eden);

        expect_lloyd_max_simd_dispatch_fallback_for_incompatible_dim<1>(
                level, 7, SQ::QT_1bit_tqmse);
        expect_lloyd_max_simd_dispatch_fallback_for_incompatible_dim<1>(
                level, 7, SQ::QT_1bit_eden);
        expect_lloyd_max_simd_dispatch_fallback_for_incompatible_dim<2>(
                level, 9, SQ::QT_2bit_tqmse);
        expect_lloyd_max_simd_dispatch_fallback_for_incompatible_dim<2>(
                level, 9, SQ::QT_2bit_eden);
        expect_lloyd_max_simd_dispatch_fallback_for_incompatible_dim<3>(
                level, 11, SQ::QT_3bit_tqmse);
        expect_lloyd_max_simd_dispatch_fallback_for_incompatible_dim<3>(
                level, 11, SQ::QT_3bit_eden);
        expect_lloyd_max_simd_dispatch_fallback_for_incompatible_dim<4>(
                level, 13, SQ::QT_4bit_tqmse);
        expect_lloyd_max_simd_dispatch_fallback_for_incompatible_dim<4>(
                level, 13, SQ::QT_4bit_eden);
        expect_lloyd_max_simd_dispatch_fallback_for_incompatible_dim<8>(
                level, 33, SQ::QT_8bit_tqmse);
        expect_lloyd_max_simd_dispatch_fallback_for_incompatible_dim<8>(
                level, 33, SQ::QT_8bit_eden);
    }
}

TEST(ScalarQuantizer, LloydMaxSimdDistancePathParity) {
    const std::vector<faiss::SIMDLevel> levels =
            available_sq_parity_simd_levels();
    if (levels.empty()) {
        GTEST_SKIP() << "No SIMD level available for Lloyd-Max parity tests";
    }

    for (faiss::SIMDLevel level : levels) {
        SCOPED_TRACE(faiss::to_string(level));
        for (auto qtype :
             {faiss::ScalarQuantizer::QT_1bit_tqmse,
              faiss::ScalarQuantizer::QT_2bit_tqmse,
              faiss::ScalarQuantizer::QT_3bit_tqmse,
              faiss::ScalarQuantizer::QT_4bit_tqmse,
              faiss::ScalarQuantizer::QT_8bit_tqmse,
              faiss::ScalarQuantizer::QT_1bit_eden,
              faiss::ScalarQuantizer::QT_2bit_eden,
              faiss::ScalarQuantizer::QT_3bit_eden,
              faiss::ScalarQuantizer::QT_4bit_eden,
              faiss::ScalarQuantizer::QT_8bit_eden}) {
            SCOPED_TRACE(static_cast<int>(qtype));
            check_sq_distance_path_parity(level, qtype, faiss::METRIC_L2, 32);
        }
    }
}

TEST(ScalarQuantizer, LloydMaxSimdBoundaryEncodeParity) {
    const std::vector<faiss::SIMDLevel> levels =
            available_sq_parity_simd_levels();
    if (levels.empty()) {
        GTEST_SKIP() << "No SIMD level available for Lloyd-Max parity tests";
    }

    for (faiss::SIMDLevel level : levels) {
        SCOPED_TRACE(faiss::to_string(level));
        for (auto qtype :
             {faiss::ScalarQuantizer::QT_1bit_tqmse,
              faiss::ScalarQuantizer::QT_2bit_tqmse,
              faiss::ScalarQuantizer::QT_1bit_eden,
              faiss::ScalarQuantizer::QT_2bit_eden}) {
            SCOPED_TRACE(static_cast<int>(qtype));
            check_lloyd_max_boundary_encode_parity(level, qtype);
        }
    }
}

// Dimensions straddle VLMAX = VLEN / 4 (e32m8) for VLEN 128 to 2048.
// QT_fp16 is in RVVFP16DistancePathParity.
TEST(ScalarQuantizer, RVVDistancePathParity) {
    if (!faiss::SIMDConfig::is_simd_level_available(
                faiss::SIMDLevel::RISCV_RVV)) {
        GTEST_SKIP() << "RISCV_RVV not available on this machine";
    }

    const std::vector<faiss::ScalarQuantizer::QuantizerType> qtypes = {
            faiss::ScalarQuantizer::QT_8bit,
            faiss::ScalarQuantizer::QT_4bit,
            faiss::ScalarQuantizer::QT_6bit,
            faiss::ScalarQuantizer::QT_8bit_uniform,
            faiss::ScalarQuantizer::QT_4bit_uniform,
            faiss::ScalarQuantizer::QT_bf16,
            faiss::ScalarQuantizer::QT_8bit_direct,
            faiss::ScalarQuantizer::QT_8bit_direct_signed,
            faiss::ScalarQuantizer::QT_1bit_eden,
            faiss::ScalarQuantizer::QT_2bit_eden,
            faiss::ScalarQuantizer::QT_3bit_eden,
            faiss::ScalarQuantizer::QT_4bit_eden,
            faiss::ScalarQuantizer::QT_8bit_eden,
            // The QT_*_tqmse types currently alias QuantizerLloydMax, so
            // they share the EDEN RVV kernels; listing them makes that
            // dispatch coverage explicit for every metric and distance API.
            faiss::ScalarQuantizer::QT_1bit_tqmse,
            faiss::ScalarQuantizer::QT_2bit_tqmse,
            faiss::ScalarQuantizer::QT_3bit_tqmse,
            faiss::ScalarQuantizer::QT_4bit_tqmse,
            faiss::ScalarQuantizer::QT_8bit_tqmse,
    };
    const std::vector<size_t> dims = {
            1,
            7,
            31,
            32,
            33,
            63,
            64,
            65,
            127,
            128,
            129,
            255,
            256,
            257,
            511,
            512,
            513};
    const std::vector<faiss::MetricType> metrics = {
            faiss::METRIC_L2, faiss::METRIC_INNER_PRODUCT};

    for (auto qtype : qtypes) {
        for (size_t d : dims) {
            for (auto metric : metrics) {
                SCOPED_TRACE(
                        testing::Message()
                        << "qtype=" << static_cast<int>(qtype) << " d=" << d
                        << " metric=" << metric);
                check_sq_distance_path_parity(
                        faiss::SIMDLevel::RISCV_RVV, qtype, metric, d);
            }
        }
    }
}

// Separate test: the RVV FP16 kernel needs Zvfhmin (SIGILL without it).
TEST(ScalarQuantizer, RVVFP16DistancePathParity) {
    if (!faiss::SIMDConfig::is_simd_level_available(
                faiss::SIMDLevel::RISCV_RVV)) {
        GTEST_SKIP() << "RISCV_RVV not available on this machine";
    }

    const std::vector<size_t> dims = {
            1,
            7,
            31,
            32,
            33,
            63,
            64,
            65,
            127,
            128,
            129,
            255,
            256,
            257,
            511,
            512,
            513};
    const std::vector<faiss::MetricType> metrics = {
            faiss::METRIC_L2, faiss::METRIC_INNER_PRODUCT};

    for (size_t d : dims) {
        for (auto metric : metrics) {
            SCOPED_TRACE(
                    testing::Message() << "d=" << d << " metric=" << metric);
            check_sq_distance_path_parity(
                    faiss::SIMDLevel::RISCV_RVV,
                    faiss::ScalarQuantizer::QT_fp16,
                    metric,
                    d);
        }
    }
}

// d == 0 early returns in the RVV DCTemplate loops must give exactly 0
// without reading the query or the codes.
TEST(ScalarQuantizer, RVVZeroDimDistancePathParity) {
    if (!faiss::SIMDConfig::is_simd_level_available(
                faiss::SIMDLevel::RISCV_RVV)) {
        GTEST_SKIP() << "RISCV_RVV not available on this machine";
    }
    ScopedSIMDLevel scoped(faiss::SIMDLevel::RISCV_RVV);

    // trained layouts each quantizer expects at d == 0: non-uniform
    // templates take 2 * d (empty) values, uniform templates take {vmin,
    // vdiff}, LloydMax takes 2^k - 1 centroids/boundaries, and the raw
    // codecs (fp16/bf16/direct) ignore trained entirely.
    const std::vector<std::pair<
            faiss::ScalarQuantizer::QuantizerType,
            std::vector<float>>>
            cases = {
                    {faiss::ScalarQuantizer::QT_8bit, {}},
                    {faiss::ScalarQuantizer::QT_4bit, {}},
                    {faiss::ScalarQuantizer::QT_6bit, {}},
                    {faiss::ScalarQuantizer::QT_8bit_uniform, {0.0f, 1.0f}},
                    {faiss::ScalarQuantizer::QT_4bit_uniform, {0.0f, 1.0f}},
                    {faiss::ScalarQuantizer::QT_fp16, {}},
                    {faiss::ScalarQuantizer::QT_bf16, {}},
                    {faiss::ScalarQuantizer::QT_8bit_direct, {}},
                    {faiss::ScalarQuantizer::QT_8bit_direct_signed, {}},
                    {faiss::ScalarQuantizer::QT_1bit_eden,
                     std::vector<float>(3, 0.0f)},
                    {faiss::ScalarQuantizer::QT_2bit_eden,
                     std::vector<float>(7, 0.0f)},
                    {faiss::ScalarQuantizer::QT_3bit_eden,
                     std::vector<float>(15, 0.0f)},
                    {faiss::ScalarQuantizer::QT_4bit_eden,
                     std::vector<float>(31, 0.0f)},
                    {faiss::ScalarQuantizer::QT_8bit_eden,
                     std::vector<float>(511, 0.0f)},
            };
    const std::vector<faiss::MetricType> metrics = {
            faiss::METRIC_L2, faiss::METRIC_INNER_PRODUCT};

    uint8_t dummy_code[8] = {};

    for (const auto& [qtype, trained] : cases) {
        for (auto metric : metrics) {
            SCOPED_TRACE(
                    testing::Message() << "qtype=" << static_cast<int>(qtype)
                                       << " metric=" << metric);

            faiss::ScalarQuantizer sq(0, qtype);
            sq.trained = trained;

            std::unique_ptr<faiss::ScalarQuantizer::SQDistanceComputer>
                    scalar_dc(
                            faiss::scalar_quantizer::
                                    sq_select_distance_computer<
                                            faiss::SIMDLevel::NONE>(
                                            metric, qtype, 0, sq.trained));
            std::unique_ptr<faiss::ScalarQuantizer::SQDistanceComputer> simd_dc(
                    sq.get_distance_computer(metric));
            ASSERT_NE(scalar_dc, nullptr);
            ASSERT_NE(simd_dc, nullptr);

            scalar_dc->set_query(nullptr);
            simd_dc->set_query(nullptr);

            EXPECT_FLOAT_EQ(scalar_dc->query_to_code(dummy_code), 0.0f);
            EXPECT_FLOAT_EQ(simd_dc->query_to_code(dummy_code), 0.0f);

            float scalar_dis[4], simd_dis[4];
            scalar_dc->query_to_codes_batch_4(
                    dummy_code,
                    dummy_code,
                    dummy_code,
                    dummy_code,
                    scalar_dis[0],
                    scalar_dis[1],
                    scalar_dis[2],
                    scalar_dis[3]);
            simd_dc->query_to_codes_batch_4(
                    dummy_code,
                    dummy_code,
                    dummy_code,
                    dummy_code,
                    simd_dis[0],
                    simd_dis[1],
                    simd_dis[2],
                    simd_dis[3]);
            for (int k = 0; k < 4; k++) {
                EXPECT_FLOAT_EQ(scalar_dis[k], 0.0f);
                EXPECT_FLOAT_EQ(simd_dis[k], 0.0f);
            }

            scalar_dc->codes = dummy_code;
            scalar_dc->code_size = 0;
            simd_dc->codes = dummy_code;
            simd_dc->code_size = 0;
            EXPECT_FLOAT_EQ(scalar_dc->symmetric_dis(0, 1), 0.0f);
            EXPECT_FLOAT_EQ(simd_dc->symmetric_dis(0, 1), 0.0f);
        }
    }
}
