/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 *
 * SQ-RVV review-response tests (see
 * docs/sq_rvv_review_analysis_and_test_plan.md for the full analysis):
 *
 *   D1  CounterexampleRegression — uniform L2 half-grid semantics (the
 *       review's literal q == recon(c) counterexample)
 *   D2  TailBoundaryParity      — d past VLMAX so the tail pass runs
 *       (may pass on well-behaved hardware; regression + VLEN sensitivity)
 *   D3  DirectBitExact          — in-contract integer data => exactly equal
 *   D5  OverflowGuard           — large-d float-domain parity for the direct
 *       codecs (the kernels accumulate in f32; f32 saturates near 3.4e38,
 *       so there is no integer-wraparound failure mode to guard)
 *   D5b LargeDimParity          — large-d direct-codec parity vs the exact
 *       double reference (regression net for the old per-lane i32/u32
 *       wraparound and the signed-IP query-bias overflow, both gone now
 *       that the kernels are float-domain)
 *   D6  ZeroDim                 — d == 0 must return 0, not hang; each case
 *       runs in a forked child under alarm() so a hang only fails its own
 *       case (POSIX only; skipped on Windows hosts)
 *   D7  SpecialQueries          — clamp boundaries / systematic-value
 *       queries / huge-but-finite magnitudes
 *   D8  SmallDimensionParity    — odd & very small d (1/2/3/5/7) across every
 *       qtype x metric (packing tail: 4-bit odd-d padding, 6-bit ng==0)
 *   D9  ReconQuery              — q == recon(code) => L2 distance ~ 0, for
 *       every qtype (not just uniform)
 *   D10 FloatQueryParity        — direct codecs retain FLOAT-query
 *       semantics: fractional queries must match the scalar NONE
 *       reference (the review's literal q=0.75 counterexample; an
 *       earlier integer-domain kernel truncated the query and ranked
 *       candidates differently)
 *
 * Expected status against the CURRENT sq-rvv.cpp (all codecs, direct
 * included, are float-domain in the query):
 *   D1: PASS (uniform L2/IP float-domain semantics)
 *   D2: PASS expected on board
 *   D3: PASS expected (in-contract integer data keeps f32 sums exact)
 *   D5: PASS (float-domain, no wraparound)
 *   D5b: PASS (float-domain, no wraparound / bias overflow)
 *   D6: PASS (all kernels guard vsetvl(0) with `d > 0 ? d : 1`)
 *   D7: PASS (exact float-domain)
 *   D8: PASS expected
 *   D9: PASS expected
 *   D10: PASS (float-query semantics retained)
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <random>
#include <vector>

#ifndef _WIN32
#include <sys/wait.h>
#include <unistd.h>
#include <csignal>
#endif

// D5b derives the test dimensions from the board's real VLMAX.
#if defined(__riscv) && defined(COMPILE_SIMD_RISCV_RVV)
#include <riscv_vector.h>
#define FAISS_RVV_HAS_INTRINSICS 1
#endif

#include <faiss/IndexIVF.h>
#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/impl/scalar_quantizer/distance_computers.h>
#include <faiss/impl/scalar_quantizer/quantizers.h>
#include <faiss/impl/scalar_quantizer/scanners.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/random.h>
#include <faiss/utils/simd_levels.h>

namespace {

using SQ = faiss::ScalarQuantizer;
using faiss::METRIC_INNER_PRODUCT;
using faiss::METRIC_L2;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

struct ScopedSIMDLevel {
    faiss::SIMDLevel original;
    explicit ScopedSIMDLevel(faiss::SIMDLevel level)
            : original(faiss::SIMDConfig::get_level()) {
        faiss::SIMDConfig::set_level(level);
    }
    ~ScopedSIMDLevel() {
        faiss::SIMDConfig::set_level(original);
    }
};

std::vector<float> make_random_vectors(size_t n, size_t d, int seed = 1234) {
    std::vector<float> x(n * d);
    faiss::float_randn(x.data(), x.size(), seed);
    return x;
}

/// Integer-valued floats uniformly in [lo, hi] — exact in f32.
static std::vector<float> random_int_floats(
        size_t n,
        size_t d,
        int lo,
        int hi,
        int seed) {
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> dist(lo, hi);
    std::vector<float> x(n * d);
    for (size_t i = 0; i < x.size(); i++)
        x[i] = static_cast<float>(dist(rng));
    return x;
}

static const char* metric_name(faiss::MetricType m) {
    return m == METRIC_L2 ? "L2" : "IP";
}

/// Codec family classification (drives data/query policy in D2/D7).
enum class QClass { FloatDomain, UniformApprox, Direct };

static QClass class_of(SQ::QuantizerType qtype) {
    switch (qtype) {
        case SQ::QT_4bit_uniform:
        case SQ::QT_8bit_uniform:
            return QClass::UniformApprox;
        case SQ::QT_8bit_direct:
        case SQ::QT_8bit_direct_signed:
            return QClass::Direct;
        default:
            return QClass::FloatDomain;
    }
}

static bool is_direct(SQ::QuantizerType qtype) {
    return class_of(qtype) == QClass::Direct;
}

/// All SQ qtypes with RVV specializations.
static const std::vector<SQ::QuantizerType>& all_rvv_qtypes() {
    static const std::vector<SQ::QuantizerType> qtypes = {
            SQ::QT_4bit,
            SQ::QT_4bit_uniform,
            SQ::QT_6bit,
            SQ::QT_8bit,
            SQ::QT_8bit_uniform,
            SQ::QT_8bit_direct,
            SQ::QT_8bit_direct_signed,
            SQ::QT_fp16,
            SQ::QT_bf16,
    };
    return qtypes;
}

/// One scalar (NONE) + one RVV distance computer over the same trained
/// parameters and code array. codes may be null when only query_to_code on
/// caller-supplied bytes is used.
struct DcPair {
    std::unique_ptr<SQ::SQDistanceComputer> scalar_dc;
    std::unique_ptr<SQ::SQDistanceComputer> rvv_dc;
};

static DcPair make_dc_pair(
        SQ::QuantizerType qtype,
        faiss::MetricType metric,
        size_t d,
        const std::vector<float>& trained,
        const uint8_t* codes,
        size_t code_size) {
    DcPair p;
    p.scalar_dc.reset(
            faiss::scalar_quantizer::sq_select_distance_computer<
                    faiss::SIMDLevel::NONE>(metric, qtype, d, trained));
    {
        ScopedSIMDLevel _(faiss::SIMDLevel::RISCV_RVV);
        p.rvv_dc.reset(
                faiss::scalar_quantizer::sq_select_distance_computer<
                        faiss::SIMDLevel::RISCV_RVV>(
                        metric, qtype, d, trained));
    }
    EXPECT_NE(p.scalar_dc, nullptr);
    EXPECT_NE(p.rvv_dc, nullptr);
    if (p.scalar_dc) {
        p.scalar_dc->codes = codes;
        p.scalar_dc->code_size = code_size;
    }
    if (p.rvv_dc) {
        p.rvv_dc->codes = codes;
        p.rvv_dc->code_size = code_size;
    }
    return p;
}

/// trained-layout access: uniform = [vmin, vdiff]; nonuniform = [vmin(d),
/// vdiff(d)]; fp16/bf16/direct have no trained range.
static void trained_range(
        SQ::QuantizerType qtype,
        size_t d,
        const std::vector<float>& trained,
        float& vmin,
        float& vdiff,
        bool& has_range) {
    has_range = false;
    vmin = 0.0f;
    vdiff = 1.0f;
    switch (class_of(qtype)) {
        case QClass::UniformApprox:
            if (trained.size() >= 2) {
                vmin = trained[0];
                vdiff = trained[1];
                has_range = true;
            }
            break;
        case QClass::FloatDomain:
            // fp16/bf16 have empty trained; 4/6/8bit nonuniform have 2*d
            if (trained.size() >= 2 * d && d > 0) {
                vmin = trained[0];
                vdiff = trained[d];
                has_range = true;
            }
            break;
        case QClass::Direct:
            break;
    }
}

// ===========================================================================
// D1 — Uniform L2 must keep the scalar float-domain semantics.
//
// Scalar: L2(q, code) = (q - recon(c))^2 with recon(c) = vmin +
// vdiff*(c+0.5)/max_code — a HALF-INTEGER grid in code units.
//
// The float-domain kernels retain the original query (no integer-grid
// pre-quantization), so they must match the scalar argmin and distance.
// Sweep A pins the review's literal counterexample (q = recon(c) must be
// distance 0 to code c); sweep B pins full parity at integer queries.
// ===========================================================================

struct UniformL2Params {
    SQ::QuantizerType qtype;
    int nbits; // 4 or 8
    const char* name;
};

class SQRVVCounterexample : public ::testing::TestWithParam<UniformL2Params> {};

TEST_P(SQRVVCounterexample, HalfGridCounterexample) {
    const auto& p = GetParam();
    if (!faiss::SIMDConfig::is_simd_level_available(
                faiss::SIMDLevel::RISCV_RVV)) {
        GTEST_SKIP() << "RISCV_RVV not available";
    }
    const int max_code = (1 << p.nbits) - 1; // 15 or 255
    const size_t d = 1;
    // vmin=0, vdiff=max_code => recon(c) = c + 0.5, code-unit step exactly 1
    std::vector<float> trained = {0.0f, float(max_code)};
    // code_size: d=1 -> 1 byte for both codecs
    const size_t code_size = 1;

    DcPair dc =
            make_dc_pair(p.qtype, METRIC_L2, d, trained, nullptr, code_size);
    ASSERT_TRUE(dc.scalar_dc && dc.rvv_dc);

    // Codes: dim 0 lives in the low nibble for Codec4bit; byte value == code.
    std::vector<uint8_t> bytes(max_code + 1);
    for (int c = 0; c <= max_code; c++) {
        bytes[c] = static_cast<uint8_t>(c);
    }

    // ---- Sweep A: q = recon(c) = c + 0.5 (the review counterexample) ----
    for (int code = 0; code <= max_code; code++) {
        std::vector<float> q = {float(code) + 0.5f};
        dc.scalar_dc->set_query(q.data());
        dc.rvv_dc->set_query(q.data());

        std::vector<float> ref(max_code + 1), tst(max_code + 1);
        for (int c = 0; c <= max_code; c++) {
            ref[c] = dc.scalar_dc->query_to_code(&bytes[c]);
            tst[c] = dc.rvv_dc->query_to_code(&bytes[c]);
        }

        // (a) distance to the reconstruction point of `code` is minimal
        int ref_argmin =
                int(std::min_element(ref.begin(), ref.end()) - ref.begin());
        int tst_argmin =
                int(std::min_element(tst.begin(), tst.end()) - tst.begin());
        EXPECT_EQ(ref_argmin, code)
                << p.name << " sweep A: scalar argmin wrong at code=" << code;
        EXPECT_EQ(tst_argmin, code)
                << p.name << " sweep A: RVV argmin=" << tst_argmin
                << ", expected " << code << " (half-grid semantics violated)";

        // (b) RVV distance at the reconstruction point is ~0
        EXPECT_NEAR(tst[code], 0.0f, 1e-3f)
                << p.name << " sweep A: q=recon(" << code
                << ") but RVV distance to code " << code << " = " << tst[code];
    }

    // ---- Sweep B: q = c (integer code units) — full parity required ----
    for (int code = 0; code <= max_code; code++) {
        std::vector<float> q = {float(code)};
        dc.scalar_dc->set_query(q.data());
        dc.rvv_dc->set_query(q.data());
        for (int c = 0; c <= max_code; c++) {
            float ref = dc.scalar_dc->query_to_code(&bytes[c]);
            float tst = dc.rvv_dc->query_to_code(&bytes[c]);
            // tie cases (c == code +/- 0.5 recon) are exact for scalar too
            EXPECT_NEAR(tst, ref, 1e-3f)
                    << p.name << " sweep B: q=" << code << " code=" << c
                    << " ref=" << ref << " rvv=" << tst;
        }
    }
}

INSTANTIATE_TEST_SUITE_P(
        D1,
        SQRVVCounterexample,
        ::testing::Values(
                UniformL2Params{SQ::QT_4bit_uniform, 4, "4bit_uniform_L2"},
                UniformL2Params{SQ::QT_8bit_uniform, 8, "8bit_uniform_L2"}));

/// vdiff == 0: the trained range collapses to a single point recon(c) ==
/// vmin for every code. The scalar distance is therefore NOT the all-zero
/// defect:
///     L2(q, code) = sum_i (q_i - vmin)^2
///     IP(q, code) = vmin * sum_i q_i
/// Covers every uniform codec under both metrics — the float-domain kernels
/// must match the scalar NONE reference to reassociation precision.
/// Case A: all-zero code. Case B: max code (0xFF / 0x0F) with a huge
/// finite query — the review's literal counterexample. An intermediate
/// sum(q_i*c_i) overflows to inf there, and the retired factorization
/// K_q + scale*sum(q_i*c_i) evaluated 0*inf == NaN; the per-dim
/// pre-scaled form K_q + sum((q_i*scale)*c_i) must return exactly what
/// the scalar reference returns (0 for vmin==0, vmin*sum(q) otherwise),
/// finite in every case.
struct ZeroVdiffParams {
    SQ::QuantizerType qtype;
    faiss::MetricType metric;
    const char* name;
};

class SQRVVZeroVdiff : public ::testing::TestWithParam<ZeroVdiffParams> {};

TEST_P(SQRVVZeroVdiff, UniformVdiffZero) {
    const auto& p = GetParam();
    if (!faiss::SIMDConfig::is_simd_level_available(
                faiss::SIMDLevel::RISCV_RVV)) {
        GTEST_SKIP() << "RISCV_RVV not available";
    }
    const size_t d = 8;
    const float vmin = 3.25f;
    std::vector<float> trained = {vmin, 0.0f}; // vdiff == 0

    std::vector<float> xq = make_random_vectors(1, d, 99);

    // code_size: 4-bit packs two dims per byte, 8-bit one byte per dim.
    const size_t code_size = (p.qtype == SQ::QT_4bit_uniform) ? (d + 1) / 2 : d;

    DcPair dc = make_dc_pair(p.qtype, p.metric, d, trained, nullptr, code_size);
    ASSERT_TRUE(dc.scalar_dc && dc.rvv_dc);
    dc.scalar_dc->set_query(xq.data());
    dc.rvv_dc->set_query(xq.data());

    // ---- Case A: all-zero code (recon == vmin for every dim) ----
    {
        std::vector<uint8_t> zero_code(code_size, 0);
        float ref = dc.scalar_dc->query_to_code(zero_code.data());
        float tst = dc.rvv_dc->query_to_code(zero_code.data());

        float expect = 0.0f;
        for (size_t i = 0; i < d; i++) {
            if (p.metric == METRIC_L2) {
                float diff = xq[i] - vmin;
                expect += diff * diff;
            } else {
                expect += xq[i] * vmin;
            }
        }

        const float tol = 1e-3f * std::max(1.0f, std::abs(expect));
        EXPECT_NEAR(ref, expect, tol) << p.name << ": scalar reference drifted";
        EXPECT_NEAR(tst, expect, tol)
                << p.name << ": vdiff==0 RVV returned " << tst << ", expected "
                << expect << " (all-distances-zero defect)";
    }

    // ---- Case B: max code + huge finite query (review counterexample) ----
    // d=1, trained={0,0}, q=FLT_MAX, code 15/255. IP: the scalar per-dim
    // form is q_i * recon_i = q_i * 0 == 0, so the exact answer is 0; the
    // retired factorization K_q + scale*sum(q_i*c_i) returned NaN there
    // (intermediate sum overflowed to inf, then 0*inf). L2 legitimately
    // overflows in BOTH implementations ((FLT_MAX)^2 -> inf, per-dim
    // semantics), so it only asserts scalar/RVV agreement.
    {
        const size_t d1 = 1;
        std::vector<float> trained1 = {0.0f, 0.0f}; // vdiff == 0, vmin == 0
        DcPair dc1 = make_dc_pair(p.qtype, p.metric, d1, trained1, nullptr, 1);
        ASSERT_TRUE(dc1.scalar_dc && dc1.rvv_dc);
        std::vector<float> q1 = {std::numeric_limits<float>::max()};
        dc1.scalar_dc->set_query(q1.data());
        dc1.rvv_dc->set_query(q1.data());
        std::vector<uint8_t> max_code1(1, 0xFF);
        float ref = dc1.scalar_dc->query_to_code(max_code1.data());
        float tst = dc1.rvv_dc->query_to_code(max_code1.data());
        if (p.metric == METRIC_INNER_PRODUCT) {
            EXPECT_EQ(ref, 0.0f) << p.name << ": scalar reference drifted";
            EXPECT_EQ(tst, 0.0f)
                    << p.name << ": expected exact 0 (recon==0, huge q), got "
                    << tst << " (0*inf NaN from sum-then-multiply?)";
        } else {
            EXPECT_FALSE(std::isfinite(ref))
                    << p.name << ": scalar L2 should overflow";
            EXPECT_EQ(tst, ref)
                    << p.name << ": L2 overflow must agree with scalar";
        }
    }

    // ---- Case C: max code + huge finite query, vmin != 0 (IP only) ----
    // vdiff == 0, vmin = 0.5: the true IP is finite and nonzero
    // (vmin * q), while an overflowing sum(q_i*c_i) intermediate would
    // still yield NaN — catches kernels that collapse the result to
    // zero as well as the NaN defect.
    if (p.metric == METRIC_INNER_PRODUCT) {
        const size_t d1 = 1;
        const float vmin1 = 0.5f;
        std::vector<float> trained1 = {vmin1, 0.0f}; // vdiff == 0
        DcPair dc1 = make_dc_pair(p.qtype, p.metric, d1, trained1, nullptr, 1);
        ASSERT_TRUE(dc1.scalar_dc && dc1.rvv_dc);
        std::vector<float> q1 = {3.0e36f}; // 3e36 * 255 overflows f32
        dc1.scalar_dc->set_query(q1.data());
        dc1.rvv_dc->set_query(q1.data());
        std::vector<uint8_t> max_code1(1, 0xFF);
        float ref = dc1.scalar_dc->query_to_code(max_code1.data());
        float tst = dc1.rvv_dc->query_to_code(max_code1.data());
        EXPECT_TRUE(std::isfinite(tst))
                << p.name << ": Case C returned non-finite " << tst;
        EXPECT_NEAR(tst, ref, std::abs(ref) * 1e-6f)
                << p.name << ": Case C parity, expected " << ref;
        EXPECT_NEAR(ref, vmin1 * q1[0], std::abs(vmin1 * q1[0]) * 1e-6f)
                << p.name << ": Case C scalar reference drifted";
    }
}

INSTANTIATE_TEST_SUITE_P(
        D1,
        SQRVVZeroVdiff,
        ::testing::Values(
                ZeroVdiffParams{
                        SQ::QT_8bit_uniform,
                        METRIC_L2,
                        "8bit_uniform_L2"},
                ZeroVdiffParams{
                        SQ::QT_4bit_uniform,
                        METRIC_L2,
                        "4bit_uniform_L2"},
                ZeroVdiffParams{
                        SQ::QT_8bit_uniform,
                        METRIC_INNER_PRODUCT,
                        "8bit_uniform_IP"},
                ZeroVdiffParams{
                        SQ::QT_4bit_uniform,
                        METRIC_INNER_PRODUCT,
                        "4bit_uniform_IP"}),
        [](const ::testing::TestParamInfo<ZeroVdiffParams>& info) {
            return std::string(info.param.name);
        });

// ===========================================================================
// D1c — Uniform IP query-constant overflow semantics. IP = K_q + S with
//       K_q = sum_i (q_i * c0) accumulated PER DIM. The retired form
//       K_q = c0 * sum(q) overflowed the bare sum(q) to inf first
//       (multi-dim huge finite query), then c0 == 0 produced 0*inf ==
//       NaN; the scalar per-dim form returns exactly 0 (every term
//       q_i * 0 == 0). Same contract checked for both uniform codecs
//       with c0 == 0 (vmin == 0, vdiff == 0).
// ===========================================================================

TEST(SQRVVQueryConstOverflow, HugeQueryZeroC0ReturnsZero) {
    if (!faiss::SIMDConfig::is_simd_level_available(
                faiss::SIMDLevel::RISCV_RVV)) {
        GTEST_SKIP() << "RISCV_RVV not available";
    }
    // d=8 so sum(q) = 8*FLT_MAX overflows f32; each individual q_i*c0
    // term stays exactly 0.
    const size_t d = 8;
    std::vector<float> trained = {0.0f, 0.0f}; // vdiff == 0, vmin == 0
    std::vector<float> q(d, std::numeric_limits<float>::max());

    const size_t code_size_8 = d;
    const size_t code_size_4 = (d + 1) / 2;
    struct Case {
        SQ::QuantizerType qtype;
        faiss::MetricType metric;
        size_t code_size;
        const char* name;
    };
    const Case cases[] = {
            {SQ::QT_8bit_uniform,
             METRIC_INNER_PRODUCT,
             code_size_8,
             "8bit_uniform_IP"},
            {SQ::QT_4bit_uniform,
             METRIC_INNER_PRODUCT,
             code_size_4,
             "4bit_uniform_IP"},
    };
    for (const auto& c : cases) {
        DcPair dc = make_dc_pair(
                c.qtype, c.metric, d, trained, nullptr, c.code_size);
        ASSERT_TRUE(dc.scalar_dc && dc.rvv_dc);
        dc.scalar_dc->set_query(q.data());
        dc.rvv_dc->set_query(q.data());
        std::vector<uint8_t> code(c.code_size, 0xFF);
        float ref = dc.scalar_dc->query_to_code(code.data());
        float tst = dc.rvv_dc->query_to_code(code.data());
        EXPECT_EQ(ref, 0.0f) << c.name << ": scalar per-dim reference drifted";
        EXPECT_EQ(tst, 0.0f) << c.name << ": huge q with c0==0 returned " << tst
                             << " (0*inf NaN from c0*sum(q)?)";
    }
}

// ===========================================================================
// D1b — symmetric_dis overflow semantics. Identical codes must return
//       exactly 0. The retired 8bit_uniform-L2 factored form
//       float(sum(delta_c^2)) * (a*a) returned 0 * inf == NaN when
//       a = vdiff/255 was so large that a*a overflowed; the per-dim
//       scaled form (diff = a*(c1-c2) before squaring) keeps identical
//       codes at exact 0. Same contract checked for all uniform codecs.
// ===========================================================================

TEST(SQRVVSymmetricOverflow, HugeVdiffIdenticalCodesReturnZero) {
    if (!faiss::SIMDConfig::is_simd_level_available(
                faiss::SIMDLevel::RISCV_RVV)) {
        GTEST_SKIP() << "RISCV_RVV not available";
    }
    const size_t d = 8;
    // a = vdiff/255 ~ 1.18e36, a*a ~ 1.4e72 -> inf in f32. Finite codes
    // stay well inside the byte grid; recon itself does not overflow
    // (a*255 == vdiff <= ~3e38).
    const float huge_vdiff = 3.0e38f;
    const float vmin = 0.0f;

    struct Case {
        SQ::QuantizerType qtype;
        size_t code_size;
        const char* name;
    };
    const Case cases[] = {
            {SQ::QT_8bit_uniform, d, "8bit_uniform_L2"},
            {SQ::QT_4bit_uniform, (d + 1) / 2, "4bit_uniform_L2"},
    };
    for (const auto& c : cases) {
        std::vector<float> trained = {vmin, huge_vdiff};
        DcPair dc = make_dc_pair(
                c.qtype, METRIC_L2, d, trained, nullptr, c.code_size);
        ASSERT_TRUE(dc.scalar_dc && dc.rvv_dc);
        // Two all-zero codes (identical pair (0,0) and the degenerate
        // pair (0,1): both dims equal in every code).
        std::vector<uint8_t> zero_codes(2 * c.code_size, 0);
        dc.scalar_dc->codes = zero_codes.data();
        dc.scalar_dc->code_size = c.code_size;
        dc.rvv_dc->codes = zero_codes.data();
        dc.rvv_dc->code_size = c.code_size;

        // (a) identical codes -> exact 0 (the review's counterexample)
        for (size_t i = 0; i < 2; i++) {
            float ref = dc.scalar_dc->symmetric_dis(i, i);
            float tst = dc.rvv_dc->symmetric_dis(i, i);
            EXPECT_EQ(ref, 0.0f) << c.name << ": scalar identical-code drifted";
            EXPECT_EQ(tst, 0.0f)
                    << c.name << ": identical codes returned " << tst
                    << " (0*inf NaN from float(sum)*(a*a)?)";
        }
        // (b) distinct codes -> parity with the scalar per-dim form
        float ref = dc.scalar_dc->symmetric_dis(0, 1);
        float tst = dc.rvv_dc->symmetric_dis(0, 1);
        EXPECT_EQ(tst, ref) << c.name << ": distinct-code parity rvv=" << tst
                            << " scalar=" << ref;
    }
}

// ===========================================================================
// D2 — Tail-boundary parity: dimensions that split the vector loop into
//      (full chunks) + (short tail). The tail pass must not corrupt lanes
//      covered by the final reduction. On well-behaved hardware this may
//      pass even with the tail-agnostic bug; it is the regression net and
//      the VLEN-sensitivity probe (run also under QEMU vlen=256/512).
// ===========================================================================

struct TailParams {
    SQ::QuantizerType qtype;
    faiss::MetricType metric;
    size_t d;
    float tol;
    const char* name;
};

class SQRVVTailBoundary : public ::testing::TestWithParam<TailParams> {};

TEST_P(SQRVVTailBoundary, ParityAtTailDimensions) {
    const auto& p = GetParam();
    if (!faiss::SIMDConfig::is_simd_level_available(
                faiss::SIMDLevel::RISCV_RVV)) {
        GTEST_SKIP() << "RISCV_RVV not available";
    }

    const size_t n_train = 128;
    const size_t n_db = 4;

    // direct codecs: encoder casts float->uint8; keep ALL data/query values
    // on the in-contract integer grid (out-of-contract values make the
    // scalar float-domain reference and the clamping integer kernel
    // incomparable by design).
    const bool direct = is_direct(p.qtype);
    const int dlo = (p.qtype == SQ::QT_8bit_direct_signed) ? -128 : 0;
    const int dhi = (p.qtype == SQ::QT_8bit_direct_signed) ? 127 : 255;

    std::vector<float> xb = direct
            ? random_int_floats(n_train, p.d, dlo, dhi, 42)
            : make_random_vectors(n_train, p.d, 42);
    std::vector<float> xq = direct ? random_int_floats(1, p.d, dlo, dhi, 99)
                                   : make_random_vectors(1, p.d, 99);
    if (p.metric == METRIC_INNER_PRODUCT && !direct) {
        // Normalize for IP parity. NOTE: skip for direct codecs — renorm
        // would leave the integer grid and break the in-contract premise
        // (both computers still see identical data either way).
        faiss::fvec_renorm_L2(p.d, n_train, xb.data());
        faiss::fvec_renorm_L2(p.d, 1, xq.data());
    }

    SQ sq(p.d, p.qtype);
    sq.train(n_train, xb.data());
    std::vector<uint8_t> db_codes(sq.code_size * n_db, 0);
    sq.compute_codes(xb.data(), db_codes.data(), n_db);

    DcPair dc = make_dc_pair(
            p.qtype, p.metric, p.d, sq.trained, db_codes.data(), sq.code_size);
    ASSERT_TRUE(dc.scalar_dc && dc.rvv_dc);

    dc.scalar_dc->set_query(xq.data());
    dc.rvv_dc->set_query(xq.data());

    SCOPED_TRACE(
            ::testing::Message()
            << p.name << " metric=" << metric_name(p.metric) << " d=" << p.d);

    // query_to_code
    for (size_t i = 0; i < n_db; i++) {
        const uint8_t* code = db_codes.data() + i * sq.code_size;
        float ref = dc.scalar_dc->query_to_code(code);
        float tst = dc.rvv_dc->query_to_code(code);
        ASSERT_TRUE(std::isfinite(tst)) << p.name << " non-finite";
        ASSERT_TRUE(std::isfinite(ref)) << p.name << " non-finite ref";
        EXPECT_NEAR(ref, tst, p.tol + std::abs(ref) * 1e-4f)
                << p.name << " query_to_code code=" << i;
    }

    // zero + max codes
    {
        std::vector<uint8_t> zero_code(sq.code_size, 0);
        float ref = dc.scalar_dc->query_to_code(zero_code.data());
        float tst = dc.rvv_dc->query_to_code(zero_code.data());
        EXPECT_NEAR(ref, tst, p.tol + std::abs(ref) * 1e-4f)
                << p.name << " zero code";
    }
    if (p.qtype != SQ::QT_fp16 && p.qtype != SQ::QT_bf16) {
        std::vector<uint8_t> max_code(sq.code_size, 0xff);
        float ref = dc.scalar_dc->query_to_code(max_code.data());
        float tst = dc.rvv_dc->query_to_code(max_code.data());
        EXPECT_NEAR(ref, tst, p.tol + std::abs(ref) * 1e-4f)
                << p.name << " max code";
    }

    // symmetric_dis
    for (size_t i = 0; i + 1 < n_db; i += 2) {
        float ref = dc.scalar_dc->symmetric_dis(i, i + 1);
        float tst = dc.rvv_dc->symmetric_dis(i, i + 1);
        EXPECT_NEAR(ref, tst, p.tol + std::abs(ref) * 1e-4f)
                << p.name << " symmetric_dis(" << i << "," << (i + 1) << ")";
    }

    // batch_4
    {
        const uint8_t *c0 = db_codes.data(),
                      *c1 = db_codes.data() + sq.code_size,
                      *c2 = db_codes.data() + 2 * sq.code_size,
                      *c3 = db_codes.data() + 3 * sq.code_size;
        float r0, r1, r2, r3, t0, t1, t2, t3;
        dc.scalar_dc->query_to_codes_batch_4(c0, c1, c2, c3, r0, r1, r2, r3);
        dc.rvv_dc->query_to_codes_batch_4(c0, c1, c2, c3, t0, t1, t2, t3);
        EXPECT_NEAR(r0, t0, p.tol + std::abs(r0) * 1e-4f)
                << p.name << " batch_4[0]";
        EXPECT_NEAR(r1, t1, p.tol + std::abs(r1) * 1e-4f)
                << p.name << " batch_4[1]";
        EXPECT_NEAR(r2, t2, p.tol + std::abs(r2) * 1e-4f)
                << p.name << " batch_4[2]";
        EXPECT_NEAR(r3, t3, p.tol + std::abs(r3) * 1e-4f)
                << p.name << " batch_4[3]";
    }

    // scanner parity
    std::unique_ptr<faiss::InvertedListScanner> scalar_scanner(
            faiss::scalar_quantizer::sq_select_InvertedListScanner<
                    faiss::SIMDLevel::NONE>(
                    p.qtype,
                    p.metric,
                    p.d,
                    sq.code_size,
                    sq.trained,
                    nullptr,
                    false,
                    nullptr,
                    false));
    ASSERT_NE(scalar_scanner, nullptr);
    scalar_scanner->set_query(xq.data());
    scalar_scanner->set_list(0, 0.0f);

    std::unique_ptr<faiss::InvertedListScanner> rvv_scanner;
    {
        ScopedSIMDLevel _(faiss::SIMDLevel::RISCV_RVV);
        rvv_scanner.reset(sq.select_InvertedListScanner(
                p.metric, nullptr, false, nullptr, false));
    }
    ASSERT_NE(rvv_scanner, nullptr);
    rvv_scanner->set_query(xq.data());
    rvv_scanner->set_list(0, 0.0f);

    for (size_t i = 0; i < n_db; i++) {
        const uint8_t* code = db_codes.data() + i * sq.code_size;
        float ref = scalar_scanner->distance_to_code(code);
        float tst = rvv_scanner->distance_to_code(code);
        EXPECT_NEAR(ref, tst, p.tol + std::abs(ref) * 1e-4f)
                << p.name << " scanner code=" << i;
    }
}

/*
 * Dimension selection (VLEN=128 board): each case guarantees
 * ">=1 full chunk + 0 < tail < VLMAX" for that kernel's SEW/LMUL.
 *   e8m2 on nb=ceil(d/2) bytes (4bit_uniform L2): d=94 -> 47B = 32+15,
 *       d=127 -> 63B = 32+31, d=191 -> 95B = 64+31
 *   e8m2 on d (8bit_uniform L2/IP): 33 = 32+1, 97 = 96+1, 129 = 128+1
 *   e8m1 on nb bytes (4bit nonuniform L2/IP): 65 -> 33B = 32+1,
 *       94 -> 47B = 32+15, 129 -> 65B = 64+1
 *   e8m1 on d (8bit nonuniform): 17 = 16+1, 33 = 32+1, 81 = 80+1
 *   6bit on ng=d/4 groups: 68 -> 17g = 16+1, 69 -> +1 scalar dim,
 *       127 -> 31g = 16+15 + 3 scalar dims
 *   bf16 e16m2: 17 = 16+1, 33 = 32+1;  fp16 e16m1: 9, 17, 23
 * Tolerances: all float-domain codecs (uniform and non-uniform) 1e-3
 * (reassociation); direct in-contract integer data 1e-3 (exact integer
 * sums). Uniform L2/IP are now float-domain — their semantic contract is
 * pinned tightly by D1.
 */
INSTANTIATE_TEST_SUITE_P(
        D2,
        SQRVVTailBoundary,
        ::testing::Values(
                // ---- QT_4bit (nonuniform): e8m1 on ceil(d/2) bytes
                TailParams{SQ::QT_4bit, METRIC_L2, 65, 1e-3f, "4bit/d65"},
                TailParams{SQ::QT_4bit, METRIC_L2, 94, 1e-3f, "4bit/d94"},
                TailParams{SQ::QT_4bit, METRIC_L2, 129, 1e-3f, "4bit/d129"},
                TailParams{
                        SQ::QT_4bit,
                        METRIC_INNER_PRODUCT,
                        65,
                        1e-3f,
                        "4bit_IP/d65"},
                TailParams{
                        SQ::QT_4bit,
                        METRIC_INNER_PRODUCT,
                        94,
                        1e-3f,
                        "4bit_IP/d94"},
                TailParams{
                        SQ::QT_4bit,
                        METRIC_INNER_PRODUCT,
                        129,
                        1e-3f,
                        "4bit_IP/d129"},
                // ---- QT_4bit_uniform: L2 e8m2 on nbv bytes (odd-d
                //      padding byte is excluded -> nbv = (d+1)/2 - 1),
                //      IP e8m1 on nb bytes
                TailParams{
                        SQ::QT_4bit_uniform,
                        METRIC_L2,
                        94,
                        1e-3f,
                        "4bit_uniform/d94"},
                TailParams{
                        SQ::QT_4bit_uniform,
                        METRIC_L2,
                        127,
                        1e-3f,
                        "4bit_uniform/d127"},
                TailParams{
                        SQ::QT_4bit_uniform,
                        METRIC_L2,
                        191,
                        1e-3f,
                        "4bit_uniform/d191"},
                TailParams{
                        SQ::QT_4bit_uniform,
                        METRIC_INNER_PRODUCT,
                        65,
                        1e-3f,
                        "4bit_uniform_IP/d65"},
                TailParams{
                        SQ::QT_4bit_uniform,
                        METRIC_INNER_PRODUCT,
                        129,
                        1e-3f,
                        "4bit_uniform_IP/d129"},
                // ---- QT_6bit: vl=16 groups; d%4 scalar tail at 69/127
                TailParams{SQ::QT_6bit, METRIC_L2, 68, 1e-3f, "6bit/d68"},
                TailParams{SQ::QT_6bit, METRIC_L2, 69, 1e-3f, "6bit/d69"},
                TailParams{SQ::QT_6bit, METRIC_L2, 127, 1e-3f, "6bit/d127"},
                TailParams{
                        SQ::QT_6bit,
                        METRIC_INNER_PRODUCT,
                        68,
                        1e-3f,
                        "6bit_IP/d68"},
                TailParams{
                        SQ::QT_6bit,
                        METRIC_INNER_PRODUCT,
                        127,
                        1e-3f,
                        "6bit_IP/d127"},
                // ---- QT_8bit (nonuniform): e8m1
                TailParams{SQ::QT_8bit, METRIC_L2, 17, 1e-3f, "8bit/d17"},
                TailParams{SQ::QT_8bit, METRIC_L2, 33, 1e-3f, "8bit/d33"},
                TailParams{SQ::QT_8bit, METRIC_L2, 81, 1e-3f, "8bit/d81"},
                TailParams{
                        SQ::QT_8bit,
                        METRIC_INNER_PRODUCT,
                        17,
                        1e-3f,
                        "8bit_IP/d17"},
                TailParams{
                        SQ::QT_8bit,
                        METRIC_INNER_PRODUCT,
                        81,
                        1e-3f,
                        "8bit_IP/d81"},
                // ---- QT_8bit_uniform: e8m2 on d
                TailParams{
                        SQ::QT_8bit_uniform,
                        METRIC_L2,
                        33,
                        1e-3f,
                        "8bit_uniform/d33"},
                TailParams{
                        SQ::QT_8bit_uniform,
                        METRIC_L2,
                        97,
                        1e-3f,
                        "8bit_uniform/d97"},
                TailParams{
                        SQ::QT_8bit_uniform,
                        METRIC_L2,
                        129,
                        1e-3f,
                        "8bit_uniform/d129"},
                TailParams{
                        SQ::QT_8bit_uniform,
                        METRIC_INNER_PRODUCT,
                        33,
                        1e-3f,
                        "8bit_uniform_IP/d33"},
                TailParams{
                        SQ::QT_8bit_uniform,
                        METRIC_INNER_PRODUCT,
                        129,
                        1e-3f,
                        "8bit_uniform_IP/d129"},
                // ---- QT_8bit_direct (in-contract data -> ~exact)
                TailParams{
                        SQ::QT_8bit_direct,
                        METRIC_L2,
                        33,
                        1e-3f,
                        "8bit_direct/d33"},
                TailParams{
                        SQ::QT_8bit_direct,
                        METRIC_L2,
                        129,
                        1e-3f,
                        "8bit_direct/d129"},
                TailParams{
                        SQ::QT_8bit_direct,
                        METRIC_INNER_PRODUCT,
                        33,
                        1e-3f,
                        "8bit_direct_IP/d33"},
                TailParams{
                        SQ::QT_8bit_direct,
                        METRIC_INNER_PRODUCT,
                        129,
                        1e-3f,
                        "8bit_direct_IP/d129"},
                // ---- QT_8bit_direct_signed
                TailParams{
                        SQ::QT_8bit_direct_signed,
                        METRIC_L2,
                        33,
                        1e-3f,
                        "8bit_direct_signed/d33"},
                TailParams{
                        SQ::QT_8bit_direct_signed,
                        METRIC_L2,
                        129,
                        1e-3f,
                        "8bit_direct_signed/d129"},
                TailParams{
                        SQ::QT_8bit_direct_signed,
                        METRIC_INNER_PRODUCT,
                        33,
                        1e-3f,
                        "8bit_direct_signed_IP/d33"},
                TailParams{
                        SQ::QT_8bit_direct_signed,
                        METRIC_INNER_PRODUCT,
                        129,
                        1e-3f,
                        "8bit_direct_signed_IP/d129"},
                // ---- QT_bf16: e16m2 (vl=16)
                TailParams{SQ::QT_bf16, METRIC_L2, 17, 1e-3f, "bf16/d17"},
                TailParams{SQ::QT_bf16, METRIC_L2, 33, 1e-3f, "bf16/d33"},
                TailParams{
                        SQ::QT_bf16,
                        METRIC_INNER_PRODUCT,
                        17,
                        1e-3f,
                        "bf16_IP/d17"},
                TailParams{
                        SQ::QT_bf16,
                        METRIC_INNER_PRODUCT,
                        33,
                        1e-3f,
                        "bf16_IP/d33"},
                // ---- QT_fp16: e16m1 (vl=8)
                TailParams{SQ::QT_fp16, METRIC_L2, 9, 1e-3f, "fp16/d9"},
                TailParams{SQ::QT_fp16, METRIC_L2, 17, 1e-3f, "fp16/d17"},
                TailParams{SQ::QT_fp16, METRIC_L2, 23, 1e-3f, "fp16/d23"},
                TailParams{
                        SQ::QT_fp16,
                        METRIC_INNER_PRODUCT,
                        9,
                        1e-3f,
                        "fp16_IP/d9"},
                TailParams{
                        SQ::QT_fp16,
                        METRIC_INNER_PRODUCT,
                        23,
                        1e-3f,
                        "fp16_IP/d23"}));

// ===========================================================================
// D3 — Direct codecs: in-contract (integer-grid) data must give EXACTLY
//      equal results (both kernels are float-domain now; integer-grid
//      distances and their partial sums stay < 2^24 at these d, so f32
//      represents them exactly and the f32 accumulators cannot round).
// ===========================================================================

struct DirectExactParams {
    SQ::QuantizerType qtype;
    faiss::MetricType metric;
    size_t d;
    const char* name;
};

class SQRVVDirectBitExact : public ::testing::TestWithParam<DirectExactParams> {
};

TEST_P(SQRVVDirectBitExact, InContractExact) {
    const auto& p = GetParam();
    if (!faiss::SIMDConfig::is_simd_level_available(
                faiss::SIMDLevel::RISCV_RVV)) {
        GTEST_SKIP() << "RISCV_RVV not available";
    }
    const size_t n_train = 64;
    const size_t n_db = 4;
    const int lo = (p.qtype == SQ::QT_8bit_direct_signed) ? -128 : 0;
    const int hi = (p.qtype == SQ::QT_8bit_direct_signed) ? 127 : 255;

    std::vector<float> xb = random_int_floats(n_train, p.d, lo, hi, 42);
    std::vector<float> xq = random_int_floats(1, p.d, lo, hi, 99);

    SQ sq(p.d, p.qtype);
    sq.train(n_train, xb.data());
    std::vector<uint8_t> codes(sq.code_size * n_db, 0);
    sq.compute_codes(xb.data(), codes.data(), n_db);

    DcPair dc = make_dc_pair(
            p.qtype, p.metric, p.d, sq.trained, codes.data(), sq.code_size);
    ASSERT_TRUE(dc.scalar_dc && dc.rvv_dc);

    // queries: random in-contract + the two clamp edges (all integer-grid)
    std::vector<std::vector<float>> queries;
    queries.push_back(xq);
    {
        std::vector<float> q_edge(p.d);
        for (size_t i = 0; i < p.d; i++) {
            q_edge[i] = (i % 2 == 0) ? float(lo) : float(hi);
        }
        queries.push_back(std::move(q_edge));
    }

    for (size_t qi = 0; qi < queries.size(); qi++) {
        dc.scalar_dc->set_query(queries[qi].data());
        dc.rvv_dc->set_query(queries[qi].data());
        for (size_t i = 0; i < n_db; i++) {
            const uint8_t* code = codes.data() + i * sq.code_size;
            float ref = dc.scalar_dc->query_to_code(code);
            float tst = dc.rvv_dc->query_to_code(code);
            ASSERT_NEAR(tst, ref, 0.0f)
                    << p.name << " query=" << qi << " code=" << i
                    << " ref=" << ref << " rvv=" << tst
                    << " — in-contract data must be bit-exact";
        }
        float rs = dc.scalar_dc->symmetric_dis(0, 1);
        float ts = dc.rvv_dc->symmetric_dis(0, 1);
        EXPECT_EQ(rs, ts) << p.name << " query=" << qi << " symmetric_dis";
    }
}

INSTANTIATE_TEST_SUITE_P(
        D3,
        SQRVVDirectBitExact,
        ::testing::Values(
                DirectExactParams{
                        SQ::QT_8bit_direct,
                        METRIC_L2,
                        33,
                        "8bit_direct_L2_d33"},
                DirectExactParams{
                        SQ::QT_8bit_direct,
                        METRIC_L2,
                        96,
                        "8bit_direct_L2_d96"},
                DirectExactParams{
                        SQ::QT_8bit_direct,
                        METRIC_L2,
                        129,
                        "8bit_direct_L2_d129"},
                DirectExactParams{
                        SQ::QT_8bit_direct,
                        METRIC_INNER_PRODUCT,
                        33,
                        "8bit_direct_IP_d33"},
                DirectExactParams{
                        SQ::QT_8bit_direct,
                        METRIC_INNER_PRODUCT,
                        129,
                        "8bit_direct_IP_d129"},
                DirectExactParams{
                        SQ::QT_8bit_direct_signed,
                        METRIC_L2,
                        33,
                        "8bit_direct_signed_L2_d33"},
                DirectExactParams{
                        SQ::QT_8bit_direct_signed,
                        METRIC_L2,
                        96,
                        "8bit_direct_signed_L2_d96"},
                DirectExactParams{
                        SQ::QT_8bit_direct_signed,
                        METRIC_L2,
                        129,
                        "8bit_direct_signed_L2_d129"},
                DirectExactParams{
                        SQ::QT_8bit_direct_signed,
                        METRIC_INNER_PRODUCT,
                        33,
                        "8bit_direct_signed_IP_d33"},
                DirectExactParams{
                        SQ::QT_8bit_direct_signed,
                        METRIC_INNER_PRODUCT,
                        129,
                        "8bit_direct_signed_IP_d129"}));

// ===========================================================================
// D5 — Large-d parity for the direct codecs. The kernels are float-domain
//      (f32 accumulators saturate near 3.4e38), so magnitudes that used to
//      overflow 32-bit integer accumulators must simply stay in parity with
//      the scalar reference.
// ===========================================================================

static void run_overflow_case(
        SQ::QuantizerType qtype,
        faiss::MetricType metric,
        size_t d,
        uint8_t code_byte,
        float q_value,
        const char* name,
        double expected_exact) {
    if (!faiss::SIMDConfig::is_simd_level_available(
                faiss::SIMDLevel::RISCV_RVV)) {
        GTEST_SKIP() << "RISCV_RVV not available";
    }
    // direct codecs ignore trained params; d=34000 training would be
    // wasteful, so hand the computer a minimal trained vector.
    std::vector<float> trained;
    DcPair dc = make_dc_pair(qtype, metric, d, trained, nullptr, d);
    ASSERT_TRUE(dc.scalar_dc && dc.rvv_dc);

    std::vector<float> q(d, q_value);
    dc.scalar_dc->set_query(q.data());
    dc.rvv_dc->set_query(q.data());

    std::vector<uint8_t> code(d, code_byte);
    float ref = dc.scalar_dc->query_to_code(code.data());
    float tst = dc.rvv_dc->query_to_code(code.data());

    // Scalar reference accumulates in f32 — allow 1e-3 relative slack;
    // the RVV float kernel matches it to reassociation precision.
    const float tol = float(expected_exact * 1e-3);
    EXPECT_NEAR(tst, expected_exact, tol)
            << name << " d=" << d << " rvv=" << tst
            << " expected=" << expected_exact;
    EXPECT_NEAR(ref, expected_exact, tol) << name << " scalar ref drifted";
}

TEST(D5Overflow, DirectL2LargeDim) {
    // 65025 per dim — magnitudes that overflowed the old i32 accumulator
    // at d > ~33025; the float kernel must track the scalar reference.
    const size_t d = 34000;
    run_overflow_case(
            SQ::QT_8bit_direct,
            METRIC_L2,
            d,
            0x00,
            255.0f,
            "8bit_direct_L2",
            double(d) * 65025.0);
}

TEST(D5Overflow, DirectIPLargeDim) {
    const size_t d = 68000;
    run_overflow_case(
            SQ::QT_8bit_direct,
            METRIC_INNER_PRODUCT,
            d,
            0xff,
            255.0f,
            "8bit_direct_IP",
            double(d) * 65025.0);
}

TEST(D5Overflow, DirectSignedL2LargeDim) {
    // query 127 in value space <-> storage byte 127+128 = 255;
    // code byte 0x00 (value -128): diff 127-(-128) = 255 -> 65025 per dim.
    const size_t d = 34000;
    run_overflow_case(
            SQ::QT_8bit_direct_signed,
            METRIC_L2,
            d,
            0x00,
            127.0f,
            "8bit_direct_signed_L2",
            double(d) * 65025.0);
}

// ===========================================================================
// D5b — Large-d parity at (and past) the old per-lane i32/u32 wrap point
//       and the signed-IP query-bias overflow point. These dimensions sat
//       exactly past the wrap thresholds of the retired integer-domain
//       kernels; they are a regression net pinning the float-domain
//       semantics at scale (RVV vs scalar parity + exact double reference).
//       The chunk count is parameterized per case so EVERY integer
//       accumulator a regression could reintroduce is pushed past its own
//       wrap threshold:
//         - L2 / signed i32 lanes:        ceil(2^31 / 65025) = 33026
//         - IP / unsigned u32 lanes:      ceil(2^32 / 65025) = 66052
//         - signed-IP worst-case term     ceil(2^31 / 32640) = 65794
//           (qs = -128 * code value +127 -> |term| = 32640 per chunk)
// ===========================================================================

static void run_flush_case(
        SQ::QuantizerType qtype,
        faiss::MetricType metric,
        uint8_t code_byte,
        float q_value,
        size_t n_chunks,
        const char* name,
        double per_dim) {
    if (!faiss::SIMDConfig::is_simd_level_available(
                faiss::SIMDLevel::RISCV_RVV)) {
        GTEST_SKIP() << "RISCV_RVV not available";
    }
    // VLMAX at this VLEN, derived at runtime so the dimensions below sit
    // past the old wrap thresholds on VLEN=128/256/... boards alike.
#if defined(FAISS_RVV_HAS_INTRINSICS)
    const size_t vl = __riscv_vsetvlmax_e8m2();
#else
    GTEST_SKIP() << "no RVV intrinsics at test compile time";
    const size_t vl = 32;
#endif
    // d = n_chunks full per-lane chunks + 1 tail dim. n_chunks must be
    // chosen per case to cross THAT case's integer-accumulator threshold
    // (see the derivation in the block comment above); the tail dim adds
    // one extra term to lane 0 only and never rescues an undershoot.
    const size_t d = vl * n_chunks + 1;

    std::vector<float> trained;
    DcPair dc = make_dc_pair(qtype, metric, d, trained, nullptr, d);
    ASSERT_TRUE(dc.scalar_dc && dc.rvv_dc);

    std::vector<float> q(d, q_value);
    dc.scalar_dc->set_query(q.data());
    dc.rvv_dc->set_query(q.data());

    std::vector<uint8_t> code(d, code_byte);
    // Exact reference computed in the test (double: no drift at these
    // magnitudes). Both kernels accumulate in f32, so the sequential /
    // lane-reassociated sums drift by up to ~n*ulp(intermediate) — about
    // 1e-2 relative at d ~ 5e5 (ulp ~4096 once the partial sum passes
    // 2^35). That drift is still 100x tighter than a 2^31 integer
    // wraparound (>= 100% error), which is the failure mode this test is
    // a regression net for.
    const double expected_exact = per_dim * double(d);
    const float ref = dc.scalar_dc->query_to_code(code.data());
    const float tst = dc.rvv_dc->query_to_code(code.data());
    const double tol = std::fabs(expected_exact) * 2e-2;
    EXPECT_NEAR(tst, expected_exact, tol)
            << name << " d=" << d << " rvv=" << tst
            << " expected=" << expected_exact;
    EXPECT_NEAR(ref, tst, tol)
            << name << " d=" << d << " rvv=" << tst << " scalar=" << ref;
}

TEST(D5bLargeDim, DirectL2ParityPastOldWrap) {
    // q=255, code=0: 65025 per dim. 33026 chunks push a per-lane signed
    // i32 accumulator past 2^31 (33026*65025 = 2,147,565,650).
    run_flush_case(
            SQ::QT_8bit_direct,
            METRIC_L2,
            0x00,
            255.0f,
            33026,
            "8bit_direct_L2_large",
            65025.0);
}

TEST(D5bLargeDim, DirectIPParityPastOldWrap) {
    // q=255, code=255: 65025 per dim, unsigned domain. An unsigned u32
    // lane accumulator only wraps at 2^32: 66052 chunks are needed
    // (66052*65025 = 4,295,215,800 > 2^32; 66051 falls short).
    run_flush_case(
            SQ::QT_8bit_direct,
            METRIC_INNER_PRODUCT,
            0xff,
            255.0f,
            66052,
            "8bit_direct_IP_large",
            65025.0);
}

TEST(D5bLargeDim, DirectSignedL2ParityPastOldWrap) {
    // q=-128 vs code 0xff (value 127): diff -128-127 = -255 -> 65025/dim.
    // Signed i32 lanes wrap past 2^31 at 33026 chunks.
    run_flush_case(
            SQ::QT_8bit_direct_signed,
            METRIC_L2,
            0xff,
            -128.0f,
            33026,
            "8bit_direct_signed_L2_large",
            65025.0);
}

TEST(D5bLargeDim, DirectSignedIPParityPastWorstTermWrap) {
    // Worst-case SIGNED-IP decomposition term: query -128 against code
    // 0xff (value +127) -> q_s * byte = -128*255 = -32640 per dim, the
    // largest magnitude a per-lane i32 accumulator sees before the bias
    // split. Signed i32 lanes wrap past 2^31 at 65794 chunks
    // (65793*32640 = 2,147,483,520 < 2^31; 65794 crosses). The exact
    // value-domain result stays -16256 * d.
    run_flush_case(
            SQ::QT_8bit_direct_signed,
            METRIC_INNER_PRODUCT,
            0xff,
            -128.0f,
            65794,
            "8bit_direct_signed_IP_large",
            -16256.0);
}

TEST(D5bLargeDim, DirectSignedIPParityPastOldBiasOverflow) {
    // An all-127 query makes the old integer bias 128*sum(q) cross 2^31
    // at d = 132105 (int32_t UB there). The float kernel must stay in
    // parity: exact = d * 127 * (0 - 128) = -16256 * d.
    if (!faiss::SIMDConfig::is_simd_level_available(
                faiss::SIMDLevel::RISCV_RVV)) {
        GTEST_SKIP() << "RISCV_RVV not available";
    }
    const size_t d = 132105;
    std::vector<float> trained;
    DcPair dc = make_dc_pair(
            SQ::QT_8bit_direct_signed,
            METRIC_INNER_PRODUCT,
            d,
            trained,
            nullptr,
            d);
    ASSERT_TRUE(dc.scalar_dc && dc.rvv_dc);

    std::vector<float> q(d, 127.0f);
    dc.scalar_dc->set_query(q.data());
    dc.rvv_dc->set_query(q.data());
    std::vector<uint8_t> code(d, 0x00); // value -128: qs*(c-128) = 127*128

    const double expected_exact = -16256.0 * double(d);
    const float ref = dc.scalar_dc->query_to_code(code.data());
    const float tst = dc.rvv_dc->query_to_code(code.data());
    // f32 accumulation drift bound: ~n*ulp(intermediate), ~2e-2 relative
    // here (vs the >=100% error a 2^31 bias overflow would produce).
    const double tol = std::fabs(expected_exact) * 2e-2;
    EXPECT_NEAR(tst, expected_exact, tol)
            << "signed-IP bias overflow at d=" << d << " rvv=" << tst;
    EXPECT_NEAR(ref, tst, tol) << "signed-IP parity at d=" << d
                               << " rvv=" << tst << " scalar=" << ref;
}

// ===========================================================================
// D10 — Direct codecs retain FLOAT-query semantics. The review's literal
//       counterexample: d=1, q=0.75, codes {0, 1} — the scalar L2
//       distances are 0.5625 / 0.0625 so code 1 must win. An earlier
//       integer-domain RVV kernel truncated the query to 0, flipping the
//       ranking. Fractional queries must match the scalar NONE reference
//       for all four direct codec x metric combinations.
// ===========================================================================

static void run_float_query_case(
        SQ::QuantizerType qtype,
        faiss::MetricType metric,
        const char* name) {
    if (!faiss::SIMDConfig::is_simd_level_available(
                faiss::SIMDLevel::RISCV_RVV)) {
        GTEST_SKIP() << "RISCV_RVV not available";
    }
    const size_t d = 1;
    std::vector<float> trained;
    DcPair dc = make_dc_pair(qtype, metric, d, trained, nullptr, d);
    ASSERT_TRUE(dc.scalar_dc && dc.rvv_dc);

    // ---- The review's literal counterexample ----
    // codes 0 and 1 (storage byte = value for direct; value + 128 for
    // signed direct). q = 0.75 must sit strictly closer to code 1.
    {
        uint8_t bytes[2];
        float r[2], t[2];
        if (qtype == SQ::QT_8bit_direct_signed) {
            bytes[0] = 0 + 128; // value 0
            bytes[1] = 1 + 128; // value 1
        } else {
            bytes[0] = 0;
            bytes[1] = 1;
        }
        std::vector<float> q = {0.75f};
        dc.scalar_dc->set_query(q.data());
        dc.rvv_dc->set_query(q.data());
        r[0] = dc.scalar_dc->query_to_code(&bytes[0]);
        r[1] = dc.scalar_dc->query_to_code(&bytes[1]);
        t[0] = dc.rvv_dc->query_to_code(&bytes[0]);
        t[1] = dc.rvv_dc->query_to_code(&bytes[1]);

        // Scalar L2 reference: 0.5625 and 0.0625.
        if (metric == METRIC_L2) {
            EXPECT_NEAR(r[0], 0.5625f, 1e-6f)
                    << name << " scalar d(q=0.75, code0)";
            EXPECT_NEAR(r[1], 0.0625f, 1e-6f)
                    << name << " scalar d(q=0.75, code1)";
        }
        EXPECT_NEAR(t[0], r[0], 1e-5f)
                << name << " RVV d(q=0.75, code0)=" << t[0]
                << " scalar=" << r[0] << " (query truncated?)";
        EXPECT_NEAR(t[1], r[1], 1e-5f)
                << name << " RVV d(q=0.75, code1)=" << t[1]
                << " scalar=" << r[1] << " (query truncated?)";
        // The ranking must match the scalar reference: code 1 wins under
        // L2; under IP, code 1 has the larger score.
        if (metric == METRIC_L2) {
            EXPECT_LT(t[1], t[0]) << name
                                  << " ranking flipped for q=0.75 (truncated "
                                     "query picked code 0)";
        } else {
            EXPECT_GT(t[1], t[0]) << name << " ranking flipped for q=0.75";
        }
    }

    // ---- Random fractional queries vs both candidate codes ----
    {
        std::mt19937 rng(7);
        std::uniform_real_distribution<float> frac(-10.0f, 265.0f);
        for (int trial = 0; trial < 64; trial++) {
            std::vector<float> q = {frac(rng)};
            dc.scalar_dc->set_query(q.data());
            dc.rvv_dc->set_query(q.data());
            for (int c = 0; c <= 255; c += 1) {
                uint8_t byte;
                float value;
                if (qtype == SQ::QT_8bit_direct_signed) {
                    byte = uint8_t(c + 128);
                    value = float(c - 128);
                } else {
                    byte = uint8_t(c);
                    value = float(c);
                }
                float ref = dc.scalar_dc->query_to_code(&byte);
                float tst = dc.rvv_dc->query_to_code(&byte);
                ASSERT_TRUE(std::isfinite(ref) && std::isfinite(tst))
                        << name << " non-finite at q=" << q[0] << " code=" << c;
                EXPECT_NEAR(tst, ref, 1e-3f + std::abs(ref) * 1e-4f)
                        << name << " q=" << q[0] << " code value=" << value;
            }
        }
    }
}

TEST(D10FloatQuery, DirectL2Counterexample) {
    run_float_query_case(SQ::QT_8bit_direct, METRIC_L2, "8bit_direct_L2");
}

TEST(D10FloatQuery, DirectIPFractionalParity) {
    run_float_query_case(
            SQ::QT_8bit_direct, METRIC_INNER_PRODUCT, "8bit_direct_IP");
}

TEST(D10FloatQuery, DirectSignedL2Counterexample) {
    run_float_query_case(
            SQ::QT_8bit_direct_signed, METRIC_L2, "8bit_direct_signed_L2");
}

TEST(D10FloatQuery, DirectSignedIPFractionalParity) {
    run_float_query_case(
            SQ::QT_8bit_direct_signed,
            METRIC_INNER_PRODUCT,
            "8bit_direct_signed_IP");
}

// ===========================================================================
// D6 — d == 0 must return 0 without hanging. Kernels guard vsetvl with
//      `d > 0 ? d : 1` and the tail with `remaining > 0`, so d == 0 must
//      terminate immediately. Each case runs in a forked child under
//      alarm() so a hang only fails its own case (POSIX only).
// ===========================================================================

struct ZeroDimParams {
    SQ::QuantizerType qtype;
    faiss::MetricType metric;
    const char* name;
};

class SQRVVZeroDim : public ::testing::TestWithParam<ZeroDimParams> {};

#ifndef _WIN32
TEST_P(SQRVVZeroDim, ReturnsZeroNoHang) {
    const auto& p = GetParam();
    if (!faiss::SIMDConfig::is_simd_level_available(
                faiss::SIMDLevel::RISCV_RVV)) {
        GTEST_SKIP() << "RISCV_RVV not available";
    }

    // Minimal trained vector per codec family (d=0 loops never read it,
    // but the uniform constructors index [0] and [1]).
    std::vector<float> trained;
    if (class_of(p.qtype) == QClass::UniformApprox) {
        trained = {0.0f, 1.0f};
    }

    const size_t d = 0;

    pid_t pid = fork();
    ASSERT_GE(pid, 0) << "fork failed";
    if (pid == 0) {
        // child: run the kernel under a hard deadline; _exit a status code
        alarm(20);
        DcPair dc = [&] {
            DcPair dp;
            dp.scalar_dc.reset(
                    faiss::scalar_quantizer::sq_select_distance_computer<
                            faiss::SIMDLevel::NONE>(
                            p.metric, p.qtype, d, trained));
            {
                ScopedSIMDLevel _(faiss::SIMDLevel::RISCV_RVV);
                dp.rvv_dc.reset(
                        faiss::scalar_quantizer::sq_select_distance_computer<
                                faiss::SIMDLevel::RISCV_RVV>(
                                p.metric, p.qtype, d, trained));
            }
            return dp;
        }();
        if (!dc.scalar_dc || !dc.rvv_dc)
            _exit(3);
        float q_dummy = 0.0f;
        dc.scalar_dc->set_query(&q_dummy);
        dc.rvv_dc->set_query(&q_dummy);
        uint8_t code_dummy = 0;
        float ref = dc.scalar_dc->query_to_code(&code_dummy);
        float tst = dc.rvv_dc->query_to_code(&code_dummy);
        if (!std::isfinite(tst) || !std::isfinite(ref))
            _exit(4);
        if (tst != 0.0f)
            _exit(1); // must be exactly 0 for d=0
        if (std::abs(ref - tst) > 1e-6f)
            _exit(2); // parity broken
        _exit(0);
    }

    int status = 0;
    waitpid(pid, &status, 0);
    if (WIFSIGNALED(status) && WTERMSIG(status) == SIGALRM) {
        FAIL() << "qtype=" << int(p.qtype) << " " << metric_name(p.metric)
               << ": d=0 kernel HUNG (vsetvl(0)==0 loop) — killed after 20s";
    } else if (WIFEXITED(status)) {
        int rc = WEXITSTATUS(status);
        switch (rc) {
            case 0:
                GTEST_SUCCEED();
                break;
            case 1:
                FAIL() << "d=0 RVV distance != 0 (qtype=" << int(p.qtype) << " "
                       << metric_name(p.metric) << ")";
                break;
            case 2:
                FAIL() << "d=0 RVV != scalar parity (qtype=" << int(p.qtype)
                       << " " << metric_name(p.metric) << ")";
                break;
            case 3:
                FAIL() << "distance computer is null (qtype=" << int(p.qtype)
                       << " " << metric_name(p.metric) << ")";
                break;
            case 4:
                FAIL() << "d=0 non-finite result (qtype=" << int(p.qtype) << " "
                       << metric_name(p.metric) << ")";
                break;
            default:
                FAIL() << "child exit " << rc;
        }
    } else {
        FAIL() << "child abnormal exit";
    }
}
#endif // !_WIN32

#ifdef _WIN32
TEST_P(SQRVVZeroDim, ReturnsZeroNoHang) {
    GTEST_SKIP() << "fork-based hang isolation requires POSIX";
}
#endif

INSTANTIATE_TEST_SUITE_P(
        D6,
        SQRVVZeroDim,
        ::testing::Values(
                ZeroDimParams{SQ::QT_4bit, METRIC_L2, "4bit_L2"},
                ZeroDimParams{SQ::QT_4bit, METRIC_INNER_PRODUCT, "4bit_IP"},
                ZeroDimParams{
                        SQ::QT_4bit_uniform,
                        METRIC_L2,
                        "4bit_uniform_L2"},
                ZeroDimParams{
                        SQ::QT_4bit_uniform,
                        METRIC_INNER_PRODUCT,
                        "4bit_uniform_IP"},
                ZeroDimParams{SQ::QT_6bit, METRIC_L2, "6bit_L2"},
                ZeroDimParams{SQ::QT_6bit, METRIC_INNER_PRODUCT, "6bit_IP"},
                ZeroDimParams{SQ::QT_8bit, METRIC_L2, "8bit_L2"},
                ZeroDimParams{SQ::QT_8bit, METRIC_INNER_PRODUCT, "8bit_IP"},
                ZeroDimParams{
                        SQ::QT_8bit_uniform,
                        METRIC_L2,
                        "8bit_uniform_L2"},
                ZeroDimParams{
                        SQ::QT_8bit_uniform,
                        METRIC_INNER_PRODUCT,
                        "8bit_uniform_IP"},
                ZeroDimParams{SQ::QT_8bit_direct, METRIC_L2, "8bit_direct_L2"},
                ZeroDimParams{
                        SQ::QT_8bit_direct,
                        METRIC_INNER_PRODUCT,
                        "8bit_direct_IP"},
                ZeroDimParams{
                        SQ::QT_8bit_direct_signed,
                        METRIC_L2,
                        "8bit_direct_signed_L2"},
                ZeroDimParams{
                        SQ::QT_8bit_direct_signed,
                        METRIC_INNER_PRODUCT,
                        "8bit_direct_signed_IP"},
                ZeroDimParams{SQ::QT_bf16, METRIC_L2, "bf16_L2"},
                ZeroDimParams{SQ::QT_bf16, METRIC_INNER_PRODUCT, "bf16_IP"},
                ZeroDimParams{SQ::QT_fp16, METRIC_L2, "fp16_L2"},
                ZeroDimParams{SQ::QT_fp16, METRIC_INNER_PRODUCT, "fp16_IP"}),
        [](const ::testing::TestParamInfo<ZeroDimParams>& info) {
            return std::string(info.param.name);
        });

// ===========================================================================
// D7 — Special queries.
//   FloatDomain + UniformApprox: exact float-domain kernels — clamp
//       boundaries, far-out-of-range, and huge-but-finite magnitudes must
//       stay in parity with the scalar reference (1e-4 rel).
//   Direct: integer-grid in-contract queries — exact.
// ===========================================================================

struct SpecialQParams {
    SQ::QuantizerType qtype;
    faiss::MetricType metric;
    size_t d;
    float tol;
    const char* name;
};

class SQRVVSpecialQueries : public ::testing::TestWithParam<SpecialQParams> {};

TEST_P(SQRVVSpecialQueries, EdgeQueriesParity) {
    const auto& p = GetParam();
    if (!faiss::SIMDConfig::is_simd_level_available(
                faiss::SIMDLevel::RISCV_RVV)) {
        GTEST_SKIP() << "RISCV_RVV not available";
    }
    const size_t n_train = 64;
    const size_t n_db = 4;

    const QClass qclass = class_of(p.qtype);
    const bool direct = (qclass == QClass::Direct);
    const int dlo = (p.qtype == SQ::QT_8bit_direct_signed) ? -128 : 0;
    const int dhi = (p.qtype == SQ::QT_8bit_direct_signed) ? 127 : 255;

    std::vector<float> xt = direct
            ? random_int_floats(n_train, p.d, dlo, dhi, 42)
            : random_int_floats(n_train, p.d, -40, 200, 42);
    std::vector<float> xb =
            direct ? xt : random_int_floats(n_db, p.d, -40, 200, 43);

    SQ sq(p.d, p.qtype);
    sq.train(n_train, xt.data());
    std::vector<uint8_t> codes(sq.code_size * n_db, 0);
    sq.compute_codes(xb.data(), codes.data(), n_db);

    DcPair dc = make_dc_pair(
            p.qtype, p.metric, p.d, sq.trained, codes.data(), sq.code_size);
    ASSERT_TRUE(dc.scalar_dc && dc.rvv_dc);

    float vmin = 0, vdiff = 0;
    bool has_range = false;
    trained_range(p.qtype, p.d, sq.trained, vmin, vdiff, has_range);

    // Query set per class
    std::vector<std::vector<float>> queries;
    queries.emplace_back(p.d, 0.0f); // all-zero
    if (direct) {
        queries.emplace_back(p.d, float(dlo));
        queries.emplace_back(p.d, float(dhi));
        queries.emplace_back(p.d, float(dlo) + 1.0f);
        queries.emplace_back(p.d, float(dhi) - 1.0f);
        {
            std::vector<float> q(p.d);
            for (size_t i = 0; i < p.d; i++)
                q[i] = (i % 2 == 0) ? float(dlo) : float(dhi);
            queries.push_back(std::move(q));
        }
    } else if (qclass == QClass::UniformApprox) {
        // Uniform codecs are now float-domain (exact affine), so any query
        // stays in parity; in-range systematic values are a conservative
        // subset that still exercises the affine reconstruction path.
        if (has_range) {
            queries.emplace_back(p.d, vmin);
            queries.emplace_back(p.d, vmin + vdiff);
            queries.emplace_back(p.d, vmin + 0.5f * vdiff);
            {
                std::vector<float> q(p.d);
                for (size_t i = 0; i < p.d; i++)
                    q[i] = (i % 2 == 0) ? vmin : vmin + vdiff;
                queries.push_back(std::move(q));
            }
        }
    } else {
        // FloatDomain: exact kernels — full-range + far-out + huge finite
        queries.emplace_back(p.d, 1.0f);
        queries.emplace_back(p.d, -1.0f);
        queries.emplace_back(p.d, 1e4f);
        queries.emplace_back(p.d, -1e4f);
        queries.emplace_back(p.d, 1e18f); // huge but finite in f32
        if (has_range) {
            queries.emplace_back(p.d, vmin);
            queries.emplace_back(p.d, vmin + vdiff);
            queries.emplace_back(p.d, vmin - 1000.0f);
            queries.emplace_back(p.d, vmin + vdiff + 1000.0f);
        }
    }

    // Relative tolerance: 1e-4 for all float-domain / exact kernels
    // (uniform L2/IP are now float-domain, same precision class as the
    // non-uniform codecs).
    const float rel_tol = 1e-4f;

    for (size_t qi = 0; qi < queries.size(); qi++) {
        SCOPED_TRACE(
                ::testing::Message()
                << p.name << " query=" << qi << " d=" << p.d);
        dc.scalar_dc->set_query(queries[qi].data());
        dc.rvv_dc->set_query(queries[qi].data());
        for (size_t i = 0; i < n_db; i++) {
            const uint8_t* code = codes.data() + i * sq.code_size;
            float ref = dc.scalar_dc->query_to_code(code);
            float tst = dc.rvv_dc->query_to_code(code);
            ASSERT_TRUE(std::isfinite(ref)) << "scalar non-finite";
            ASSERT_TRUE(std::isfinite(tst))
                    << "qtype=" << int(p.qtype) << " non-finite";
            EXPECT_NEAR(ref, tst, p.tol + std::abs(ref) * rel_tol);
        }
    }
}

/*
 * d=17 crosses VLMAX for the e8m1 kernels AND exercises tails, keeping
 * runtime modest across all 9 qtypes x 2 metrics.
 */
INSTANTIATE_TEST_SUITE_P(
        D7,
        SQRVVSpecialQueries,
        ::testing::Values(
                SpecialQParams{SQ::QT_4bit, METRIC_L2, 17, 1e-3f, "4bit"},
                SpecialQParams{
                        SQ::QT_4bit,
                        METRIC_INNER_PRODUCT,
                        17,
                        1e-3f,
                        "4bit_IP"},
                SpecialQParams{
                        SQ::QT_4bit_uniform,
                        METRIC_L2,
                        17,
                        1e-3f,
                        "4bit_uniform"},
                SpecialQParams{
                        SQ::QT_4bit_uniform,
                        METRIC_INNER_PRODUCT,
                        17,
                        1e-3f,
                        "4bit_uniform_IP"},
                SpecialQParams{SQ::QT_6bit, METRIC_L2, 17, 1e-3f, "6bit"},
                SpecialQParams{
                        SQ::QT_6bit,
                        METRIC_INNER_PRODUCT,
                        17,
                        1e-3f,
                        "6bit_IP"},
                SpecialQParams{SQ::QT_8bit, METRIC_L2, 17, 1e-3f, "8bit"},
                SpecialQParams{
                        SQ::QT_8bit,
                        METRIC_INNER_PRODUCT,
                        17,
                        1e-3f,
                        "8bit_IP"},
                SpecialQParams{
                        SQ::QT_8bit_uniform,
                        METRIC_L2,
                        17,
                        1e-3f,
                        "8bit_uniform"},
                SpecialQParams{
                        SQ::QT_8bit_uniform,
                        METRIC_INNER_PRODUCT,
                        17,
                        1e-3f,
                        "8bit_uniform_IP"},
                SpecialQParams{
                        SQ::QT_8bit_direct,
                        METRIC_L2,
                        17,
                        1e-3f,
                        "8bit_direct"},
                SpecialQParams{
                        SQ::QT_8bit_direct,
                        METRIC_INNER_PRODUCT,
                        17,
                        1e-3f,
                        "8bit_direct_IP"},
                SpecialQParams{
                        SQ::QT_8bit_direct_signed,
                        METRIC_L2,
                        17,
                        1e-3f,
                        "8bit_direct_signed"},
                SpecialQParams{
                        SQ::QT_8bit_direct_signed,
                        METRIC_INNER_PRODUCT,
                        17,
                        1e-3f,
                        "8bit_direct_signed_IP"},
                SpecialQParams{SQ::QT_bf16, METRIC_L2, 17, 1e-3f, "bf16"},
                SpecialQParams{
                        SQ::QT_bf16,
                        METRIC_INNER_PRODUCT,
                        17,
                        1e-3f,
                        "bf16_IP"},
                SpecialQParams{SQ::QT_fp16, METRIC_L2, 17, 1e-3f, "fp16"},
                SpecialQParams{
                        SQ::QT_fp16,
                        METRIC_INNER_PRODUCT,
                        17,
                        1e-3f,
                        "fp16_IP"}));

// ===========================================================================
// D8 — Very small / odd dimensions. The review asks for "odd and very small
//      dimensions" in addition to VLMAX boundaries. d < 8 exercises the
//      packing tail: 4-bit odd-d padding nibble, 6-bit ng==0 (d<4) pure
//      scalar tail vs ng>=1 with a d%4 scalar tail, and the single-lane /
//      tail-only vector path for every codec.
// ===========================================================================

struct SmallDimParams {
    SQ::QuantizerType qtype;
    faiss::MetricType metric;
    size_t d;
    const char* name;
};

class SQRVVSmallDim : public ::testing::TestWithParam<SmallDimParams> {};

TEST_P(SQRVVSmallDim, SmallDimensionParity) {
    const auto& p = GetParam();
    if (!faiss::SIMDConfig::is_simd_level_available(
                faiss::SIMDLevel::RISCV_RVV)) {
        GTEST_SKIP() << "RISCV_RVV not available";
    }
    const size_t n_train = 64;
    const size_t n_db = 4;

    const bool direct = is_direct(p.qtype);
    const int dlo = (p.qtype == SQ::QT_8bit_direct_signed) ? -128 : 0;
    const int dhi = (p.qtype == SQ::QT_8bit_direct_signed) ? 127 : 255;

    std::vector<float> xb = direct
            ? random_int_floats(n_train, p.d, dlo, dhi, 42)
            : make_random_vectors(n_train, p.d, 42);
    std::vector<float> xq = direct ? random_int_floats(1, p.d, dlo, dhi, 99)
                                   : make_random_vectors(1, p.d, 99);
    if (p.metric == METRIC_INNER_PRODUCT && !direct) {
        faiss::fvec_renorm_L2(p.d, n_train, xb.data());
        faiss::fvec_renorm_L2(p.d, 1, xq.data());
    }

    SQ sq(p.d, p.qtype);
    sq.train(n_train, xb.data());
    std::vector<uint8_t> codes(sq.code_size * n_db, 0);
    sq.compute_codes(xb.data(), codes.data(), n_db);

    DcPair dc = make_dc_pair(
            p.qtype, p.metric, p.d, sq.trained, codes.data(), sq.code_size);
    ASSERT_TRUE(dc.scalar_dc && dc.rvv_dc);
    dc.scalar_dc->set_query(xq.data());
    dc.rvv_dc->set_query(xq.data());

    SCOPED_TRACE(
            ::testing::Message()
            << p.name << " d=" << p.d << " " << metric_name(p.metric));

    // All kernels are float-domain after the review fix -> reassociation
    // precision only; direct codecs are additionally integer-exact here.
    const float tol = 1e-3f;

    for (size_t i = 0; i < n_db; i++) {
        const uint8_t* code = codes.data() + i * sq.code_size;
        float ref = dc.scalar_dc->query_to_code(code);
        float tst = dc.rvv_dc->query_to_code(code);
        ASSERT_TRUE(std::isfinite(ref) && std::isfinite(tst)) << p.name;
        EXPECT_NEAR(ref, tst, tol + std::abs(ref) * 1e-4f)
                << p.name << " query_to_code code=" << i;
    }

    {
        std::vector<uint8_t> zero_code(sq.code_size, 0);
        float ref = dc.scalar_dc->query_to_code(zero_code.data());
        float tst = dc.rvv_dc->query_to_code(zero_code.data());
        EXPECT_NEAR(ref, tst, tol + std::abs(ref) * 1e-4f)
                << p.name << " zero code";
    }

    for (size_t i = 0; i + 1 < n_db; i += 2) {
        float ref = dc.scalar_dc->symmetric_dis(i, i + 1);
        float tst = dc.rvv_dc->symmetric_dis(i, i + 1);
        EXPECT_NEAR(ref, tst, tol + std::abs(ref) * 1e-4f)
                << p.name << " symmetric_dis(" << i << "," << (i + 1) << ")";
    }

    {
        const uint8_t *c0 = codes.data(), *c1 = codes.data() + sq.code_size,
                      *c2 = codes.data() + 2 * sq.code_size,
                      *c3 = codes.data() + 3 * sq.code_size;
        float r[4], t[4];
        dc.scalar_dc->query_to_codes_batch_4(
                c0, c1, c2, c3, r[0], r[1], r[2], r[3]);
        dc.rvv_dc->query_to_codes_batch_4(
                c0, c1, c2, c3, t[0], t[1], t[2], t[3]);
        for (int j = 0; j < 4; j++)
            EXPECT_NEAR(r[j], t[j], tol + std::abs(r[j]) * 1e-4f)
                    << p.name << " batch_4[" << j << "]";
    }
}

// 9 qtypes x {L2, IP} x {1, 2, 3, 5, 7}
#define SMALL_DIM_CASES(QT, METRIC, TAG)              \
    SmallDimParams{QT, METRIC, 1, TAG "_d1"},         \
            SmallDimParams{QT, METRIC, 2, TAG "_d2"}, \
            SmallDimParams{QT, METRIC, 3, TAG "_d3"}, \
            SmallDimParams{QT, METRIC, 5, TAG "_d5"}, \
            SmallDimParams{QT, METRIC, 7, TAG "_d7"}

INSTANTIATE_TEST_SUITE_P(
        D8,
        SQRVVSmallDim,
        ::testing::Values(
                SMALL_DIM_CASES(SQ::QT_4bit, METRIC_L2, "4bit_L2"),
                SMALL_DIM_CASES(SQ::QT_4bit, METRIC_INNER_PRODUCT, "4bit_IP"),
                SMALL_DIM_CASES(
                        SQ::QT_4bit_uniform,
                        METRIC_L2,
                        "4bit_uniform_L2"),
                SMALL_DIM_CASES(
                        SQ::QT_4bit_uniform,
                        METRIC_INNER_PRODUCT,
                        "4bit_uniform_IP"),
                SMALL_DIM_CASES(SQ::QT_6bit, METRIC_L2, "6bit_L2"),
                SMALL_DIM_CASES(SQ::QT_6bit, METRIC_INNER_PRODUCT, "6bit_IP"),
                SMALL_DIM_CASES(SQ::QT_8bit, METRIC_L2, "8bit_L2"),
                SMALL_DIM_CASES(SQ::QT_8bit, METRIC_INNER_PRODUCT, "8bit_IP"),
                SMALL_DIM_CASES(
                        SQ::QT_8bit_uniform,
                        METRIC_L2,
                        "8bit_uniform_L2"),
                SMALL_DIM_CASES(
                        SQ::QT_8bit_uniform,
                        METRIC_INNER_PRODUCT,
                        "8bit_uniform_IP"),
                SMALL_DIM_CASES(
                        SQ::QT_8bit_direct,
                        METRIC_L2,
                        "8bit_direct_L2"),
                SMALL_DIM_CASES(
                        SQ::QT_8bit_direct,
                        METRIC_INNER_PRODUCT,
                        "8bit_direct_IP"),
                SMALL_DIM_CASES(
                        SQ::QT_8bit_direct_signed,
                        METRIC_L2,
                        "8bit_direct_signed_L2"),
                SMALL_DIM_CASES(
                        SQ::QT_8bit_direct_signed,
                        METRIC_INNER_PRODUCT,
                        "8bit_direct_signed_IP"),
                SMALL_DIM_CASES(SQ::QT_bf16, METRIC_L2, "bf16_L2"),
                SMALL_DIM_CASES(SQ::QT_bf16, METRIC_INNER_PRODUCT, "bf16_IP"),
                SMALL_DIM_CASES(SQ::QT_fp16, METRIC_L2, "fp16_L2"),
                SMALL_DIM_CASES(SQ::QT_fp16, METRIC_INNER_PRODUCT, "fp16_IP")),
        [](const ::testing::TestParamInfo<SmallDimParams>& info) {
            return std::string(info.param.name);
        });

// ===========================================================================
// D9 — q == recon(code). For every qtype under L2, decode a code back to its
//      reconstructed vector and use that as the query; the L2 distance to
//      that same code must be ~0 (the code reconstructs exactly to the
//      query). This pins the general semantic baseline the review asks for
//      ("queries equal to reconstructed code values"), beyond the uniform-L2
//      counterexample already covered by D1.
// ===========================================================================

struct ReconParams {
    SQ::QuantizerType qtype;
    size_t d;
    const char* name;
};

class SQRVVReconQuery : public ::testing::TestWithParam<ReconParams> {};

TEST_P(SQRVVReconQuery, QueryEqualsReconstructedValue) {
    const auto& p = GetParam();
    if (!faiss::SIMDConfig::is_simd_level_available(
                faiss::SIMDLevel::RISCV_RVV)) {
        GTEST_SKIP() << "RISCV_RVV not available";
    }
    const size_t n_train = 64;

    const bool direct = is_direct(p.qtype);
    const int dlo = (p.qtype == SQ::QT_8bit_direct_signed) ? -128 : 0;
    const int dhi = (p.qtype == SQ::QT_8bit_direct_signed) ? 127 : 255;

    std::vector<float> xb = direct
            ? random_int_floats(n_train, p.d, dlo, dhi, 42)
            : make_random_vectors(n_train, p.d, 42);
    SQ sq(p.d, p.qtype);
    sq.train(n_train, xb.data());

    // Encode one in-contract vector, then decode it back to recon(code).
    std::vector<float> xsrc = direct ? random_int_floats(1, p.d, dlo, dhi, 7)
                                     : make_random_vectors(1, p.d, 7);
    std::vector<uint8_t> code(sq.code_size);
    sq.compute_codes(xsrc.data(), code.data(), 1);
    std::vector<float> q(p.d);
    sq.decode(code.data(), q.data(), 1); // q == recon(code)

    DcPair dc = make_dc_pair(
            p.qtype, METRIC_L2, p.d, sq.trained, code.data(), sq.code_size);
    ASSERT_TRUE(dc.scalar_dc && dc.rvv_dc);
    dc.scalar_dc->set_query(q.data());
    dc.rvv_dc->set_query(q.data());

    float ref = dc.scalar_dc->query_to_code(code.data());
    float tst = dc.rvv_dc->query_to_code(code.data());

    ASSERT_TRUE(std::isfinite(ref) && std::isfinite(tst)) << p.name;
    EXPECT_NEAR(ref, 0.0f, 1e-3f)
            << p.name << " scalar reference nonzero for own recon";
    EXPECT_NEAR(tst, 0.0f, 1e-3f)
            << p.name << " RVV distance to own recon = " << tst;
}

INSTANTIATE_TEST_SUITE_P(
        D9,
        SQRVVReconQuery,
        ::testing::Values(
                ReconParams{SQ::QT_4bit, 8, "4bit_d8"},
                ReconParams{SQ::QT_4bit, 17, "4bit_d17"},
                ReconParams{SQ::QT_4bit_uniform, 8, "4bit_uniform_d8"},
                ReconParams{SQ::QT_4bit_uniform, 17, "4bit_uniform_d17"},
                ReconParams{SQ::QT_6bit, 8, "6bit_d8"},
                ReconParams{SQ::QT_6bit, 17, "6bit_d17"},
                ReconParams{SQ::QT_8bit, 8, "8bit_d8"},
                ReconParams{SQ::QT_8bit, 17, "8bit_d17"},
                ReconParams{SQ::QT_8bit_uniform, 8, "8bit_uniform_d8"},
                ReconParams{SQ::QT_8bit_uniform, 17, "8bit_uniform_d17"},
                ReconParams{SQ::QT_8bit_direct, 8, "8bit_direct_d8"},
                ReconParams{SQ::QT_8bit_direct, 17, "8bit_direct_d17"},
                ReconParams{
                        SQ::QT_8bit_direct_signed,
                        8,
                        "8bit_direct_signed_d8"},
                ReconParams{
                        SQ::QT_8bit_direct_signed,
                        17,
                        "8bit_direct_signed_d17"},
                ReconParams{SQ::QT_bf16, 8, "bf16_d8"},
                ReconParams{SQ::QT_bf16, 17, "bf16_d17"},
                ReconParams{SQ::QT_fp16, 8, "fp16_d8"},
                ReconParams{SQ::QT_fp16, 17, "fp16_d17"}),
        [](const ::testing::TestParamInfo<ReconParams>& info) {
            return std::string(info.param.name);
        });

} // namespace
