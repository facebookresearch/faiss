/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <functional>
#include <future>
#include <limits>
#include <memory>
#include <random>
#include <string>
#include <thread>
#include <utility>
#include <vector>

#include <faiss/IndexScalarQuantizer.h>
#include <faiss/IndexSQFP16Gemm.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/io.h>
#include <faiss/impl/scalar_quantizer/sq-fp16-gemm.h>
#include <faiss/index_io.h>
#include <faiss/utils/fp16.h>

#ifdef _OPENMP
#include <omp.h>
#endif

using faiss::idx_t;
namespace fp16_gemm = faiss::fp16_gemm;

// These tests cover the two pieces contributed alongside
// IndexSQFP16Gemm: the standalone fp16_gemm kernels (pack/gemm/rounding,
// namespace faiss::fp16_gemm) and the Index subclass that wraps them. Both
// SCALAR (always present) and, when the build and CPU support them, FMA and
// AMX-FP16 are exercised; hardware/compiler-gated paths use GTEST_SKIP
// rather than failing when unavailable.
namespace {

std::vector<fp16_gemm::GemmKernel> available_kernels() {
    std::vector<fp16_gemm::GemmKernel> kernels{fp16_gemm::GemmKernel::SCALAR};
    if (fp16_gemm::fma_available()) {
        kernels.push_back(fp16_gemm::GemmKernel::FMA);
    }
    if (fp16_gemm::amx_fp16_available()) {
        kernels.push_back(fp16_gemm::GemmKernel::AMX);
    }
    return kernels;
}

std::vector<float> sample(size_t n, size_t d, unsigned int seed) {
    std::mt19937 generator(seed);
    std::uniform_real_distribution<float> values(-1.f, 1.f);
    std::vector<float> result(n * d);
    for (float& value : result) {
        value = values(generator) / std::sqrt(static_cast<float>(d));
    }
    return result;
}

// Runs fn and reports what happened as text, so a serial and a parallel
// call that are supposed to fail identically can be compared with EXPECT_EQ
// instead of just "both threw something".
std::string round_outcome(const std::function<void()>& fn) {
    try {
        fn();
        return "no exception";
    } catch (const std::exception& error) {
        return error.what();
    }
}

// ---------------------------------------------------------------------
// faiss::fp16_gemm kernel-level tests
// ---------------------------------------------------------------------

// Pins down the packing order documented in sq-fp16-gemm.h: centroids are
// packed 16 at a time (a "group"), every dimension of the group stored
// contiguously, 32 dimensions per 1KiB segment.
TEST(SQFP16GemmKernels, PackLayoutPinsDownOrdering) {
    constexpr size_t d = 70, n = 64, nreal = 50, stride = 75;
    std::vector<uint16_t> rows(nreal * stride);
    for (size_t i = 0; i < rows.size(); ++i) {
        rows[i] = static_cast<uint16_t>(i * 7 + 1);
    }
    std::vector<uint16_t> packed(fp16_gemm::packed_b_size(n, d), 0xffff);
    ASSERT_EQ(packed.size(), (d / 32) * n * 32) << "packed buffer size formula changed";
    fp16_gemm::pack_centroids(rows.data(), nreal, stride, n, d, packed.data());
    const size_t kblocks = d / 32;
    for (size_t jb = 0; jb < n / 16; ++jb) {
        for (size_t kb = 0; kb < kblocks; ++kb) {
            for (size_t kk = 0; kk < 16; ++kk) {
                for (size_t nn = 0; nn < 16; ++nn) {
                    for (size_t p = 0; p < 2; ++p) {
                        const size_t col = jb * 16 + nn;
                        const uint16_t expected = col < nreal
                                ? rows[col * stride + kb * 32 + kk * 2 + p]
                                : uint16_t{0};
                        const size_t at = (jb * kblocks + kb) * 512 + kk * 32 + nn * 2 + p;
                        ASSERT_EQ(packed[at], expected)
                                << "packing order does not match sq-fp16-gemm.h's documented layout";
                    }
                }
            }
        }
    }
}

// Compares a kernel's full m x n_padded output against a double-precision
// reference, checks zero-padding of unused output columns and that the
// caller's row stride is never written past, then checks that computing
// fewer rows gives a bit-identical prefix of the full result (or throws if
// the row count is not a multiple of what the kernel requires).
void check_gemm_shape(fp16_gemm::GemmKernel kernel, size_t d, bool wide) {
    constexpr size_t m = 64, n = 96, nreal = 73;
    const size_t astride = d + 7, bstride = d + 5, ostride = n + 9;
    std::mt19937 generator(static_cast<unsigned int>(d + (wide ? 4000 : 0)));
    std::uniform_real_distribution<float> values(-1.f, 1.f);
    std::vector<uint16_t> a(m * astride, 0), b(nreal * bstride, 0);
    for (size_t i = 0; i < m; ++i) {
        for (size_t kk = 0; kk < d; ++kk) {
            float value = values(generator) / std::sqrt(static_cast<float>(d));
            if (wide) {
                value = std::ldexp(values(generator), static_cast<int>((i + kk) % 29) - 14);
            }
            a[i * astride + kk] = faiss::encode_fp16(value);
        }
    }
    for (size_t j = 0; j < nreal; ++j) {
        for (size_t kk = 0; kk < d; ++kk) {
            float value = values(generator) / std::sqrt(static_cast<float>(d));
            if (wide) {
                value = std::ldexp(values(generator), static_cast<int>((j * 3 + kk) % 29) - 14);
            }
            b[j * bstride + kk] = faiss::encode_fp16(value);
        }
    }
    std::vector<uint16_t> packed(fp16_gemm::packed_b_size(n, d));
    fp16_gemm::pack_centroids(b.data(), nreal, bstride, n, d, packed.data());
    std::vector<float> out(m * ostride, 123456.f);
    fp16_gemm::get_gemm_fn(kernel)(
            a.data(), m, astride, packed.data(), b.data(), bstride, n, nreal, d,
            out.data(), ostride);

    constexpr double unit_roundoff = 0x1p-24;
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < nreal; ++j) {
            double exact = 0, sum_absolute = 0;
            for (size_t kk = 0; kk < d; ++kk) {
                const double product = static_cast<double>(faiss::decode_fp16(a[i * astride + kk])) *
                        faiss::decode_fp16(b[j * bstride + kk]);
                exact += product;
                sum_absolute += std::abs(product);
            }
            const double error = std::abs(static_cast<double>(out[i * ostride + j]) - exact);
            const double bound = wide
                    ? (2 * d * unit_roundoff / (1 - 2 * d * unit_roundoff)) * sum_absolute
                    : 1e-5 + 1e-4 * std::abs(exact);
            ASSERT_LE(error, bound) << fp16_gemm::kernel_name(kernel) << " exceeds its error bound: d=" << d
                                     << " row=" << i << " col=" << j;
        }
        for (size_t j = nreal; j < n; ++j) {
            ASSERT_EQ(out[i * ostride + j], 0.f) << "padding column past n_real is not zero";
        }
        for (size_t j = n; j < ostride; ++j) {
            ASSERT_EQ(out[i * ostride + j], 123456.f) << "kernel wrote past the caller's row stride";
        }
    }

    // Computing fewer rows must give a bit-identical prefix of the m-row
    // result (not a separately-rounded approximation), and a row count
    // that isn't a multiple of what the kernel requires must be rejected.
    const size_t multiple = fp16_gemm::gemm_row_multiple(kernel);
    const size_t row_counts[] = {0, 1, 2, 3, 7, 8, 9, 15, 16, 17, 31, 32, 33, 48, 63};
    for (size_t rows : row_counts) {
        std::vector<float> part(m * ostride, 123456.f);
        const auto call = [&] {
            fp16_gemm::get_gemm_fn(kernel)(
                    a.data(), rows, astride, packed.data(), b.data(), bstride, n,
                    nreal, d, part.data(), ostride);
        };
        if (rows % multiple) {
            EXPECT_ANY_THROW(call()) << "kernel accepted a row count that is not a multiple of " << multiple;
            continue;
        }
        call();
        EXPECT_EQ(0, std::memcmp(part.data(), out.data(), rows * ostride * sizeof(float)))
                << fp16_gemm::kernel_name(kernel)
                << " partial row count is not bit-identical to the full computation: rows=" << rows << " d=" << d;
        for (size_t idx = rows * ostride; idx < part.size(); ++idx) {
            ASSERT_EQ(part[idx], 123456.f) << "kernel wrote past row " << rows;
        }
    }
}

TEST(SQFP16GemmKernels, GemmMatchesReferenceAndRespectsRowMultiple) {
    for (const auto kernel : available_kernels()) {
        SCOPED_TRACE(fp16_gemm::kernel_name(kernel));
        for (size_t d : {size_t{1}, size_t{31}, size_t{32}, size_t{33}, size_t{767}, size_t{768}, size_t{769}}) {
            check_gemm_shape(kernel, d, /*wide=*/false);
            check_gemm_shape(kernel, d, /*wide=*/true);
        }
    }
}

// A subnormal fp16 input contributes a tiny but nonzero product; some AMX
// generations flush subnormal inputs to zero in hardware and the kernel
// must patch the contribution back in rather than silently dropping it
// (and must not double count it either).
void check_subnormal(fp16_gemm::GemmKernel kernel, size_t d, size_t m) {
    std::vector<uint16_t> a(32 * d, 0), b(4 * d, 0);
    a[0] = 1; // smallest positive subnormal
    a[d] = 0x8001; // smallest negative subnormal
    b[0] = 0x3c00; // 1.0
    b[d] = 1;
    b[2 * d] = 0x8001;
    b[3 * d] = 0x7bff; // largest finite
    std::vector<uint16_t> packed(fp16_gemm::packed_b_size(32, d));
    fp16_gemm::pack_centroids(b.data(), 4, d, 32, d, packed.data());
    std::vector<float> out(32 * 32);
    fp16_gemm::get_gemm_fn(kernel)(a.data(), m, d, packed.data(), b.data(), d, 32, 4, d, out.data(), 32);
    for (size_t i = 0; i < std::min<size_t>(2, m); ++i) {
        for (size_t j = 0; j < 4; ++j) {
            const float exact = faiss::decode_fp16(a[i * d]) * faiss::decode_fp16(b[j * d]);
            EXPECT_EQ(out[i * 32 + j], exact)
                    << fp16_gemm::kernel_name(kernel)
                    << " lost or double-counted a subnormal contribution: d=" << d << " m=" << m << " i=" << i
                    << " j=" << j;
        }
    }
}

TEST(SQFP16GemmKernels, SubnormalContributionsPreserved) {
    for (const auto kernel : available_kernels()) {
        SCOPED_TRACE(fp16_gemm::kernel_name(kernel));
        for (size_t d : {size_t{1}, size_t{31}, size_t{32}, size_t{33}, size_t{767}, size_t{768}, size_t{769}}) {
            check_subnormal(kernel, d, 32);
            check_subnormal(kernel, d, fp16_gemm::gemm_row_multiple(kernel));
        }
    }
}

// AMX tile configuration/permission is a per-thread kernel resource; a
// freshly spawned thread must be able to request it on its own.
TEST(SQFP16GemmKernels, ThreadLocalAmxTilePermission) {
    if (!fp16_gemm::amx_fp16_available()) {
        GTEST_SKIP() << "AMX-FP16 not available on this build/CPU";
    }
    std::exception_ptr error;
    std::thread worker([&] {
        try {
            EXPECT_TRUE(fp16_gemm::amx_fp16_available()) << "new thread could not acquire AMX tile permission";
            check_subnormal(fp16_gemm::GemmKernel::AMX, 33, 32);
        } catch (...) {
            error = std::current_exception();
        }
    });
    worker.join();
    if (error) {
        std::rethrow_exception(error);
    }
}

TEST(SQFP16GemmKernels, ResolveKernelRejectsExplicitlyUnavailable) {
    EXPECT_NO_THROW(fp16_gemm::resolve_kernel(fp16_gemm::GemmKernel::SCALAR));
    EXPECT_NO_THROW(fp16_gemm::resolve_kernel(fp16_gemm::GemmKernel::AUTO));
    if (fp16_gemm::fma_available()) {
        EXPECT_EQ(fp16_gemm::resolve_kernel(fp16_gemm::GemmKernel::FMA), fp16_gemm::GemmKernel::FMA);
    } else {
        EXPECT_ANY_THROW(fp16_gemm::resolve_kernel(fp16_gemm::GemmKernel::FMA))
                << "explicit FMA request should fail when unavailable instead of silently falling back";
    }
    if (fp16_gemm::amx_fp16_available()) {
        EXPECT_EQ(fp16_gemm::resolve_kernel(fp16_gemm::GemmKernel::AMX), fp16_gemm::GemmKernel::AMX);
    } else {
        EXPECT_ANY_THROW(fp16_gemm::resolve_kernel(fp16_gemm::GemmKernel::AMX))
                << "explicit AMX request should fail when unavailable instead of silently falling back";
    }
    const auto auto_pick = fp16_gemm::resolve_kernel(fp16_gemm::GemmKernel::AUTO);
    if (fp16_gemm::amx_fp16_available()) {
        EXPECT_EQ(auto_pick, fp16_gemm::GemmKernel::AMX);
    } else if (fp16_gemm::fma_available()) {
        EXPECT_EQ(auto_pick, fp16_gemm::GemmKernel::FMA);
    } else {
        EXPECT_EQ(auto_pick, fp16_gemm::GemmKernel::SCALAR);
    }
}

TEST(SQFP16GemmKernels, PackedBSizeRejectsNonMultipleOf32) {
    EXPECT_ANY_THROW(fp16_gemm::packed_b_size(31, 32)) << "non-multiple-of-32 padded centroid count was accepted";
}

TEST(SQFP16GemmKernels, RoundValuesRejectsNonFiniteAndOverflow) {
    float output = 0;
    for (float value : {std::numeric_limits<float>::quiet_NaN(),
                         std::numeric_limits<float>::infinity(),
                         -std::numeric_limits<float>::infinity(),
                         65520.f,
                         -65520.f}) {
        EXPECT_ANY_THROW(fp16_gemm::round_values(&value, &output, 1))
                << "round_values accepted a non-finite or overflowing input: " << value;
    }
}

TEST(SQFP16GemmKernels, RoundValuesParallelMatchesSerial) {
#ifdef _OPENMP
    const int saved_threads = omp_get_max_threads();
    omp_set_num_threads(4);
#endif
    const float nan_value = std::numeric_limits<float>::quiet_NaN();
    const float inf_value = std::numeric_limits<float>::infinity();
    const float too_big = 65520.f; // rounds up to infinity under round-to-even
    for (size_t block : {size_t{256}, size_t{1024}, fp16_gemm::kRoundBlock}) {
        // Nine full blocks plus a short tail: crosses block and thread
        // boundaries as well as the final partial block.
        const size_t n = block * 9 + 77;
        auto x = sample(n, 1, static_cast<unsigned int>(6007 + block));
        for (size_t i = 0; i < n; i += 97) {
            x[i] = faiss::decode_fp16(static_cast<uint16_t>(i % 0x7bff)) * (i % 2 ? -1.f : 1.f);
        }
        x[1] = -0.f;
        x[2] = faiss::decode_fp16(1);
        x[3] = 65504.f;
        x[4] = 65519.f;
        std::vector<float> serial(n), parallel(n, -1.f), inplace(x);
        fp16_gemm::round_values(x.data(), serial.data(), n);
        fp16_gemm::round_values_parallel(x.data(), parallel.data(), n, block);
        fp16_gemm::round_values_parallel(inplace.data(), inplace.data(), n, block);
        EXPECT_EQ(0, std::memcmp(serial.data(), parallel.data(), n * sizeof(float)))
                << "parallel conversion differs from serial for block=" << block;
        EXPECT_EQ(0, std::memcmp(serial.data(), inplace.data(), n * sizeof(float)))
                << "in-place parallel conversion differs from serial for block=" << block;

        // Each case is a set of (position, bad value) pairs; serial and
        // parallel must report the exact same error, including which
        // element they blame (the lowest-numbered bad position).
        const std::vector<std::vector<std::pair<size_t, float>>> cases = {
                {{n - 1, nan_value}},
                {{5, nan_value}, {block * 8, too_big}},
                {{block * 4, inf_value}},
                {{block * 6 + 3, too_big}},
                {{block * 7, nan_value}, {block * 2 + 10, too_big}},
                {{block * 5 + 10, too_big}, {block * 5 + 200, -inf_value}},
                {{block * 8 + 255, -inf_value}, {block * 3 + 300, too_big}, {block * 4 + 1, nan_value}},
        };
        for (const auto& bad : cases) {
            auto y = x;
            for (const auto& [position, value] : bad) {
                y[position] = value;
            }
            std::vector<float> out(n);
            const auto expected = round_outcome([&] { fp16_gemm::round_values(y.data(), out.data(), n); });
            const auto actual =
                    round_outcome([&] { fp16_gemm::round_values_parallel(y.data(), out.data(), n, block); });
            ASSERT_NE(expected, "no exception") << "test case did not actually produce a bad value";
            EXPECT_EQ(expected, actual) << "parallel error differs from serial for block=" << block;
        }
    }

    // A single block, a single thread, or already being inside a parallel
    // region must all fall back to the serial path; results must match.
    const size_t n = 3 * fp16_gemm::kRoundBlock + 5;
    const auto x = sample(n, 1, 6203);
    std::vector<float> serial(n), parallel(n);
    fp16_gemm::round_values(x.data(), serial.data(), n);
    fp16_gemm::round_values_parallel(x.data(), parallel.data(), 100, 256);
    EXPECT_EQ(0, std::memcmp(serial.data(), parallel.data(), 100 * sizeof(float)))
            << "single-block result differs from serial";
#ifdef _OPENMP
    omp_set_num_threads(1);
    std::fill(parallel.begin(), parallel.end(), 0.f);
    fp16_gemm::round_values_parallel(x.data(), parallel.data(), n);
    EXPECT_EQ(0, std::memcmp(serial.data(), parallel.data(), n * sizeof(float)))
            << "single-thread result differs from serial";
    omp_set_num_threads(4);
    std::fill(parallel.begin(), parallel.end(), 0.f);
    std::string nested = "did not run";
#pragma omp parallel num_threads(2)
    {
#pragma omp single
        nested = round_outcome([&] { fp16_gemm::round_values_parallel(x.data(), parallel.data(), n); });
    }
    EXPECT_EQ(nested, "no exception");
    EXPECT_EQ(0, std::memcmp(serial.data(), parallel.data(), n * sizeof(float)))
            << "result differs when called from inside an existing parallel region";
#endif
    {
        float v = 0;
        EXPECT_ANY_THROW(fp16_gemm::round_values_parallel(&v, &v, 1, 0)) << "zero block size was not rejected";
    }
    {
        float v = 0;
        EXPECT_ANY_THROW(fp16_gemm::round_values_parallel(&v, &v, 1, 100))
                << "block size that is not a multiple of 256 was not rejected";
    }
    {
        float v[2] = {};
        EXPECT_ANY_THROW(fp16_gemm::round_values_range(v, v, 1, 2))
                << "range not starting on a 256-element boundary was not rejected";
    }
#ifdef _OPENMP
    omp_set_num_threads(saved_threads);
#endif
}

// ---------------------------------------------------------------------
// faiss::IndexSQFP16Gemm index-level tests
// ---------------------------------------------------------------------

// Searches index and a freshly constructed native IndexScalarQuantizer
// (sharing the same codes) side by side; every id and rescored distance
// must match bit-for-bit. Used throughout the lifecycle test below, after
// each mutation, to confirm the pack cache and native codes stay consistent.
void compare_with_native(const faiss::IndexSQFP16Gemm& index, const std::vector<float>& q) {
    faiss::IndexScalarQuantizer native(index.d, faiss::ScalarQuantizer::QT_fp16, index.metric_type);
    native.add_sa_codes(index.ntotal, index.codes.data(), nullptr);
    const idx_t k = std::min<idx_t>(5, index.ntotal);
    if (k == 0) {
        return;
    }
    std::vector<float> qr(q.size());
    fp16_gemm::round_values(q.data(), qr.data(), qr.size());
    const idx_t nq = static_cast<idx_t>(q.size() / static_cast<size_t>(index.d));
    std::vector<float> da(nq * k), db(nq * k);
    std::vector<idx_t> ia(nq * k), ib(nq * k);
    index.search(nq, q.data(), k, da.data(), ia.data());
    native.search(nq, qr.data(), k, db.data(), ib.data());
    for (size_t i = 0; i < ia.size(); ++i) {
        EXPECT_EQ(ia[i], ib[i]) << "candidate ids differ from native, i=" << i;
        EXPECT_EQ(da[i], db[i]) << "rescored distance is not bit-identical to native, i=" << i;
    }
}

TEST(SQFP16GemmIndex, NativeCodecMatchesScalarQuantizer) {
    for (int d : {1, 31, 32, 33, 767, 768, 769}) {
        SCOPED_TRACE(d);
        auto x = sample(5, d, static_cast<unsigned int>(d));
        for (size_t i = 0; i < x.size(); ++i) {
            const uint16_t h = static_cast<uint16_t>(i % 0x7bff);
            x[i] = (faiss::decode_fp16(h) + faiss::decode_fp16(static_cast<uint16_t>(h + 1))) * 0.5f;
        }
        if (!x.empty()) {
            x[0] = -0.f;
        }
        if (x.size() > 1) {
            x[1] = faiss::decode_fp16(1);
        }

        faiss::IndexScalarQuantizer native(d, faiss::ScalarQuantizer::QT_fp16);
        native.add(5, x.data());

        faiss::IndexSQFP16Gemm custom(d);
        custom.kernel = fp16_gemm::GemmKernel::SCALAR;
        custom.add(5, x.data());

        ASSERT_EQ(native.codes.size(), custom.codes.size()) << "code length differs from native";
        EXPECT_EQ(0, std::memcmp(native.codes.data(), custom.codes.data(), native.codes.size()))
                << "codes differ from native IndexScalarQuantizer";

        // IndexSQFP16Gemm inherits reconstruct_n from IndexFlatCodes; the
        // round trip through fp16 must match direct encode/decode.
        std::vector<float> reconstructed(x.size());
        custom.reconstruct_n(0, 5, reconstructed.data());
        for (size_t i = 0; i < x.size(); ++i) {
            const float expected = faiss::decode_fp16(faiss::encode_fp16(x[i]));
            EXPECT_EQ(0, std::memcmp(&expected, &reconstructed[i], sizeof(float)))
                    << "reconstruct_n did not round-trip through the same fp16 rounding, i=" << i;
        }
    }
}

struct ExactMatch {
    double score;
    idx_t id;
};

std::vector<std::vector<ExactMatch>> brute_force(
        const std::vector<float>& xr,
        const std::vector<float>& qr,
        size_t nb,
        size_t nq,
        int d,
        faiss::MetricType metric) {
    const bool is_l2 = metric == faiss::METRIC_L2;
    std::vector<std::vector<ExactMatch>> exact(nq);
    for (size_t i = 0; i < nq; ++i) {
        exact[i].reserve(nb);
        for (size_t j = 0; j < nb; ++j) {
            double sum = 0;
            for (int kk = 0; kk < d; ++kk) {
                const double qv = qr[i * d + kk], bv = xr[j * d + kk];
                sum += is_l2 ? (qv - bv) * (qv - bv) : qv * bv;
            }
            exact[i].push_back({sum, static_cast<idx_t>(j)});
        }
        std::sort(exact[i].begin(), exact[i].end(), [is_l2](const ExactMatch& a, const ExactMatch& b) {
            if (a.score != b.score) {
                return is_l2 ? a.score < b.score : a.score > b.score;
            }
            return a.id < b.id;
        });
    }
    return exact;
}

TEST(SQFP16GemmIndex, SearchMatchesBruteForceReference) {
    constexpr size_t nb = 67, nq = 7, k = 11;
    for (int d : {1, 31, 32, 33, 767, 768, 769}) {
        for (auto metric : {faiss::METRIC_INNER_PRODUCT, faiss::METRIC_L2}) {
            SCOPED_TRACE(::testing::Message() << "d=" << d << " metric=" << metric);
            const auto x = sample(nb, d, static_cast<unsigned int>(700 + d));
            const auto q = sample(nq, d, static_cast<unsigned int>(900 + d));
            std::vector<float> xr(x.size()), qr(q.size());
            fp16_gemm::round_values(x.data(), xr.data(), xr.size());
            fp16_gemm::round_values(q.data(), qr.data(), qr.size());
            const auto exact = brute_force(xr, qr, nb, nq, d, metric);

            for (const auto kernel : available_kernels()) {
                SCOPED_TRACE(fp16_gemm::kernel_name(kernel));
                faiss::IndexSQFP16Gemm index(d, metric);
                index.kernel = kernel;
                index.bs_x = 3;
                index.bs_y = 39;
                index.add(nb, x.data());
                for (bool rescore : {true, false}) {
                    index.rescore_native = rescore;
                    std::vector<float> distances(nq * k);
                    std::vector<idx_t> labels(nq * k);
                    index.search(nq, q.data(), k, distances.data(), labels.data());
                    for (size_t i = 0; i < nq; ++i) {
                        const double boundary_gap = std::abs(exact[i][k - 1].score - exact[i][k].score);
                        ASSERT_GT(boundary_gap, 2e-5)
                                << "random sample has an unwanted near-tie at the top-k boundary";
                        for (size_t rank = 0; rank < k; ++rank) {
                            const idx_t id = labels[i * k + rank];
                            ASSERT_TRUE(id >= 0 && id < static_cast<idx_t>(nb)) << "returned an invalid id";
                            EXPECT_EQ(id, exact[i][rank].id)
                                    << "rank " << rank << " id differs from brute-force reference";
                            const double expected = exact[i][rank].score;
                            EXPECT_LE(std::abs(distances[i * k + rank] - expected), 1e-5 + 1e-4 * std::abs(expected))
                                    << "rank " << rank << " score outside tolerance";
                        }
                    }
                }
            }
        }
    }
}

TEST(SQFP16GemmIndex, EdgeCasesEmptyPaddingAndRejections) {
    constexpr int d = 33;
    for (const auto kernel : available_kernels()) {
        SCOPED_TRACE(fp16_gemm::kernel_name(kernel));
        faiss::IndexSQFP16Gemm index(d, faiss::METRIC_INNER_PRODUCT);
        index.kernel = kernel;
        std::vector<float> q(d, 0);
        q[0] = 1;
        std::vector<float> distances(5);
        std::vector<idx_t> ids(5);

        index.search(1, q.data(), 5, distances.data(), ids.data());
        for (size_t i = 0; i < 5; ++i) {
            EXPECT_EQ(ids[i], -1) << "empty index result was not padded with -1";
            EXPECT_EQ(distances[i], -std::numeric_limits<float>::infinity()) << "empty index padding score wrong";
        }

        // q is zero outside dim 0, so only each vector's dim-0 value
        // matters: vector 0 is clearly dominated (-2), vectors 1 and 2
        // are an exact tie (1 each) for the top spot.
        std::vector<float> x(3 * d, 0);
        x[0] = -2;
        x[d] = 1;
        x[2 * d] = 1;
        index.add(3, x.data());
        index.search(1, q.data(), 5, distances.data(), ids.data());
        EXPECT_TRUE((ids[0] == 1 && ids[1] == 2) || (ids[0] == 2 && ids[1] == 1))
                << "tied top candidates were not both ranked first, ids[0]=" << ids[0] << " ids[1]=" << ids[1];
        EXPECT_EQ(distances[0], 1.f);
        EXPECT_EQ(distances[1], 1.f);
        EXPECT_EQ(ids[2], 0) << "dominated candidate was not ranked last";
        EXPECT_EQ(distances[2], -2.f);
        EXPECT_EQ(ids[3], -1);
        EXPECT_EQ(ids[4], -1);
        EXPECT_TRUE(std::isinf(distances[3]) && distances[3] < 0) << "k > ntotal padding is wrong";

        // Candidates 1 and 2 are an exact tie: either is an acceptable
        // top-1, we only need to rule out returning the dominated one (0).
        index.search(1, q.data(), 1, distances.data(), ids.data());
        EXPECT_TRUE(ids[0] == 1 || ids[0] == 2) << "exact tie returned a dominated candidate, id=" << ids[0];

        x[0] = std::numeric_limits<float>::quiet_NaN();
        EXPECT_ANY_THROW(index.add(1, x.data())) << "adding a NaN vector did not throw";
        EXPECT_EQ(index.ntotal, 3) << "a failed add corrupted the index size";

        q[0] = 65520.f;
        EXPECT_ANY_THROW(index.search(1, q.data(), 1, distances.data(), ids.data()))
                << "a query that overflows fp16 did not throw";
    }
}

TEST(SQFP16GemmIndex, NearTiesKeepAcceptableCandidate) {
    constexpr int d = 33;
    std::vector<float> x(3 * d, 0), q(d, 0);
    q[0] = 1;
    q[1] = faiss::decode_fp16(1);
    x[0] = 1;
    x[d] = 1;
    x[d + 1] = 1;
    x[2 * d] = 0.5f;
    for (const auto kernel : available_kernels()) {
        SCOPED_TRACE(fp16_gemm::kernel_name(kernel));
        faiss::IndexSQFP16Gemm index(d, faiss::METRIC_INNER_PRODUCT);
        index.kernel = kernel;
        index.add(3, x.data());
        float distance;
        idx_t id;
        index.search(1, q.data(), 1, &distance, &id);
        EXPECT_TRUE(id == 0 || id == 1) << "near tie returned a clearly dominated candidate, id=" << id;
    }
}

void check_lifecycle(fp16_gemm::GemmKernel kernel) {
    constexpr int d = 33;
    faiss::IndexSQFP16Gemm index(d);
    index.kernel = kernel;
    index.bs_y = 33;
    const auto x = sample(65, d, 1123), q = sample(5, d, 1251);
    index.add(31, x.data());
    compare_with_native(index, q);
    index.add(34, x.data() + 31 * d);
    compare_with_native(index, q);
    index.bs_y = 65;
    compare_with_native(index, q);
    index.invalidate_pack();
    compare_with_native(index, q);

    // Directly mutating codes bypasses repack(); callers must explicitly
    // invalidate the cache, after which the next search rebuilds it.
    uint16_t changed = faiss::encode_fp16(0.375f);
    std::memcpy(index.codes.data(), &changed, sizeof(changed));
    index.invalidate_pack();
    compare_with_native(index, q);

    index.metric_type = faiss::METRIC_L2;
    compare_with_native(index, q);
    index.metric_type = faiss::METRIC_INNER_PRODUCT;
    compare_with_native(index, q);

    faiss::IDSelectorRange remove(0, 3);
    EXPECT_EQ(index.remove_ids(remove), 3u) << "remove_ids removed the wrong count";
    compare_with_native(index, q);

    std::vector<uint8_t> codes(2 * d * 2);
    index.sa_encode(2, x.data(), codes.data());
    index.add_sa_codes(2, codes.data(), nullptr);
    compare_with_native(index, q);

    faiss::IndexSQFP16Gemm other(d);
    other.kernel = kernel;
    other.add(3, x.data() + 12 * d);
    index.merge_from(other, 0);
    EXPECT_EQ(other.ntotal, 0) << "merge_from did not empty the source index";
    EXPECT_EQ(other.packed_bytes(), 0u) << "merge_from did not clear the source's pack cache";
    compare_with_native(index, q);

    // FAISS serialization only knows about the native IndexScalarQuantizer
    // state; our own fields (kernel, block sizes, rescore_native) are not
    // part of the file format and must be restored by the caller.
    faiss::VectorIOWriter writer;
    faiss::write_index(&index, &writer);
    faiss::VectorIOReader reader;
    reader.data = writer.data;
    std::unique_ptr<faiss::Index> loaded(faiss::read_index(&reader));
    auto* loaded_sq = dynamic_cast<faiss::IndexScalarQuantizer*>(loaded.get());
    ASSERT_NE(loaded_sq, nullptr) << "serialized index did not come back as IndexScalarQuantizer";
    EXPECT_EQ(loaded_sq->sq.qtype, faiss::ScalarQuantizer::QT_fp16);
    EXPECT_EQ(loaded_sq->metric_type, index.metric_type);
    EXPECT_EQ(dynamic_cast<faiss::IndexSQFP16Gemm*>(loaded.get()), nullptr)
            << "a plain IndexScalarQuantizer file must not deserialize as IndexSQFP16Gemm";

    faiss::IndexSQFP16Gemm restored(d, loaded_sq->metric_type);
    restored.kernel = kernel;
    restored.bs_x = index.bs_x;
    restored.bs_y = index.bs_y;
    restored.rescore_native = index.rescore_native;
    restored.add_sa_codes(loaded_sq->ntotal, loaded_sq->codes.data(), nullptr);
    compare_with_native(restored, q);

    // Concurrent read-only search() calls must be safe.
    index.invalidate_pack();
    std::vector<std::future<void>> tasks;
    for (int task = 0; task < 4; ++task) {
        tasks.push_back(std::async(std::launch::async, [&] {
#ifdef _OPENMP
            omp_set_num_threads(1);
#endif
            for (int repeat = 0; repeat < 3; ++repeat) {
                compare_with_native(index, q);
            }
        }));
    }
    for (auto& task : tasks) {
        task.get();
    }

    const auto saved_type = index.sq.qtype;
    index.sq.qtype = faiss::ScalarQuantizer::QT_bf16;
    EXPECT_ANY_THROW(compare_with_native(index, q)) << "format change away from QT_fp16 was not rejected";
    index.sq.qtype = saved_type;

    index.reset();
    EXPECT_EQ(index.packed_bytes(), 0u) << "reset() did not release the pack cache";
    index.add(65, x.data());
    compare_with_native(index, q);
}

TEST(SQFP16GemmIndex, LifecycleRepackMergeAndSerialization) {
    for (const auto kernel : available_kernels()) {
        SCOPED_TRACE(fp16_gemm::kernel_name(kernel));
        check_lifecycle(kernel);
    }
}

TEST(SQFP16GemmIndex, MultipleInputBlocksUseGemmPath) {
    constexpr idx_t nq = 257, nb = 83;
#ifdef _OPENMP
    const int saved_threads = omp_get_max_threads();
    omp_set_num_threads(4);
#endif
    for (const auto kernel : available_kernels()) {
        SCOPED_TRACE(fp16_gemm::kernel_name(kernel));
        for (int d : {33, 769}) {
            const auto x = sample(nb, d, static_cast<unsigned int>(2417 + d));
            const auto q = sample(nq, d, static_cast<unsigned int>(3501 + d));
            for (auto metric : {faiss::METRIC_INNER_PRODUCT, faiss::METRIC_L2}) {
                faiss::IndexSQFP16Gemm index(d, metric);
                index.kernel = kernel;
                index.bs_x = 32;
                index.bs_y = 33;
                index.add(nb, x.data());
                index.invalidate_pack();
                compare_with_native(index, q);
                EXPECT_EQ(index.nb_gemm_queries.load(), static_cast<size_t>(nq))
                        << "multi-block query did not take the gemm path";
            }
        }
    }
#ifdef _OPENMP
    omp_set_num_threads(saved_threads);
#endif
}

TEST(SQFP16GemmIndex, QueryCountsRowConsistency) {
    constexpr idx_t nb = 83, k = 5;
    for (const auto kernel : available_kernels()) {
        SCOPED_TRACE(fp16_gemm::kernel_name(kernel));
        for (int d : {768, 769}) {
            const auto x = sample(nb, d, static_cast<unsigned int>(4211 + d));
            const auto q = sample(40, d, static_cast<unsigned int>(4307 + d));
            faiss::IndexSQFP16Gemm index(d);
            index.kernel = kernel;
            index.bs_y = 64;
            index.add(nb, x.data());
            for (idx_t nq : {1, 2, 15, 16, 17, 31, 33, 40}) {
                const std::vector<float> qs(q.begin(), q.begin() + nq * d);
                index.rescore_native = true;
                compare_with_native(index, qs);
                index.rescore_native = false;
                std::vector<float> batch_d(nq * k), one_d(k);
                std::vector<idx_t> batch_i(nq * k), one_i(k);
                index.search(nq, qs.data(), k, batch_d.data(), batch_i.data());
                for (idx_t i = 0; i < nq; ++i) {
                    index.search(1, qs.data() + i * d, k, one_d.data(), one_i.data());
                    EXPECT_EQ(0, std::memcmp(one_d.data(), batch_d.data() + i * k, k * sizeof(float)))
                            << "single-query score differs from the same row in a batch, nq=" << nq << " i=" << i;
                    EXPECT_TRUE(std::equal(one_i.begin(), one_i.end(), batch_i.begin() + i * k))
                            << "single-query ids differ from the same row in a batch, nq=" << nq << " i=" << i;
                }
            }
            index.rescore_native = true;
        }
    }
}

// Exercises IndexSQFP16Gemm::search()'s non-gemm path (use_gemm == false),
// which rounds the whole query batch with round_values_parallel before
// handing off to the inherited IndexScalarQuantizer::search(). The query
// batch below crosses several rounding blocks, including two with a bad
// value, so this also checks that round_values_parallel's error reporting
// (see RoundValuesParallelMatchesSerial) is actually wired into search().
TEST(SQFP16GemmIndex, ParallelRoundSearchMatchesNativeScalarQuantizer) {
#ifdef _OPENMP
    const int saved_threads = omp_get_max_threads();
    omp_set_num_threads(4);
#endif
    constexpr int d = 769;
    constexpr idx_t nb = 97, nq = 200, k = 4;
    static_assert(
            static_cast<size_t>(nq) * d > 2 * fp16_gemm::kRoundBlock,
            "query batch must cross at least three rounding blocks");
    const auto x = sample(nb, d, 5101), q = sample(nq, d, 5203);
    std::vector<float> xr(x.size()), qr(q.size());
    fp16_gemm::round_values(x.data(), xr.data(), xr.size());
    fp16_gemm::round_values(q.data(), qr.data(), qr.size());
    auto bad = q;
    bad[100 * d + 5] = 65520.f; // second block: overflows once rounded
    bad[171 * d] = std::numeric_limits<float>::quiet_NaN(); // third block: NaN
    std::vector<float> scratch(q.size());
    const auto expected_error =
            round_outcome([&] { fp16_gemm::round_values(bad.data(), scratch.data(), bad.size()); });
    ASSERT_NE(expected_error.find("overflow"), std::string::npos)
            << "constructed bad query did not report overflow first: " << expected_error;

    for (auto metric : {faiss::METRIC_INNER_PRODUCT, faiss::METRIC_L2}) {
        const bool is_l2 = metric == faiss::METRIC_L2;
        SCOPED_TRACE(metric);
        faiss::IndexSQFP16Gemm index(d, metric);
        index.use_gemm = false; // force the round + native search() path
        index.add(nb, x.data());

        faiss::IndexScalarQuantizer native(d, faiss::ScalarQuantizer::QT_fp16, metric);
        native.add(nb, x.data());

        std::vector<float> got_d(nq * k), ref_d(nq * k);
        std::vector<idx_t> got_i(nq * k), ref_i(nq * k);
        index.search(nq, q.data(), k, got_d.data(), got_i.data());
        native.search(nq, qr.data(), k, ref_d.data(), ref_i.data());
        for (idx_t i = 0; i < nq; ++i) {
            // Native results are re-sorted by (score, id) to match
            // IndexSQFP16Gemm::search()'s own tie-breaking rule.
            std::vector<std::pair<float, idx_t>> row;
            for (idx_t j = 0; j < k; ++j) {
                row.push_back({ref_d[i * k + j], ref_i[i * k + j]});
            }
            std::sort(row.begin(), row.end(), [is_l2](const auto& a, const auto& b) {
                if (a.first != b.first) {
                    return is_l2 ? a.first < b.first : a.first > b.first;
                }
                return a.second < b.second;
            });
            for (idx_t j = 0; j < k; ++j) {
                EXPECT_EQ(got_i[i * k + j], row[j].second) << "id differs from native, query=" << i;
                EXPECT_EQ(got_d[i * k + j], row[j].first) << "score is not bit-identical to native, query=" << i;
            }
        }
        const auto actual_error =
                round_outcome([&] { index.search(nq, bad.data(), k, got_d.data(), got_i.data()); });
        EXPECT_EQ(actual_error, expected_error) << "batched search error differs from serial conversion";
    }
#ifdef _OPENMP
    omp_set_num_threads(saved_threads);
#endif
}

} // namespace
