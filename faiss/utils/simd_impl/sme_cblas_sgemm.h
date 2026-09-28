/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

// Shared by faiss/utils/distances.cpp and
// faiss/utils/simd_impl/distances_arm_sve.cpp, both of which declare their
// own `extern "C"` FINTEGER-based `sgemm_`/`cblas_sgemm` prototypes ahead of
// including this header.
//
// FAISS_SME_CBLAS_SGEMM redirects hot sgemm_ call sites through
// cblas_sgemm(RowMajor, NoTrans, NoTrans, ...) instead of the Fortran sgemm_
// entry point, since only the CBLAS entry reaches OpenBLAS's SME
// direct-sgemm dispatch (interface/gemm.c). No-op when undefined.
#ifdef FAISS_SME_CBLAS_SGEMM

#include <type_traits>
#include <vector>

namespace faiss {

namespace {

// FAISS_ENABLE_SME_CBLAS forces FINTEGER == int; the (int) casts below rely
// on that.
static_assert(
        std::is_same_v<FINTEGER, int>,
        "FAISS_SME_CBLAS_SGEMM requires FINTEGER == int");

constexpr int kCblasRowMajor = 101;
constexpr int kCblasNoTrans = 111;

// OpenBLAS's SME dispatch only fires for NoTrans/NoTrans, so B is
// transpose-packed here rather than passed with TransB=Trans.
void sme_cblas_sgemm_tn(
        FINTEGER M,
        FINTEGER N,
        FINTEGER K,
        float alpha,
        const float* A,
        FINTEGER ldA,
        const float* B,
        FINTEGER ldB,
        float beta,
        float* C,
        FINTEGER ldC,
        std::vector<float>& scratch) {
    scratch.resize((size_t)K * (size_t)M);
    float* At = scratch.data();
    for (FINTEGER i = 0; i < M; i++) {
        const float* row = A + (size_t)i * (size_t)ldA;
        for (FINTEGER l = 0; l < K; l++) {
            At[(size_t)l * (size_t)M + i] = row[l];
        }
    }
    cblas_sgemm(
            kCblasRowMajor,
            kCblasNoTrans,
            kCblasNoTrans,
            (int)N,
            (int)M,
            (int)K,
            alpha,
            B,
            (int)ldB,
            At,
            (int)M,
            beta,
            C,
            (int)ldC);
}

} // namespace

} // namespace faiss

#endif // FAISS_SME_CBLAS_SGEMM
