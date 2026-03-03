/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Private implementation header for per-SIMD scalar quantizer TUs.
// Do not include in public APIs.

#pragma once

#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/utils/simd_levels.h>
#include <faiss/utils/simdlib.h>

#include <faiss/impl/simd_dispatch.h>

#ifdef __SSE__
#include <immintrin.h>
#endif

#include <faiss/IndexIVF.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/expanded_scanners.h>
#include <faiss/utils/bf16.h>
#include <faiss/utils/fp16.h>

// Define USE_* macros for the sub-headers. These macros gate struct template
// specializations that contain SIMD intrinsics. They must use compiler-defined
// macros (not COMPILE_SIMD_*) because in DD mode the COMPILE_SIMD_* macros are
// set globally, but the actual SIMD instructions are only available in per-SIMD
// TUs that receive the appropriate compiler flags.
#if defined(__AVX512F__) && defined(__F16C__)
#define USE_AVX512_F16C
#endif

#if defined(__AVX2__) && defined(__F16C__)
#define USE_F16C
#endif

#if defined(__aarch64__)
#if !defined(__GNUC__) || __GNUC__ >= 8
#define USE_NEON
#endif
#endif

#include <faiss/impl/scalar_quantizer/codecs.h>
#include <faiss/impl/scalar_quantizer/distance_computers.h>
#include <faiss/impl/scalar_quantizer/quantizers.h>
#include <faiss/impl/scalar_quantizer/similarities.h>

namespace faiss {

namespace scalar_quantizer {

using QuantizerType = ScalarQuantizer::QuantizerType;
using SQDistanceComputer = ScalarQuantizer::SQDistanceComputer;

/*******************************************************************
 * IVFSQScannerIP / IVFSQScannerL2 — moved from anonymous namespace
 * in ScalarQuantizer.cpp
 *******************************************************************/

template <class DCClass>
struct IVFSQScannerIP : InvertedListScanner {
    DCClass dc;
    bool by_residual;

    float accu0; /// added to all distances

    IVFSQScannerIP(
            int d,
            const std::vector<float>& trained,
            size_t code_size,
            bool store_pairs,
            const IDSelector* sel,
            bool by_residual)
            : dc(d, trained), by_residual(by_residual), accu0(0) {
        this->store_pairs = store_pairs;
        this->sel = sel;
        this->code_size = code_size;
        this->keep_max = true;
    }

    void set_query(const float* query) override {
        dc.set_query(query);
    }

    void set_list(idx_t list_no, float coarse_dis) override {
        this->list_no = list_no;
        accu0 = by_residual ? coarse_dis : 0;
    }

    float distance_to_code(const uint8_t* code) const final {
        return accu0 + dc.query_to_code(code);
    }

    size_t scan_codes(
            size_t list_size,
            const uint8_t* codes,
            const idx_t* ids,
            ResultHandler& handler) const override {
        return run_scan_codes_fix_C<CMin<float, idx_t>>(
                *this, list_size, codes, ids, handler);
    }
};

template <class DCClass>
struct IVFSQScannerL2 : InvertedListScanner {
    DCClass dc;

    bool by_residual;
    const Index* quantizer;
    const float* x; /// current query

    std::vector<float> tmp;

    IVFSQScannerL2(
            int d,
            const std::vector<float>& trained,
            size_t code_size,
            const Index* quantizer,
            bool store_pairs,
            const IDSelector* sel,
            bool by_residual)
            : dc(d, trained),
              by_residual(by_residual),
              quantizer(quantizer),
              x(nullptr),
              tmp(d) {
        this->store_pairs = store_pairs;
        this->sel = sel;
        this->code_size = code_size;
    }

    void set_query(const float* query) override {
        x = query;
        if (!quantizer) {
            dc.set_query(query);
        }
    }

    void set_list(idx_t list_no, float /*coarse_dis*/) override {
        this->list_no = list_no;
        if (by_residual) {
            quantizer->compute_residual(x, tmp.data(), list_no);
            dc.set_query(tmp.data());
        } else {
            dc.set_query(x);
        }
    }

    float distance_to_code(const uint8_t* code) const final {
        return dc.query_to_code(code);
    }

    size_t scan_codes(
            size_t list_size,
            const uint8_t* codes,
            const idx_t* ids,
            ResultHandler& handler) const override {
        return run_scan_codes_fix_C<CMax<float, idx_t>>(
                *this, list_size, codes, ids, handler);
    }
};

/*******************************************************************
 * Forward declaration of inverts list scanner
 *******************************************************************/

template <SIMDLevel SL>
InvertedListScanner* sq_select_InvertedListScanner(
        QuantizerType qtype,
        MetricType mt,
        size_t d,
        size_t code_size,
        const std::vector<float>& trained,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool by_residual);

} // namespace scalar_quantizer

} // namespace faiss
