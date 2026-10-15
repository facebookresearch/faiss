/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Decodes fp16 inputs to float and multiply-accumulates in fp32. This file
// is compiled with its own AVX-512F/BW, F16C and FMA flags (see
// CMakeLists.txt); the rest of the fp16 gemm kernels do not require them.
#include <faiss/impl/scalar_quantizer/sq-fp16-gemm.h>

#include <immintrin.h>
#include <cmath>
#include <vector>

#include <faiss/utils/fp16.h>

namespace faiss {
namespace fp16_gemm {

void encode_array_fma(const float* src, uint16_t* dst, size_t n) {
    size_t i = 0;
    for (; i + 16 <= n; i += 16) {
        const __m256i value = _mm512_cvtps_ph(
                _mm512_loadu_ps(src + i),
                _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
        _mm256_storeu_si256(reinterpret_cast<__m256i*>(dst + i), value);
    }
    // The tail uses the exact same round-to-nearest-even path, so results
    // never depend on position or length.
    for (; i < n; ++i) {
        dst[i] = faiss::encode_fp16(src[i]);
    }
}

namespace {
// ROWS rows share one pass over the packed table. Each row keeps a single
// accumulator per 16-column group and accumulates strictly in dimension
// order, so a row's result does not depend on how many other rows share
// the pass, and matches bit-for-bit what a dedicated 1-row-at-a-time pass
// would produce.
template <size_t ROWS>
void fma_rows(
        const float* expanded,
        size_t full,
        const uint16_t* packed_b,
        size_t n_padded,
        float* out,
        size_t out_stride) {
    const size_t kblocks = full / kFP16TileK;
    for (size_t j = 0; j < n_padded; j += kFP16BlockN) {
        // Every dimension of a 16-centroid group is stored contiguously;
        // the two adjacent groups making up these 32 centroids are each
        // read start to end as one sequential stream.
        const uint16_t* group0 = packed_b + (j / kFP16TileN) * kblocks * 512;
        const uint16_t* group1 = group0 + kblocks * 512;
        __m512 acc0[ROWS];
        __m512 acc1[ROWS];
        for (size_t r = 0; r < ROWS; ++r) {
            acc0[r] = _mm512_setzero_ps();
            acc1[r] = _mm512_setzero_ps();
        }
        for (size_t kb = 0; kb < kblocks; ++kb) {
            const uint16_t* block0 = group0 + kb * 512;
            const uint16_t* block1 = group1 + kb * 512;
            for (size_t kk = 0; kk < 16; ++kk) {
                const __m512i b0 = _mm512_loadu_si512(block0 + kk * 32);
                const __m512i b1 = _mm512_loadu_si512(block1 + kk * 32);
                const __m512 b0e = _mm512_cvtph_ps(_mm512_cvtepi32_epi16(b0));
                const __m512 b0o = _mm512_cvtph_ps(
                        _mm512_cvtepi32_epi16(_mm512_srli_epi32(b0, 16)));
                const __m512 b1e = _mm512_cvtph_ps(_mm512_cvtepi32_epi16(b1));
                const __m512 b1o = _mm512_cvtph_ps(
                        _mm512_cvtepi32_epi16(_mm512_srli_epi32(b1, 16)));
                for (size_t r = 0; r < ROWS; ++r) {
                    const float* row = expanded + r * full + kb * 32;
                    const __m512 ae = _mm512_set1_ps(row[kk * 2]);
                    const __m512 ao = _mm512_set1_ps(row[kk * 2 + 1]);
                    acc0[r] = _mm512_fmadd_ps(ae, b0e, acc0[r]);
                    acc1[r] = _mm512_fmadd_ps(ae, b1e, acc1[r]);
                    acc0[r] = _mm512_fmadd_ps(ao, b0o, acc0[r]);
                    acc1[r] = _mm512_fmadd_ps(ao, b1o, acc1[r]);
                }
            }
        }
        for (size_t r = 0; r < ROWS; ++r) {
            float* row = out + r * out_stride + j;
            _mm512_storeu_ps(row, acc0[r]);
            _mm512_storeu_ps(row + 16, acc1[r]);
        }
    }
}
} // namespace

void gemm_fma_impl(
        const uint16_t* a,
        size_t m,
        size_t a_stride,
        const uint16_t* packed_b,
        const uint16_t* b_rows,
        size_t b_stride,
        size_t n_padded,
        size_t n_real,
        size_t d,
        float* out,
        size_t out_stride) {
    const size_t full = (d / kFP16TileK) * kFP16TileK;
    constexpr size_t max_rows = 8;
    // One decode buffer per thread, reused across calls instead of
    // allocating and zeroing it every time.
    static thread_local std::vector<float> expanded;
    if (expanded.size() < max_rows * full) {
        expanded.resize(max_rows * full);
    }

    // At most 8 rows per pass; the last pass handles however many real
    // rows remain.
    for (size_t i = 0; i < m;) {
        const size_t rows = m - i < max_rows ? m - i : max_rows;
        for (size_t r = 0; r < rows; ++r) {
            const uint16_t* row = a + (i + r) * a_stride;
            for (size_t kk = 0; kk < full; kk += 16) {
                const __m256i half = _mm256_loadu_si256(
                        reinterpret_cast<const __m256i*>(row + kk));
                _mm512_storeu_ps(
                        expanded.data() + r * full + kk, _mm512_cvtph_ps(half));
            }
        }
        const float* e = expanded.data();
        float* o = out + i * out_stride;
        switch (rows) {
            case 8:
                fma_rows<8>(e, full, packed_b, n_padded, o, out_stride);
                break;
            case 7:
                fma_rows<7>(e, full, packed_b, n_padded, o, out_stride);
                break;
            case 6:
                fma_rows<6>(e, full, packed_b, n_padded, o, out_stride);
                break;
            case 5:
                fma_rows<5>(e, full, packed_b, n_padded, o, out_stride);
                break;
            case 4:
                fma_rows<4>(e, full, packed_b, n_padded, o, out_stride);
                break;
            case 3:
                fma_rows<3>(e, full, packed_b, n_padded, o, out_stride);
                break;
            case 2:
                fma_rows<2>(e, full, packed_b, n_padded, o, out_stride);
                break;
            default:
                fma_rows<1>(e, full, packed_b, n_padded, o, out_stride);
                break;
        }
        i += rows;
    }
    if (full < d) {
        for (size_t i = 0; i < m; ++i) {
            for (size_t j = 0; j < n_real; ++j) {
                float acc = out[i * out_stride + j];
                for (size_t kk = full; kk < d; ++kk) {
                    acc = std::fma(
                            faiss::decode_fp16(a[i * a_stride + kk]),
                            faiss::decode_fp16(b_rows[j * b_stride + kk]),
                            acc);
                }
                out[i * out_stride + j] = acc;
            }
        }
    }
}

} // namespace fp16_gemm
} // namespace faiss
