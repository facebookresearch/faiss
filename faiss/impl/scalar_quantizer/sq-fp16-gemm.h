/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>

// Batched GEMM-style kernels for the QT_fp16 scalar quantizer.
//
// IndexScalarQuantizer with QT_fp16 normally scores one database vector at
// a time through ScalarQuantizer::get_distance_computer(). For an IVF coarse
// quantizer that scores every centroid against every query, it is much
// faster to pack the fp16 centroids once and run a dedicated small-matrix
// multiply instead. This header declares that packing format together with
// three interchangeable kernels (plain scalar, AVX-512/F16C/FMA, and
// AMX-FP16) selected at runtime by CPUID, with a compile-time switch for
// AMX-FP16 since it requires a recent compiler and is not available on all
// targets that otherwise build the rest of faiss.
namespace faiss {
namespace fp16_gemm {

// Centroids are packed 16 at a time (a "group"), with every dimension of
// the group stored contiguously, 32 dimensions per 1 KiB segment. The AMX
// kernel processes two adjacent groups (32 centroids) per pass.
constexpr size_t kFP16TileM = 16;
constexpr size_t kFP16TileN = 16;
constexpr size_t kFP16TileK = 32;
// The AMX kernel scores up to 32 query rows at a time (two row tiles); with
// only 16 rows left it only loads one.
constexpr size_t kFP16BlockM = 32;
constexpr size_t kFP16BlockN = 32;

enum class GemmKernel { AUTO, AMX, FMA, SCALAR };

bool amx_fp16_available();
bool fma_available();
bool amx_fp16_compiled();

struct AMXSubnormalBehavior {
    float a_probe = 0;
    float b_probe = 0;
    bool preserves = false;
};
// Result of an actual AMX-FP16 probe instruction: different CPU generations
// flush subnormal fp16 inputs to zero or preserve them. The kernel uses this
// to decide whether it needs to patch in the subnormal contributions itself.
AMXSubnormalBehavior amx_subnormal_behavior();
bool amx_preserves_subnormals();

// requested == AUTO picks the best available kernel. An explicit request
// for a kernel that is not available throws instead of silently falling
// back, so callers can tell the difference between "used AMX" and
// "AMX was unavailable, used FMA instead".
GemmKernel resolve_kernel(GemmKernel requested);
const char* kernel_name(GemmKernel kernel);

// Bulk fp16 <-> fp32 conversion. Element-wise results match
// faiss::encode_fp16/decode_fp16 (round-to-nearest-even, exact for +/-0,
// subnormals and finite overflow-to-infinity); the array entry points exist
// to pick the AVX-512/F16C bulk path when available instead of converting
// one value at a time.
void encode_array(const float* src, uint16_t* dst, size_t n);
void encode_rows(const float* src, uint16_t* dst, size_t n, size_t d);
void decode_array(const uint16_t* src, float* dst, size_t n);

// Rejects non-finite inputs and values that overflow to +/-inf once
// rounded to fp16; safe to call with dst == src (as float, not in fp16
// code units).
void round_values(const float* src, float* dst, size_t n);
// Only processes elements [begin, end); errors are reported with the
// global element index so that splitting a range into chunks that start
// on 256-element boundaries reports the same first error as processing it
// as a single range would.
void round_values_range(const float* src, float* dst, size_t begin, size_t end);

// Parallel version of round_values: splits [0, n) into chunks of `block`
// elements (a multiple of 256) and rounds each chunk on its own OpenMP
// thread. Results, exception types and the reported element index all
// match round_values; falls back to it when there is only one chunk,
// already inside a parallel region, or only one thread is available. Safe
// to call with dst == src; on exception the contents of dst are undefined.
constexpr size_t kRoundBlock = size_t{1} << 16;
void round_values_parallel(
        const float* src, float* dst, size_t n, size_t block = kRoundBlock);

void validate_codes(const uint8_t* codes, size_t nvalues);

// Packed size, in uint16_t elements, of n_padded (multiple of 32) centroids
// of dimension d.
size_t packed_b_size(size_t n_padded, size_t d);
void pack_centroids(
        const uint16_t* centroids,
        size_t n_real,
        size_t c_stride,
        size_t n_padded,
        size_t d,
        uint16_t* packed);

// Computes the full m x n_padded inner-product matrix, accumulated in fp32.
// n_padded must be a multiple of kFP16BlockN; m must be a multiple of
// row_multiple(kernel) (only the first m rows are written). Strides are in
// elements; out_stride >= n_padded. Trailing dimensions that don't fill a
// full 32-wide block are handled by a scalar tail pass in every kernel.
using GemmFn = void (*)(
        const uint16_t*, size_t, size_t, const uint16_t*, const uint16_t*,
        size_t, size_t, size_t, size_t, float*, size_t);

void gemm_scalar(
        const uint16_t* a, size_t m, size_t a_stride,
        const uint16_t* packed_b, const uint16_t* b_rows, size_t b_stride,
        size_t n_padded, size_t n_real, size_t d, float* out,
        size_t out_stride);
void gemm_fma(
        const uint16_t* a, size_t m, size_t a_stride,
        const uint16_t* packed_b, const uint16_t* b_rows, size_t b_stride,
        size_t n_padded, size_t n_real, size_t d, float* out,
        size_t out_stride);
void gemm_amx(
        const uint16_t* a, size_t m, size_t a_stride,
        const uint16_t* packed_b, const uint16_t* b_rows, size_t b_stride,
        size_t n_padded, size_t n_real, size_t d, float* out,
        size_t out_stride);
GemmFn get_gemm_fn(GemmKernel kernel);

// Row count that m must be padded to: the AMX kernel scores a full 16-row
// tile at a time, FMA and scalar process rows one at a time and need no
// padding.
size_t gemm_row_multiple(GemmKernel kernel);

// Shared shape validation used by every gemm_* entry point.
void validate_gemm_shape(
        size_t m, size_t a_stride, size_t b_stride, size_t n_padded,
        size_t n_real, size_t d, size_t out_stride, size_t row_multiple);

} // namespace fp16_gemm
} // namespace faiss
