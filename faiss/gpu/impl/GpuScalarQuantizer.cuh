/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/IndexScalarQuantizer.h>
#include <faiss/gpu/utils/ConversionOperators.cuh>
#include <faiss/gpu/utils/DeviceTensor.cuh>
#include <faiss/gpu/utils/HostTensor.cuh>
#include <faiss/gpu/utils/PtxUtils.cuh>
#include <faiss/gpu/utils/WarpShuffles.cuh>

namespace faiss {
namespace gpu {

inline bool isSQSupported(ScalarQuantizer::QuantizerType qtype) {
    switch (qtype) {
        case ScalarQuantizer::QuantizerType::QT_8bit:
        case ScalarQuantizer::QuantizerType::QT_8bit_uniform:
        case ScalarQuantizer::QuantizerType::QT_8bit_direct:
        case ScalarQuantizer::QuantizerType::QT_4bit:
        case ScalarQuantizer::QuantizerType::QT_4bit_uniform:
        case ScalarQuantizer::QuantizerType::QT_6bit:
        case ScalarQuantizer::QuantizerType::QT_fp16:
            return true;
        default:
            return false;
    }
}

// Wrapper around the CPU ScalarQuantizer that allows storage of parameters in
// GPU memory
struct GpuScalarQuantizer : public ScalarQuantizer {
    GpuScalarQuantizer(GpuResources* res, const ScalarQuantizer& sq)
            : ScalarQuantizer(sq),
              gpuTrained(DeviceTensor<float, 1, true>(
                      res,
                      makeDevAlloc(AllocType::Quantizer, 0),
                      {(idx_t)sq.trained.size()})) {
        HostTensor<float, 1, true> cpuTrained(
                (float*)sq.trained.data(), {(idx_t)sq.trained.size()});

        auto stream = res->getDefaultStreamCurrentDevice();
        gpuTrained.copyFrom(cpuTrained, stream);
    }

    // ScalarQuantizer::trained copied to GPU memory
    DeviceTensor<float, 1, true> gpuTrained;
};

//
// Quantizer codecs
//

// QT is the quantizer type implemented
// DimMultiple is the minimum guaranteed dimension multiple of the vectors
// encoded (used for ensuring alignment for memory load/stores)
template <int QT, int DimMultiple>
struct Codec {};

/////
//
// 32 bit encodings
// (does not use qtype)
//
/////

struct CodecFloat {
    /// How many dimensions per iteration we are handling for encoding or
    /// decoding
    static constexpr int kDimPerIter = 1;

    CodecFloat(int vecBytes) : bytesPerVec(vecBytes) {}

    size_t getSmemSize(int dim) {
        return 0;
    }
    inline __device__ void initKernel(float* smem, int dim) {}

    inline __device__ void decode(void* data, idx_t vec, int d, float* out)
            const {
        float* p = (float*)&((uint8_t*)data)[vec * bytesPerVec];
        out[0] = p[d];
    }

    inline __device__ float decodePartial(
            void* data,
            idx_t vec,
            int d,
            int subD) const {
        // doesn't need implementing (kDimPerIter == 1)
        return 0.0f;
    }

    inline __device__ void encode(
            void* data,
            idx_t vec,
            int d,
            float v[kDimPerIter]) const {
        float* p = (float*)&((uint8_t*)data)[vec * bytesPerVec];
        p[d] = v[0];
    }

    inline __device__ void encodePartial(
            void* data,
            idx_t vec,
            int d,
            int remaining,
            float v[kDimPerIter]) const {
        // doesn't need implementing (kDimPerIter == 1)
    }

    //
    // new implementation
    //
    using EncodeT = float;
    static constexpr int kEncodeBits = 32;

    inline __device__ EncodeT encodeNew(int dim, float v) const {
        return v;
    }

    inline __device__ float decodeNew(int dim, EncodeT v) const {
        return v;
    }

    int bytesPerVec;
};

/////
//
// 16 bit encodings
//
/////

// Arbitrary dimension fp16
template <>
struct Codec<ScalarQuantizer::QuantizerType::QT_fp16, 1> {
    /// How many dimensions per iteration we are handling for encoding or
    /// decoding
    static constexpr int kDimPerIter = 1;

    Codec(int vecBytes) : bytesPerVec(vecBytes) {}

    size_t getSmemSize(int dim) {
        return 0;
    }
    inline __device__ void initKernel(float* smem, int dim) {}

    inline __device__ void decode(void* data, idx_t vec, int d, float* out)
            const {
        half* p = (half*)&((uint8_t*)data)[vec * bytesPerVec];
        out[0] = ConvertTo<float>::to(p[d]);
    }

    inline __device__ float decodePartial(
            void* data,
            idx_t vec,
            int d,
            int subD) const {
        // doesn't need implementing (kDimPerIter == 1)
        return 0.0f;
    }

    inline __device__ void encode(
            void* data,
            idx_t vec,
            int d,
            float v[kDimPerIter]) const {
        half* p = (half*)&((uint8_t*)data)[vec * bytesPerVec];
        p[d] = ConvertTo<half>::to(v[0]);
    }

    inline __device__ void encodePartial(
            void* data,
            idx_t vec,
            int d,
            int remaining,
            float v[kDimPerIter]) const {
        // doesn't need implementing (kDimPerIter == 1)
    }

    //
    // new implementation
    //
    using EncodeT = half;
    static constexpr int kEncodeBits = 16;

    inline __device__ EncodeT encodeNew(int dim, float v) const {
        return ConvertTo<half>::to(v);
    }

    inline __device__ float decodeNew(int dim, EncodeT v) const {
        return ConvertTo<float>::to(v);
    }

    int bytesPerVec;
};

/////
//
// 8 bit encodings
//
/////

template <int DimPerIter>
struct Get8BitType {};

template <>
struct Get8BitType<1> {
    using T = uint8_t;
};

template <>
struct Get8BitType<2> {
    using T = uint16_t;
};

template <>
struct Get8BitType<4> {
    using T = uint32_t;
};

// Uniform quantization across all dimensions
template <int DimMultiple>
struct Codec<ScalarQuantizer::QuantizerType::QT_8bit_uniform, DimMultiple> {
    /// How many dimensions per iteration we are handling for encoding or
    /// decoding
    static constexpr int kDimPerIter = DimMultiple;
    using MemT = typename Get8BitType<DimMultiple>::T;

    Codec(int vecBytes, float min, float diff)
            : bytesPerVec(vecBytes), vmin(min), vdiff(diff) {}

    size_t getSmemSize(int dim) {
        return 0;
    }
    inline __device__ void initKernel(float* smem, int dim) {
        // We are performing vmin + vdiff * (v + 0.5) / (2^bits - 1)
        // This can be simplified to vmin' + vdiff' * v where:
        // vdiff' = vdiff / (2^bits - 1)
        // vmin' = vmin + 0.5 * vdiff'
        auto vd = vdiff * (1.0f / 255.0f);
        vmin = vmin + 0.5f * vd;
        vdiff = vd;
    }

    inline __device__ float decodeHelper(uint8_t v) const {
        return vmin + (float)v * vdiff;
    }

    inline __device__ void decode(void* data, idx_t vec, int d, float* out)
            const {
        MemT* p = (MemT*)&((uint8_t*)data)[vec * bytesPerVec];
        MemT pv = p[d];

        uint8_t x[kDimPerIter];
#pragma unroll
        for (int i = 0; i < kDimPerIter; ++i) {
            x[i] = (uint8_t)((pv >> (i * 8)) & 0xffU);
        }

        float xDec[kDimPerIter];
#pragma unroll
        for (int i = 0; i < kDimPerIter; ++i) {
            xDec[i] = decodeHelper(x[i]);
        }

#pragma unroll
        for (int i = 0; i < kDimPerIter; ++i) {
            out[i] = xDec[i];
        }
    }

    inline __device__ float decodePartial(
            void* data,
            idx_t vec,
            int d,
            int subD) const {
        if (DimMultiple > 1) {
            // should not be called
            assert(false);
        }

        // otherwise does not need implementing
        return 0;
    }

    inline __device__ uint8_t encodeHelper(float v) const {
        float x = (v - vmin) / vdiff;
        x = fminf(1.0f, fmaxf(0.0f, x));
        return (uint8_t)(x * 255.0f);
    }

    inline __device__ void encode(
            void* data,
            idx_t vec,
            int d,
            float v[kDimPerIter]) const {
        MemT* p = (MemT*)&((uint8_t*)data)[vec * bytesPerVec];

        MemT x[kDimPerIter];
#pragma unroll
        for (int i = 0; i < kDimPerIter; ++i) {
            x[i] = encodeHelper(v[i]);
        }

        MemT out = 0;
#pragma unroll
        for (int i = 0; i < kDimPerIter; ++i) {
            out |= (x[i] << (i * 8));
        }

        p[d] = out;
    }

    inline __device__ void encodePartial(
            void* data,
            idx_t vec,
            int d,
            int remaining,
            float v[kDimPerIter]) const {
        if (DimMultiple > 1) {
            // should not be called
            assert(false);
        }

        // otherwise does not need implementing
    }

    //
    // interleaved code implementation
    //
    using EncodeT = uint8_t;
    static constexpr int kEncodeBits = 8;

    inline __device__ EncodeT encodeNew(int dim, float v) const {
        return encodeHelper(v);
    }

    inline __device__ float decodeNew(int dim, EncodeT v) const {
        return decodeHelper(v);
    }

    int bytesPerVec;
    float vmin;
    float vdiff;
};

// Uniform quantization per each dimension
template <int DimMultiple>
struct Codec<ScalarQuantizer::QuantizerType::QT_8bit, DimMultiple> {
    /// How many dimensions per iteration we are handling for encoding or
    /// decoding
    static constexpr int kDimPerIter = DimMultiple;
    using MemT = typename Get8BitType<DimMultiple>::T;

    Codec(int vecBytes, float* min, float* diff)
            : bytesPerVec(vecBytes),
              vmin(min),
              vdiff(diff),
              smemVmin(nullptr),
              smemVdiff(nullptr) {}

    size_t getSmemSize(int dim) {
        return sizeof(float) * dim * 2;
    }

    // Initialize shared memory and local storage
    // It is up to the user to call a trailing syncthreads (after any other
    // initialization required has been done)
    inline __device__ void initKernel(float* smem, int dim) {
        smemVmin = smem;
        smemVdiff = smem + dim;

        for (auto i = threadIdx.x; i < dim; i += blockDim.x) {
            // We are performing vmin + vdiff * (v + 0.5) / (2^bits - 1)
            // This can be simplified to vmin' + vdiff' * v where:
            // vdiff' = vdiff / (2^bits - 1)
            // vmin' = vmin + 0.5 * vdiff'
            auto vd = vdiff[i] * (1.0f / 255.0f);
            smemVmin[i] = vmin[i] + 0.5f * vd;
            smemVdiff[i] = vd;
        }
    }

    inline __device__ float decodeHelper(uint8_t v, int realDim) const {
        return smemVmin[realDim] + (float)v * smemVdiff[realDim];
    }

    inline __device__ void decode(void* data, idx_t vec, int d, float* out)
            const {
        MemT* p = (MemT*)&((uint8_t*)data)[vec * bytesPerVec];
        MemT pv = p[d];
        int realDim = d * kDimPerIter;

        uint8_t x[kDimPerIter];
#pragma unroll
        for (int i = 0; i < kDimPerIter; ++i) {
            x[i] = (uint8_t)((pv >> (i * 8)) & 0xffU);
        }

        float xDec[kDimPerIter];
#pragma unroll
        for (int i = 0; i < kDimPerIter; ++i) {
            xDec[i] = decodeHelper(x[i], realDim + i);
        }

#pragma unroll
        for (int i = 0; i < kDimPerIter; ++i) {
            out[i] = xDec[i];
        }
    }

    inline __device__ float decodePartial(
            void* data,
            idx_t vec,
            int d,
            int subD) const {
        if (DimMultiple > 1) {
            // should not be called
            assert(false);
        }

        // otherwise does not need implementing
        return 0;
    }

    inline __device__ uint8_t encodeHelper(float v, int realDim) const {
        float x = (v - vmin[realDim]) / vdiff[realDim];
        x = fminf(1.0f, fmaxf(0.0f, x));
        return (uint8_t)(x * 255.0f);
    }

    inline __device__ void encode(
            void* data,
            idx_t vec,
            int d,
            float v[kDimPerIter]) const {
        MemT* p = (MemT*)&((uint8_t*)data)[vec * bytesPerVec];
        int realDim = d * kDimPerIter;

        MemT x[kDimPerIter];
#pragma unroll
        for (int i = 0; i < kDimPerIter; ++i) {
            x[i] = encodeHelper(v[i], realDim + i);
        }

        MemT out = 0;
#pragma unroll
        for (int i = 0; i < kDimPerIter; ++i) {
            out |= (x[i] << (i * 8));
        }

        p[d] = out;
    }

    inline __device__ void encodePartial(
            void* data,
            idx_t vec,
            int d,
            int remaining,
            float v[kDimPerIter]) const {
        if (DimMultiple > 1) {
            // should not be called
            assert(false);
        }

        // otherwise does not need implementing
    }

    //
    // interleaved code implementation
    //
    using EncodeT = uint8_t;
    static constexpr int kEncodeBits = 8;

    inline __device__ EncodeT encodeNew(int dim, float v) const {
        return encodeHelper(v, dim);
    }

    inline __device__ float decodeNew(int dim, EncodeT v) const {
        return decodeHelper(v, dim);
    }

    int bytesPerVec;

    // gmem pointers
    const float* vmin;
    const float* vdiff;

    // smem pointers (configured in the kernel)
    float* smemVmin;
    float* smemVdiff;
};

template <>
struct Codec<ScalarQuantizer::QuantizerType::QT_8bit_direct, 1> {
    /// How many dimensions per iteration we are handling for encoding or
    /// decoding
    static constexpr int kDimPerIter = 1;

    Codec(int vecBytes) : bytesPerVec(vecBytes) {}

    size_t getSmemSize(int dim) {
        return 0;
    }
    inline __device__ void initKernel(float* smem, int dim) {}

    inline __device__ void decode(void* data, idx_t vec, int d, float* out)
            const {
        uint8_t* p = &((uint8_t*)data)[vec * bytesPerVec];
        out[0] = (float)p[d];
    }

    inline __device__ float decodePartial(
            void* data,
            idx_t vec,
            int d,
            int subD) const {
        // doesn't need implementing (kDimPerIter == 1)
        return 0.0f;
    }

    inline __device__ void encode(
            void* data,
            idx_t vec,
            int d,
            float v[kDimPerIter]) const {
        uint8_t* p = &((uint8_t*)data)[vec * bytesPerVec];
        p[d] = (uint8_t)v[0];
    }

    inline __device__ void encodePartial(
            void* data,
            idx_t vec,
            int d,
            int remaining,
            float v[kDimPerIter]) const {
        // doesn't need implementing (kDimPerIter == 1)
    }

    //
    // interleaved code implementation
    //
    using EncodeT = uint8_t;
    static constexpr int kEncodeBits = 8;

    inline __device__ EncodeT encodeNew(int dim, float v) const {
        return (uint8_t)v;
    }

    inline __device__ float decodeNew(int dim, EncodeT v) const {
        return (float)v;
    }

    int bytesPerVec;
};

/////
//
// 6 bit encodings
//
/////

template <>
struct Codec<ScalarQuantizer::QuantizerType::QT_6bit, 1> {
    Codec(int vecBytes, float* min, float* diff)
            : bytesPerVec(vecBytes),
              vmin(min),
              vdiff(diff),
              smemVmin(nullptr),
              smemVdiff(nullptr) {}

    size_t getSmemSize(int dim) {
        return sizeof(float) * dim * 2;
    }

    // Initialize shared memory and local storage
    // It is up to the user to call a trailing syncthreads (after any other
    // initialization required has been done)
    inline __device__ void initKernel(float* smem, int dim) {
        smemVmin = smem;
        smemVdiff = smem + dim;

        for (auto i = threadIdx.x; i < dim; i += blockDim.x) {
            // We are performing vmin + vdiff * (v + 0.5) / (2^bits - 1)
            // This can be simplified to vmin' + vdiff' * v where:
            // vdiff' = vdiff / (2^bits - 1)
            // vmin' = vmin + 0.5 * vdiff'
            auto vd = vdiff[i] * (1.0f / 63.0f);
            smemVmin[i] = vmin[i] + 0.5f * vd;
            smemVdiff[i] = vd;
        }
    }

    inline __device__ float decodeHelper(uint8_t v, int realDim) const {
        return smemVmin[realDim] + (float)v * smemVdiff[realDim];
    }

    inline __device__ uint8_t encodeHelper(float v, int realDim) const {
        float x = (v - vmin[realDim]) / vdiff[realDim];
        x = fminf(1.0f, fmaxf(0.0f, x));
        return (uint8_t)(x * 63.0f);
    }

    //
    // interleaved code implementation
    //
    using EncodeT = uint8_t;
    static constexpr int kEncodeBits = 6;

    inline __device__ EncodeT encodeNew(int dim, float v) const {
        return encodeHelper(v, dim);
    }

    inline __device__ float decodeNew(int dim, EncodeT v) const {
        return decodeHelper(v, dim);
    }

    int bytesPerVec;

    // gmem pointers
    const float* vmin;
    const float* vdiff;

    // smem pointers
    float* smemVmin;
    float* smemVdiff;
};

/////
//
// 4 bit encodings
//
/////

// Uniform quantization across all dimensions
template <>
struct Codec<ScalarQuantizer::QuantizerType::QT_4bit_uniform, 1> {
    /// How many dimensions per iteration we are handling for encoding or
    /// decoding
    static constexpr int kDimPerIter = 2;

    Codec(int vecBytes, float min, float diff)
            : bytesPerVec(vecBytes), vmin(min), vdiff(diff) {}

    size_t getSmemSize(int dim) {
        return 0;
    }
    inline __device__ void initKernel(float* smem, int dim) {
        // We are performing vmin + vdiff * (v + 0.5) / (2^bits - 1)
        // This can be simplified to vmin' + vdiff' * v where:
        // vdiff' = vdiff / (2^bits - 1)
        // vmin' = vmin + 0.5 * vdiff'
        auto vd = vdiff * (1.0f / 15.0f);
        vmin = vmin + 0.5f * vd;
        vdiff = vd;
    }

    inline __device__ float decodeHelper(uint8_t v) const {
        return vmin + (float)v * vdiff;
    }

    inline __device__ void decode(void* data, idx_t vec, int d, float* out)
            const {
        uint8_t* p = &((uint8_t*)data)[vec * bytesPerVec];
        uint8_t pv = p[d];

        out[0] = decodeHelper(pv & 0xf);
        out[1] = decodeHelper(pv >> 4);
    }

    inline __device__ float decodePartial(
            void* data,
            idx_t vec,
            int d,
            int subD /* unused */) const {
        // We can only be called for a single input
        uint8_t* p = &((uint8_t*)data)[vec * bytesPerVec];
        uint8_t pv = p[d];

        return decodeHelper(pv & 0xf);
    }

    inline __device__ uint8_t encodeHelper(float v) const {
        float x = (v - vmin) / vdiff;
        x = fminf(1.0f, fmaxf(0.0f, x));
        return (uint8_t)(x * 15.0f);
    }

    inline __device__ void encode(
            void* data,
            idx_t vec,
            int d,
            float v[kDimPerIter]) const {
        uint8_t* p = &((uint8_t*)data)[vec * bytesPerVec];
        p[d] = encodeHelper(v[0]) | (encodeHelper(v[1]) << 4);
    }

    inline __device__ void encodePartial(
            void* data,
            idx_t vec,
            int d,
            int remaining, /* unused */
            float v[kDimPerIter]) const {
        // We can only be called for a single output
        uint8_t* p = &((uint8_t*)data)[vec * bytesPerVec];
        p[d] = encodeHelper(v[0]);
    }

    //
    // interleaved code implementation
    //
    using EncodeT = uint8_t;
    static constexpr int kEncodeBits = 4;

    inline __device__ EncodeT encodeNew(int dim, float v) const {
        return encodeHelper(v);
    }

    inline __device__ float decodeNew(int dim, EncodeT v) const {
        return decodeHelper(v);
    }

    int bytesPerVec;
    float vmin;
    float vdiff;
};

template <>
struct Codec<ScalarQuantizer::QuantizerType::QT_4bit, 1> {
    /// How many dimensions per iteration we are handling for encoding or
    /// decoding
    static constexpr int kDimPerIter = 2;

    Codec(int vecBytes, float* min, float* diff)
            : bytesPerVec(vecBytes),
              vmin(min),
              vdiff(diff),
              smemVmin(nullptr),
              smemVdiff(nullptr) {}

    size_t getSmemSize(int dim) {
        return sizeof(float) * dim * 2;
    }

    inline __device__ void initKernel(float* smem, int dim) {
        smemVmin = smem;
        smemVdiff = smem + dim;

        for (auto i = threadIdx.x; i < dim; i += blockDim.x) {
            // We are performing vmin + vdiff * (v + 0.5) / (2^bits - 1)
            // This can be simplified to vmin' + vdiff' * v where:
            // vdiff' = vdiff / (2^bits - 1)
            // vmin' = vmin + 0.5 * vdiff'
            auto vd = vdiff[i] / 15.0f;
            smemVmin[i] = vmin[i] + 0.5f * vd;
            smemVdiff[i] = vd;
        }

        __syncthreads();
    }

    inline __device__ float decodeHelper(uint8_t v, int realDim) const {
        return smemVmin[realDim] + (float)v * smemVdiff[realDim];
    }

    inline __device__ void decode(void* data, idx_t vec, int d, float* out)
            const {
        uint8_t* p = &((uint8_t*)data)[vec * bytesPerVec];
        uint8_t pv = p[d];
        int realDim = d * kDimPerIter;

        out[0] = decodeHelper(pv & 0xf, realDim);
        out[1] = decodeHelper(pv >> 4, realDim + 1);
    }

    inline __device__ float decodePartial(
            void* data,
            idx_t vec,
            int d,
            int subD /* unused */) const {
        // We can only be called for a single input
        uint8_t* p = &((uint8_t*)data)[vec * bytesPerVec];
        uint8_t pv = p[d];
        int realDim = d * kDimPerIter;

        return decodeHelper(pv & 0xf, realDim);
    }

    inline __device__ uint8_t encodeHelper(float v, int realDim) const {
        float x = (v - vmin[realDim]) / vdiff[realDim];
        x = fminf(1.0f, fmaxf(0.0f, x));
        return (uint8_t)(x * 15.0f);
    }

    inline __device__ void encode(
            void* data,
            idx_t vec,
            int d,
            float v[kDimPerIter]) const {
        uint8_t* p = &((uint8_t*)data)[vec * bytesPerVec];
        int realDim = d * kDimPerIter;
        p[d] = encodeHelper(v[0], realDim) |
                (encodeHelper(v[1], realDim + 1) << 4);
    }

    inline __device__ void encodePartial(
            void* data,
            idx_t vec,
            int d,
            int remaining, /* unused */
            float v[kDimPerIter]) const {
        // We can only be called for a single output
        uint8_t* p = &((uint8_t*)data)[vec * bytesPerVec];
        int realDim = d * kDimPerIter;

        p[d] = encodeHelper(v[0], realDim);
    }

    //
    // interleaved code implementation
    //
    using EncodeT = uint8_t;
    static constexpr int kEncodeBits = 4;

    inline __device__ EncodeT encodeNew(int dim, float v) const {
        return encodeHelper(v, dim);
    }

    inline __device__ float decodeNew(int dim, EncodeT v) const {
        return decodeHelper(v, dim);
    }

    int bytesPerVec;

    // gmem pointers
    const float* vmin;
    const float* vdiff;

    // smem pointers
    float* smemVmin;
    float* smemVdiff;
};

} // namespace gpu
} // namespace faiss
