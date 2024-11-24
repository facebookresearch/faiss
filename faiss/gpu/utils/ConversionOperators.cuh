/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/MetricType.h>
#include <faiss/gpu/utils/DeviceTensor.cuh>
#include <faiss/gpu/utils/Float16.cuh>

#include <cuda.h>
#include <thrust/execution_policy.h>
#include <thrust/transform.h>

namespace faiss {
namespace gpu {

//
// Conversion utilities
//

template <typename T>
struct ConvertTo {
    template <typename U>
    static inline __device__ T to(U v) {
        return T(v);
    }
};

template <>
struct ConvertTo<float> {
    static inline __device__ float to(float v) {
        return v;
    }
    static inline __device__ float to(half v) {
        return __half2float(v);
    }

#ifndef USE_AMD_ROCM
    static inline __device__ float to(__nv_bfloat16 v) {
        return __bfloat162float(v);
    }
#endif // !USE_AMD_ROCM
};

template <>
struct ConvertTo<float2> {
    static inline __device__ float2 to(float2 v) {
        return v;
    }
    static inline __device__ float2 to(half2 v) {
        return __half22float2(v);
    }
};

template <>
struct ConvertTo<float4> {
    static inline __device__ float4 to(float4 v) {
        return v;
    }
    static inline __device__ float4 to(Half4 v) {
        return half4ToFloat4(v);
    }
};

template <>
struct ConvertTo<half> {
    static inline __device__ half to(float v) {
        return __float2half(v);
    }
    static inline __device__ half to(half v) {
        return v;
    }
};

template <>
struct ConvertTo<half2> {
    static inline __device__ half2 to(float2 v) {
        return __float22half2_rn(v);
    }
    static inline __device__ half2 to(half2 v) {
        return v;
    }
};

template <>
struct ConvertTo<Half4> {
    static inline __device__ Half4 to(float4 v) {
        return float4ToHalf4(v);
    }
    static inline __device__ Half4 to(Half4 v) {
        return v;
    }
};

// no bf16 support for AMD
#ifndef USE_AMD_ROCM

template <>
struct ConvertTo<__nv_bfloat16> {
    static inline __device__ __nv_bfloat16 to(float v) {
        return __float2bfloat16(v);
    }
    static inline __device__ __nv_bfloat16 to(half v) {
        return __float2bfloat16(__half2float(v));
    }
    static inline __device__ __nv_bfloat16 to(__nv_bfloat16 v) {
        return v;
    }
};

#endif // USE_AMD_ROCM

template <typename From, typename To>
struct Convert {
    inline __device__ To operator()(From v) const {
        return ConvertTo<To>::to(v);
    }
};

// Tensor conversion
template <typename From, typename To>
void runConvert(const From* in, To* out, size_t num, cudaStream_t stream) {
    thrust::transform(
            thrust::cuda::par.on(stream),
            in,
            in + num,
            out,
            Convert<From, To>());
}

template <typename From, typename To, int Dim>
void convertTensor(
        cudaStream_t stream,
        Tensor<From, Dim, true>& in,
        Tensor<To, Dim, true>& out) {
    FAISS_ASSERT(in.numElements() == out.numElements());

    runConvert<From, To>(in.data(), out.data(), in.numElements(), stream);
}

template <typename From, typename To, int Dim>
DeviceTensor<To, Dim, true> convertTensorTemporary(
        GpuResources* res,
        cudaStream_t stream,
        Tensor<From, Dim, true>& in) {
    FAISS_ASSERT(res);
    DeviceTensor<To, Dim, true> out(
            res, makeTempAlloc(AllocType::Other, stream), in.sizes());

    convertTensor(stream, in, out);
    return out;
}

template <typename From, typename To, int Dim>
DeviceTensor<To, Dim, true> convertTensorNonTemporary(
        GpuResources* res,
        cudaStream_t stream,
        Tensor<From, Dim, true>& in) {
    FAISS_ASSERT(res);
    DeviceTensor<To, Dim, true> out(
            res, makeDevAlloc(AllocType::Other, stream), in.sizes());

    convertTensor(stream, in, out);
    return out;
}

} // namespace gpu
} // namespace faiss
