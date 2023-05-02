/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <faiss/gpu/utils/Float16.cuh>

#ifndef __HALF2_TO_UI
// cuda_fp16.hpp doesn't export this
#define __HALF2_TO_UI(var) *(reinterpret_cast<unsigned int*>(&(var)))
#endif

//
// Templated wrappers to express load/store for different scalar and vector
// types, so kernels can have the same written form but can operate
// over half and float, and on vector types transparently
//

namespace faiss {
namespace gpu {

#ifdef USE_ROCM

template <typename T>
struct LoadStore {
    static inline __device__ T load(void* p) {
        return *((T*)p);
    }

    static inline __device__ void store(void* p, const T& v) {
        *((T*)p) = v;
    }
};

template <>
struct LoadStore<Half4> {
    static inline __device__ Half4 load(void* p) {
        Half4 out;
        // TODO
        // 64 bytes containing 4 half (float16) values, Half4 {half2 a, half2 b}
        const half* half_ptr = reinterpret_cast<const half*>(p);
        out.a.x = half_ptr[0];
        out.a.y = half_ptr[1];
        out.b.x = half_ptr[2];
        out.b.y = half_ptr[3];
        return out;
    }

    static inline __device__ void store(void* p, Half4& v) {
        // TODO
        half* half_ptr = reinterpret_cast<half*>(p);
        half_ptr[0] = v.a.x;
        half_ptr[1] = v.a.y;
        half_ptr[2] = v.b.x;
        half_ptr[3] = v.b.y;
    }
};

template <>
struct LoadStore<Half8> {
    static inline __device__ Half8 load(void* p) {
        Half8 out;
        // TODO
        // 128 bytes containing 8 half (float16) values, Half8 {Half4 a, Half4 b}
        // 1st Half4 out.a, 2nd Half4 out.b
        const half* half_ptr = reinterpret_cast<const half*>(p);
        out.b.a.x = half_ptr[4];
        out.b.a.y = half_ptr[5];
        out.b.b.x = half_ptr[6];
        out.b.b.y = half_ptr[7];
        return out;
    }

    static inline __device__ void store(void* p, Half8& v) {
        // TODO
	half* half_ptr = reinterpret_cast<half*>(p);
        half_ptr[0] = v.a.a.x;
        half_ptr[1] = v.a.a.y;
        half_ptr[2] = v.a.b.x;
        half_ptr[3] = v.a.b.y;

        half_ptr[4] = v.b.a.x;
        half_ptr[5] = v.b.a.y;
        half_ptr[6] = v.b.b.x;
        half_ptr[7] = v.b.b.y;
    }
};

#else // USE_ROCM

template <typename T>
struct LoadStore {
    static inline __device__ T load(void* p) {
        return *((T*)p);
    }

    static inline __device__ void store(void* p, const T& v) {
        *((T*)p) = v;
    }
};

template <>
struct LoadStore<Half4> {
    static inline __device__ Half4 load(void* p) {
        Half4 out;
#if CUDA_VERSION >= 9000
        asm("ld.global.v2.u32 {%0, %1}, [%2];"
            : "=r"(__HALF2_TO_UI(out.a)), "=r"(__HALF2_TO_UI(out.b))
            : "l"(p));
#else
        asm("ld.global.v2.u32 {%0, %1}, [%2];"
            : "=r"(out.a.x), "=r"(out.b.x)
            : "l"(p));
#endif
        return out;
    }

    static inline __device__ void store(void* p, Half4& v) {
#if CUDA_VERSION >= 9000
        asm("st.v2.u32 [%0], {%1, %2};"
            :
            : "l"(p), "r"(__HALF2_TO_UI(v.a)), "r"(__HALF2_TO_UI(v.b)));
#else
        asm("st.v2.u32 [%0], {%1, %2};" : : "l"(p), "r"(v.a.x), "r"(v.b.x));
#endif
    }
};

template <>
struct LoadStore<Half8> {
    static inline __device__ Half8 load(void* p) {
        Half8 out;
#if CUDA_VERSION >= 9000
        asm("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
            : "=r"(__HALF2_TO_UI(out.a.a)),
              "=r"(__HALF2_TO_UI(out.a.b)),
              "=r"(__HALF2_TO_UI(out.b.a)),
              "=r"(__HALF2_TO_UI(out.b.b))
            : "l"(p));
#else
        asm("ld.global.v4.u32 {%0, %1, %2, %3}, [%4];"
            : "=r"(out.a.a.x), "=r"(out.a.b.x), "=r"(out.b.a.x), "=r"(out.b.b.x)
            : "l"(p));
#endif
        return out;
    }

    static inline __device__ void store(void* p, Half8& v) {
#if CUDA_VERSION >= 9000
        asm("st.v4.u32 [%0], {%1, %2, %3, %4};"
            :
            : "l"(p),
              "r"(__HALF2_TO_UI(v.a.a)),
              "r"(__HALF2_TO_UI(v.a.b)),
              "r"(__HALF2_TO_UI(v.b.a)),
              "r"(__HALF2_TO_UI(v.b.b)));
#else
        asm("st.v4.u32 [%0], {%1, %2, %3, %4};"
            :
            : "l"(p), "r"(v.a.a.x), "r"(v.a.b.x), "r"(v.b.a.x), "r"(v.b.b.x));
#endif
    }
};

#endif // USE_ROCM

} // namespace gpu
} // namespace faiss
