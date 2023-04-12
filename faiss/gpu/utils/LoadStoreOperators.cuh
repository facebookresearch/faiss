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
        // const half* half_ptr = reinterpret_cast<const half*>(p);
        out.a.x = p[0];
        out.a.y = p[1];
        out.b.x = p[2];
        out.b.y = p[3];
        return out;
    }

    static inline __device__ void store(void* p, Half4& v) {
        // TODO
        p[0] = v.a.x;
        p[1] = v.a.y;
        p[2] = v.b.x;
        p[3] = v.b.y;
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
        // TODO
        // 128 bytes containing 8 half (float16) values, Half8 {Half4 a, Half4 b}
        // 1st Half4 out.a, 2nd Half4 out.b
        out.a.a.x = p[0];
        out.a.a.y = p[1];
        out.a.b.x = p[2];
        out.a.b.y = p[3];

        out.b.a.x = p[4];
        out.b.a.y = p[5];
        out.b.b.x = p[6];
        out.b.b.y = p[7];
        return out;
    }

    static inline __device__ void store(void* p, Half8& v) {
        // TODO
        p[0] = v.a.a.x;
        p[1] = v.a.a.y;
        p[2] = v.a.b.x;
        p[3] = v.a.b.y;

        p[4] = v.b.a.x;
        p[5] = v.b.a.y;
        p[6] = v.b.b.x;
        p[7] = v.b.b.y;
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
