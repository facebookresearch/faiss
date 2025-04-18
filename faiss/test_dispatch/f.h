
#include <cassert>
#include <cstdio>

enum SIMDLevel {
    NONE,
    AVX2,
    AVX512F,
    ARM_NEON,
    ARM_SVE,
};

extern SIMDLevel simd_level;

#ifdef __ARM_FEATURE_SVE
#define DISPATCH_SIMDLevel_ARM_SVE(f, ...) \
    case SIMDLevel::ARM_SVE:               \
        return f<SIMDLevel::ARM_SVE>(__VA_ARGS__)
#else
#define DISPATCH_SIMDLevel_ARM_SVE(f, ...)
#endif

// #ifdef __AVX2__
#if 1
#define DISPATCH_SIMDLevel_AVX2(f, ...) \
    case SIMDLevel::AVX2:               \
        return f<SIMDLevel::AVX2>(__VA_ARGS__)
#else
#define DISPATCH_SIMDLevel_AVX2(f, ...)
#endif

#ifdef __AVX512F__
#define DISPATCH_SIMDLevel_AVX512F(f, ...) \
    case SIMDLevel::AVX512F:               \
        return f<SIMDLevel::AVX512F>(__VA_ARGS__)
#else
#define DISPATCH_SIMDLevel_AVX512F(f, ...)
#endif

#define DISPATCH_SIMDLevel(f, ...)                      \
    switch (simd_level) {                               \
        case SIMDLevel::NONE:                           \
            return f<SIMDLevel::NONE>(__VA_ARGS__);     \
            DISPATCH_SIMDLevel_ARM_SVE(f, __VA_ARGS__); \
            DISPATCH_SIMDLevel_AVX2(f, __VA_ARGS__);    \
            DISPATCH_SIMDLevel_AVX512F(f, __VA_ARGS__); \
        default:                                        \
            assert(!"invlalid SIMD level");             \
    }

template <SIMDLevel>
float fvec_norm_L2sqr(const float* x, size_t d);

float fvec_norm_L2sqr(const float* x, size_t d);

template <typename T, int i>
struct FF {
    static void func(T* x) {
        // default implementation
    }
};

template <typename T>
struct FF<T, 1> {
    static void func(T* x) {
        printf("sizeof T = %d\n", sizeof(T));
        // specialized implementation for i = 1
    }
};

template <typename T>
struct FF<T, 1> {
    static void func(T* x) {
        printf("sizeof T = %d\n", sizeof(T));
        // specialized implementation for i = 1
    }
};
