#include <faiss/utils/distances_fused/distances_fused.h>

#include <faiss/impl/platform_macros.h>

#include <faiss/utils/distances_fused/avx512.h>
#include <faiss/utils/distances_fused/simdlib_based.h>

namespace faiss {

bool exhaustive_L2sqr_fused_cmax(
        const float* x,
        const float* y,
        size_t d,
        size_t nx,
        size_t ny,
        SingleBestResultHandler<CMax<float, int64_t>>& res,
        const float* y_norms) {
    if (nx == 0 || ny == 0) {
        // nothing to do
        return true;
    }

#ifdef __AVX512__
    // avx512 kernel
    return exhaustive_L2sqr_fused_cmax_AVX512(x, y, d, nx, ny, res, y_norms);
#elif defined(__AVX2__) || defined(__aarch64__)
    // avx2 or arm neon kernel
    return exhaustive_L2sqr_fused_cmax_simdlib(x, y, d, nx, ny, res, y_norms);
#else
    // not supported, please use a general-purpose kernel
    return false;
#endif
}

} // namespace faiss
