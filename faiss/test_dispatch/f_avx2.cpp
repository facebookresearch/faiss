
#include "f.h"

#define FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN \
    _Pragma("float_control(precise, off, push)")
#define FAISS_PRAGMA_IMPRECISE_FUNCTION_END _Pragma("float_control(pop)")

#define FAISS_PRAGMA_IMPRECISE_LOOP \
    _Pragma("clang loop vectorize(enable) interleave(enable)")

FAISS_PRAGMA_IMPRECISE_FUNCTION_BEGIN
template <>
float fvec_norm_L2sqr<SIMDLevel::AVX512F>(const float* x, size_t d) {
    // the double in the _ref is suspected to be a typo. Some of the manual
    // implementations this replaces used float.
    float res = 0;
    FAISS_PRAGMA_IMPRECISE_LOOP
    for (size_t i = 0; i != d; ++i) {
        res += x[i] * x[i];
    }

    return res;
}
FAISS_PRAGMA_IMPRECISE_FUNCTION_END
