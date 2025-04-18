

#include "f.h"

SIMDLevel simd_level = SIMDLevel::NONE;

template <>
float fvec_norm_L2sqr<SIMDLevel::NONE>(const float* x, size_t d) {
    // the double in the _ref is suspected to be a typo. Some of the manual
    // implementations this replaces used float.
    float res = 0;
    for (size_t i = 0; i != d; ++i) {
        res += x[i] * x[i];
    }
    return res;
}

float fvec_norm_L2sqr(const float* x, size_t d) {
    DISPATCH_SIMDLevel(fvec_norm_L2sqr, x, d);
}
