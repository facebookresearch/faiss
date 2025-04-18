
#include <cstdlib>
#include "f.h"

// fw declaration
template <int i>
void func();

int main() {
    func<1>();
    func<2>();

    float x[16];
    for (int i = 0; i < 16; i++) {
        x[i] = i + 1;
    }
    float norm = fvec_norm_L2sqr(x, 16);
    printf("norm=%g\n", norm);

    simd_level = SIMDLevel::AVX2;
    norm = fvec_norm_L2sqr(x, 16);
    printf("norm=%g\n", norm);

    /*
    simd_level = SIMDLevel::AVX512F;
    norm = fvec_norm_L2sqr(x, 16);
    printf("norm=%g\n", norm);
*/
    char* env_var = getenv("FAISS_SIMD_LEVEL");

    printf("env_var=%s\n", env_var);

    int i;
    FF<int, 1>::func(&i);

    return 0;
}
