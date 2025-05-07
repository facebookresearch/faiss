/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/utils/simd_levels.h>

#include <faiss/impl/FaissAssert.h>
#include <cstdlib>
#include <cstring>

namespace faiss {

SIMDLevel SIMDConfig::level = SIMDLevel::NONE;

const char* SIMDConfig::level_names[] =
        {"NONE", "AVX2", "AVX512F", "ARM_NEON", "ARM_SVE", "PPC"};

// it is there to make sure the constructor runs
static SIMDConfig dummy_config;

SIMDConfig::SIMDConfig() {
    char* env_var = getenv("FAISS_SIMD_LEVEL");
    if (env_var) {
        int i;
        for (i = 0; i <= sizeof(level_names); i++) {
            if (strcmp(env_var, level_names[i]) == 0) {
                level = (SIMDLevel)i;
                break;
            }
        }
        FAISS_THROW_IF_NOT_FMT(
                i != sizeof(level_names),
                "FAISS_SIMD_LEVEL %s unknown",
                env_var);
        return;
    }

#ifdef __x86_64__
    {
        unsigned int eax, ebx, ecx, edx;
        asm volatile("cpuid"
                     : "=a"(eax), "=b"(ebx), "=c"(ecx), "=d"(edx)
                     : "a"(0));

#ifdef COMPILE_SIMD_AVX512F
        if (ebx & (1 << 16)) {
            level = SIMDLevel::AVX512F;
        } else
#endif

#ifdef COMPILE_SIMD_AVX2
                if (ecx & 32) {
            level = SIMDLevel::AVX2;
        } else
#endif
            level = SIMDLevel::NONE;
    }
#endif
}

void SIMDConfig::set_level(SIMDLevel l) {
    level = l;
    // this could be used to set function pointers in the future
}

SIMDLevel SIMDConfig::get_level() {
    return level;
}

std::string SIMDConfig::get_level_name() {
    return std::string(level_names[int(level)]);
}

} // namespace faiss
