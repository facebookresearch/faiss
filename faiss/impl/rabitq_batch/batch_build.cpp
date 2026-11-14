/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/rabitq_batch/batch_build.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/HNSW.h>
#include <faiss/impl/RaBitQCodePairIP.h>
#include <faiss/utils/simd_levels.h>

#if FAISS_RABITQ_BATCH_BUILD
#include <cpuid.h>
#ifdef __linux__
#include <sys/syscall.h>
#include <unistd.h>
#endif
#endif

namespace faiss {
namespace rabitq_batch {

#if FAISS_RABITQ_BATCH_BUILD

namespace {

bool cpu_vnni() {
    __builtin_cpu_init();
    return __builtin_cpu_supports("avx512f") &&
            __builtin_cpu_supports("avx512bw") &&
            __builtin_cpu_supports("avx512vbmi") &&
            __builtin_cpu_supports("avx512vnni");
}

/// CPUID leaf 7 subleaf 0, EDX bits 24 and 25: AMX-TILE and AMX-INT8. On
/// Linux, the process must also be granted the tile data state.
bool cpu_amx() {
    unsigned a, b, c, d;
    if (!__get_cpuid_count(7, 0, &a, &b, &c, &d)) {
        return false;
    }
    if (!((d >> 24) & 1) || !((d >> 25) & 1)) {
        return false;
    }
#ifdef __linux__
    return syscall(SYS_arch_prctl,
                   0x1023 /* ARCH_REQ_XCOMP_PERM */,
                   18 /* XFEATURE_XTILEDATA */) == 0;
#else
    return false;
#endif
}

struct Support {
    bool vnni, amx;
};

const Support& support() {
    static const Support s = [] {
        const bool vnni = cpu_vnni();
        return Support{vnni, vnni && cpu_amx()};
    }();
    return s;
}

} // namespace

Backend select_backend(bool allow_amx) {
    if (SIMDConfig::get_level() != SIMDLevel::AVX512_SPR) {
        return Backend::none;
    }
    const Support& s = support();
    if (allow_amx && s.amx) {
        return Backend::amx;
    }
    return s.vnni ? Backend::vnni : Backend::none;
}

#else

Backend select_backend(bool) {
    return Backend::none;
}

#endif

const char* backend_name(Backend backend) {
    return backend == Backend::amx     ? "amx"
            : backend == Backend::vnni ? "vnni"
                                       : "none";
}

void build(
        HNSW& h,
        const uint8_t* codes,
        size_t n,
        size_t d,
        size_t code_size,
        double center_norm_sq,
        size_t k,
        Backend backend) {
    FAISS_THROW_IF_NOT(backend != Backend::none);
    FAISS_THROW_IF_NOT(
            d > 0 && d <= max_d && code_size == rabitq_ip::code_size_8bit(d));
    FAISS_THROW_IF_NOT(n > 0 && k > 0 && h.levels.size() == n);
#if FAISS_RABITQ_BATCH_BUILD
    build_avx512_spr(
            h,
            codes,
            n,
            d,
            code_size,
            center_norm_sq,
            k,
            backend == Backend::vnni);
#else
    (void)codes;
    (void)center_norm_sq;
    FAISS_THROW_MSG("RaBitQ batch build is not compiled in this build");
#endif
}

} // namespace rabitq_batch
} // namespace faiss
