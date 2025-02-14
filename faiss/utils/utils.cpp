/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/Index.h>
#include <faiss/utils/utils.h>

#include <cassert>
#include <cmath>
#include <cstdio>
#include <cstring>

#include <sys/types.h>

#ifdef _MSC_VER
#define NOMINMAX
#include <windows.h>
#undef NOMINMAX
#else
#include <sys/time.h>
#include <unistd.h>
#endif // !_MSC_VER

#include <omp.h>

#include <algorithm>
#include <set>
#include <type_traits>
#include <vector>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/platform_macros.h>
#include <faiss/utils/random.h>

#ifndef FINTEGER
#define FINTEGER long
#endif

extern "C" {

/* declare BLAS functions, see http://www.netlib.org/clapack/cblas/ */

int sgemm_(
        const char* transa,
        const char* transb,
        FINTEGER* m,
        FINTEGER* n,
        FINTEGER* k,
        const float* alpha,
        const float* a,
        FINTEGER* lda,
        const float* b,
        FINTEGER* ldb,
        float* beta,
        float* c,
        FINTEGER* ldc);

/* Lapack functions, see http://www.netlib.org/clapack/old/single/sgeqrf.c */

int sgeqrf_(
        FINTEGER* m,
        FINTEGER* n,
        float* a,
        FINTEGER* lda,
        float* tau,
        float* work,
        FINTEGER* lwork,
        FINTEGER* info);

int sorgqr_(
        FINTEGER* m,
        FINTEGER* n,
        FINTEGER* k,
        float* a,
        FINTEGER* lda,
        float* tau,
        float* work,
        FINTEGER* lwork,
        FINTEGER* info);

int sgemv_(
        const char* trans,
        FINTEGER* m,
        FINTEGER* n,
        float* alpha,
        const float* a,
        FINTEGER* lda,
        const float* x,
        FINTEGER* incx,
        float* beta,
        float* y,
        FINTEGER* incy);
}

/**************************************************
 * Get some stats about the system
 **************************************************/

namespace faiss {

// this will be set at load time from GPU Faiss
std::string gpu_compile_options;

std::string get_compile_options() {
    std::string options;

    // this flag is set by GCC and Clang
#ifdef __OPTIMIZE__
    options += "OPTIMIZE ";
#endif

#ifdef __AVX512F__
    options += "AVX512 ";
#elif defined(__AVX2__)
    options += "AVX2 ";
#elif defined(__ARM_FEATURE_SVE)
    options += "SVE NEON ";
#elif defined(__aarch64__)
    options += "NEON ";
#else
    options += "GENERIC ";
#endif

    options += gpu_compile_options;

    return options;
}

std::string get_version() {
    return VERSION_STRING;
}

#ifdef _MSC_VER
double getmillisecs() {
    LARGE_INTEGER ts;
    LARGE_INTEGER freq;
    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&ts);

    return (ts.QuadPart * 1e3) / freq.QuadPart;
}
#else  // _MSC_VER
double getmillisecs() {
    struct timeval tv;
    gettimeofday(&tv, nullptr);
    return tv.tv_sec * 1e3 + tv.tv_usec * 1e-3;
}
#endif // _MSC_VER

uint64_t get_cycles() {
#ifdef __x86_64__
    uint32_t high, low;
    asm volatile("rdtsc \n\t" : "=a"(low), "=d"(high));
    return ((uint64_t)high << 32) | (low);
#else
    return 0;
#endif
}

#ifdef __linux__

size_t get_mem_usage_kb() {
    int pid = getpid();
    char fname[256];
    snprintf(fname, 256, "/proc/%d/status", pid);
    FILE* f = fopen(fname, "r");
    FAISS_THROW_IF_NOT_MSG(f, "cannot open proc status file");
    size_t sz = 0;
    for (;;) {
        char buf[256];
        if (!fgets(buf, 256, f))
            break;
        if (sscanf(buf, "VmRSS: %ld kB", &sz) == 1)
            break;
    }
    fclose(f);
    return sz;
}

#else

size_t get_mem_usage_kb() {
    fprintf(stderr,
            "WARN: get_mem_usage_kb not implemented on current architecture\n");
    return 0;
}

#endif

void reflection(
        const float* __restrict u,
        float* __restrict x,
        size_t n,
        size_t d,
        size_t nu) {
    size_t i, j, l;
    for (i = 0; i < n; i++) {
        const float* up = u;
        for (l = 0; l < nu; l++) {
            float ip1 = 0, ip2 = 0;

            for (j = 0; j < d; j += 2) {
                ip1 += up[j] * x[j];
                ip2 += up[j + 1] * x[j + 1];
            }
            float ip = 2 * (ip1 + ip2);

            for (j = 0; j < d; j++)
                x[j] -= ip * up[j];
            up += d;
        }
        x += d;
    }
}

/* Reference implementation (slower) */
void reflection_ref(const float* u, float* x, size_t n, size_t d, size_t nu) {
    size_t i, j, l;
    for (i = 0; i < n; i++) {
        const float* up = u;
        for (l = 0; l < nu; l++) {
            double ip = 0;

            for (j = 0; j < d; j++)
                ip += up[j] * x[j];
            ip *= 2;

            for (j = 0; j < d; j++)
                x[j] -= ip * up[j];

            up += d;
        }
        x += d;
    }
}

/***************************************************************************
 * Some matrix manipulation functions
 ***************************************************************************/

void matrix_qr(int m, int n, float* a) {
    FAISS_THROW_IF_NOT(m >= n);
    FINTEGER mi = m, ni = n, ki = mi < ni ? mi : ni;
    std::vector<float> tau(ki);
    FINTEGER lwork = -1, info;
    float work_size;

    sgeqrf_(&mi, &ni, a, &mi, tau.data(), &work_size, &lwork, &info);
    lwork = size_t(work_size);
    std::vector<float> work(lwork);

    sgeqrf_(&mi, &ni, a, &mi, tau.data(), work.data(), &lwork, &info);

    sorgqr_(&mi, &ni, &ki, a, &mi, tau.data(), work.data(), &lwork, &info);
}

/***************************************************************************
 * Result list routines
 ***************************************************************************/

void ranklist_handle_ties(int k, int64_t* idx, const float* dis) {
    float prev_dis = -1e38;
    int prev_i = -1;
    for (int i = 0; i < k; i++) {
        if (dis[i] != prev_dis) {
            if (i > prev_i + 1) {
                // sort between prev_i and i - 1
                std::sort(idx + prev_i, idx + i);
            }
            prev_i = i;
            prev_dis = dis[i];
        }
    }
}

size_t merge_result_table_with(
        size_t n,
        size_t k,
        int64_t* I0,
        float* D0,
        const int64_t* I1,
        const float* D1,
        bool keep_min,
        int64_t translation) {
    size_t n1 = 0;

#pragma omp parallel reduction(+ : n1)
    {
        std::vector<int64_t> tmpI(k);
        std::vector<float> tmpD(k);

#pragma omp for
        for (int64_t i = 0; i < n; i++) {
            int64_t* lI0 = I0 + i * k;
            float* lD0 = D0 + i * k;
            const int64_t* lI1 = I1 + i * k;
            const float* lD1 = D1 + i * k;
            size_t r0 = 0;
            size_t r1 = 0;

            if (keep_min) {
                for (size_t j = 0; j < k; j++) {
                    if (lI0[r0] >= 0 && lD0[r0] < lD1[r1]) {
                        tmpD[j] = lD0[r0];
                        tmpI[j] = lI0[r0];
                        r0++;
                    } else if (lD1[r1] >= 0) {
                        tmpD[j] = lD1[r1];
                        tmpI[j] = lI1[r1] + translation;
                        r1++;
                    } else { // both are NaNs
                        tmpD[j] = NAN;
                        tmpI[j] = -1;
                    }
                }
            } else {
                for (size_t j = 0; j < k; j++) {
                    if (lI0[r0] >= 0 && lD0[r0] > lD1[r1]) {
                        tmpD[j] = lD0[r0];
                        tmpI[j] = lI0[r0];
                        r0++;
                    } else if (lD1[r1] >= 0) {
                        tmpD[j] = lD1[r1];
                        tmpI[j] = lI1[r1] + translation;
                        r1++;
                    } else { // both are NaNs
                        tmpD[j] = NAN;
                        tmpI[j] = -1;
                    }
                }
            }
            n1 += r1;
            memcpy(lD0, tmpD.data(), sizeof(lD0[0]) * k);
            memcpy(lI0, tmpI.data(), sizeof(lI0[0]) * k);
        }
    }

    return n1;
}

size_t ranklist_intersection_size(
        size_t k1,
        const int64_t* v1,
        size_t k2,
        const int64_t* v2_in) {
    if (k2 > k1)
        return ranklist_intersection_size(k2, v2_in, k1, v1);
    int64_t* v2 = new int64_t[k2];
    memcpy(v2, v2_in, sizeof(int64_t) * k2);
    std::sort(v2, v2 + k2);
    { // de-dup v2
        int64_t prev = -1;
        size_t wp = 0;
        for (size_t i = 0; i < k2; i++) {
            if (v2[i] != prev) {
                v2[wp++] = prev = v2[i];
            }
        }
        k2 = wp;
    }
    const int64_t seen_flag = int64_t{1} << 60;
    size_t count = 0;
    for (size_t i = 0; i < k1; i++) {
        int64_t q = v1[i];
        size_t i0 = 0, i1 = k2;
        while (i0 + 1 < i1) {
            size_t imed = (i1 + i0) / 2;
            int64_t piv = v2[imed] & ~seen_flag;
            if (piv <= q)
                i0 = imed;
            else
                i1 = imed;
        }
        if (v2[i0] == q) {
            count++;
            v2[i0] |= seen_flag;
        }
    }
    delete[] v2;

    return count;
}

double imbalance_factor(int k, const int* hist) {
    double tot = 0, uf = 0;

    for (int i = 0; i < k; i++) {
        tot += hist[i];
        uf += hist[i] * (double)hist[i];
    }
    uf = uf * k / (tot * tot);

    return uf;
}

double imbalance_factor(int n, int k, const int64_t* assign) {
    std::vector<int> hist(k, 0);
    for (int i = 0; i < n; i++) {
        hist[assign[i]]++;
    }

    return imbalance_factor(k, hist.data());
}

int ivec_hist(size_t n, const int* v, int vmax, int* hist) {
    memset(hist, 0, sizeof(hist[0]) * vmax);
    int nout = 0;
    while (n--) {
        if (v[n] < 0 || v[n] >= vmax)
            nout++;
        else
            hist[v[n]]++;
    }
    return nout;
}

void bincode_hist(size_t n, size_t nbits, const uint8_t* codes, int* hist) {
    FAISS_THROW_IF_NOT(nbits % 8 == 0);
    size_t d = nbits / 8;
    std::vector<int> accu(d * 256);
    const uint8_t* c = codes;
    for (size_t i = 0; i < n; i++)
        for (int j = 0; j < d; j++)
            accu[j * 256 + *c++]++;
    memset(hist, 0, sizeof(*hist) * nbits);
    for (int i = 0; i < d; i++) {
        const int* ai = accu.data() + i * 256;
        int* hi = hist + i * 8;
        for (int j = 0; j < 256; j++)
            for (int k = 0; k < 8; k++)
                if ((j >> k) & 1)
                    hi[k] += ai[j];
    }
}

uint64_t ivec_checksum(size_t n, const int32_t* assigned) {
    const uint32_t* a = reinterpret_cast<const uint32_t*>(assigned);
    uint64_t cs = 112909;
    while (n--) {
        cs = cs * 65713 + a[n] * 1686049;
    }
    return cs;
}

uint64_t bvec_checksum(size_t n, const uint8_t* a) {
    uint64_t cs = ivec_checksum(n / 4, (const int32_t*)a);
    for (size_t i = n / 4 * 4; i < n; i++) {
        cs = cs * 65713 + a[n] * 1686049;
    }
    return cs;
}

void bvecs_checksum(size_t n, size_t d, const uint8_t* a, uint64_t* cs) {
    // MSVC can't accept unsigned index for #pragma omp parallel for
    // so below codes only accept n <= std::numeric_limits<ssize_t>::max()
    using ssize_t = std::make_signed<std::size_t>::type;
    const ssize_t size = n;
#pragma omp parallel for if (size > 1000)
    for (ssize_t i_ = 0; i_ < size; i_++) {
        const auto i = static_cast<std::size_t>(i_);
        cs[i] = bvec_checksum(d, a + i * d);
    }
}

const float* fvecs_maybe_subsample(
        size_t d,
        size_t* n,
        size_t nmax,
        const float* x,
        bool verbose,
        int64_t seed) {
    if (*n <= nmax)
        return x; // nothing to do

    size_t n2 = nmax;
    if (verbose) {
        printf("  Input training set too big (max size is %zd), sampling "
               "%zd / %zd vectors\n",
               nmax,
               n2,
               *n);
    }
    std::vector<int> subset(*n);
    rand_perm(subset.data(), *n, seed);
    float* x_subset = new float[n2 * d];
    for (int64_t i = 0; i < n2; i++)
        memcpy(&x_subset[i * d], &x[subset[i] * size_t(d)], sizeof(x[0]) * d);
    *n = n2;
    return x_subset;
}

void binary_to_real(size_t d, const uint8_t* x_in, float* x_out) {
    for (size_t i = 0; i < d; ++i) {
        x_out[i] = 2 * ((x_in[i >> 3] >> (i & 7)) & 1) - 1;
    }
}

void real_to_binary(size_t d, const float* x_in, uint8_t* x_out) {
    for (size_t i = 0; i < d / 8; ++i) {
        uint8_t b = 0;
        for (int j = 0; j < 8; ++j) {
            if (x_in[8 * i + j] > 0) {
                b |= (1 << j);
            }
        }
        x_out[i] = b;
    }
}

// from Python's stringobject.c
uint64_t hash_bytes(const uint8_t* bytes, int64_t n) {
    const uint8_t* p = bytes;
    uint64_t x = (uint64_t)(*p) << 7;
    int64_t len = n;
    while (--len >= 0) {
        x = (1000003 * x) ^ *p++;
    }
    x ^= n;
    return x;
}

bool check_openmp() {
    omp_set_num_threads(10);

    if (omp_get_max_threads() != 10) {
        return false;
    }

    std::vector<int> nt_per_thread(10);
    size_t sum = 0;
    bool in_parallel = true;
#pragma omp parallel reduction(+ : sum)
    {
        if (!omp_in_parallel()) {
            in_parallel = false;
        }

        int nt = omp_get_num_threads();
        int rank = omp_get_thread_num();

        nt_per_thread[rank] = nt;
#pragma omp for
        for (int i = 0; i < 1000 * 1000 * 10; i++) {
            sum += i;
        }
    }

    if (!in_parallel) {
        return false;
    }
    if (nt_per_thread[0] != 10) {
        return false;
    }
    if (sum == 0) {
        return false;
    }

    return true;
}

namespace {

template <typename T>
int64_t count_lt(int64_t n, const T* row, T threshold) {
    for (int64_t i = 0; i < n; i++) {
        if (!(row[i] < threshold)) {
            return i;
        }
    }
    return n;
}

template <typename T>
int64_t count_gt(int64_t n, const T* row, T threshold) {
    for (int64_t i = 0; i < n; i++) {
        if (!(row[i] > threshold)) {
            return i;
        }
    }
    return n;
}

} // namespace

template <typename T>
void CombinerRangeKNN<T>::compute_sizes(int64_t* L_res_init) {
    this->L_res = L_res_init;
    L_res_init[0] = 0;
    int64_t j = 0;
    for (int64_t i = 0; i < nq; i++) {
        int64_t n_in;
        if (!mask || !mask[i]) {
            const T* row = D + i * k;
            n_in = keep_max ? count_gt(k, row, r2) : count_lt(k, row, r2);
        } else {
            n_in = lim_remain[j + 1] - lim_remain[j];
            j++;
        }
        L_res_init[i + 1] = n_in; // L_res_init[i] + n_in;
    }
    // cumsum
    for (int64_t i = 0; i < nq; i++) {
        L_res_init[i + 1] += L_res_init[i];
    }
}

template <typename T>
void CombinerRangeKNN<T>::write_result(T* D_res, int64_t* I_res) {
    FAISS_THROW_IF_NOT(L_res);
    int64_t j = 0;
    for (int64_t i = 0; i < nq; i++) {
        int64_t n_in = L_res[i + 1] - L_res[i];
        T* D_row = D_res + L_res[i];
        int64_t* I_row = I_res + L_res[i];
        if (!mask || !mask[i]) {
            memcpy(D_row, D + i * k, n_in * sizeof(*D_row));
            memcpy(I_row, I + i * k, n_in * sizeof(*I_row));
        } else {
            memcpy(D_row, D_remain + lim_remain[j], n_in * sizeof(*D_row));
            memcpy(I_row, I_remain + lim_remain[j], n_in * sizeof(*I_row));
            j++;
        }
    }
}

// explicit template instantiations
template struct CombinerRangeKNN<float>;
template struct CombinerRangeKNN<int16_t>;

void CodeSet::insert(size_t n, const uint8_t* codes, bool* inserted) {
    for (size_t i = 0; i < n; i++) {
        auto res = s.insert(
                std::vector<uint8_t>(codes + i * d, codes + i * d + d));
        inserted[i] = res.second;
    }
}

} // namespace faiss
