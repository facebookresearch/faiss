/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/impl/ScalarQuantizer.h>

#include <algorithm>
#include <cstdio>

#include <faiss/impl/platform_macros.h>
#include <omp.h>

#ifdef __SSE__
#include <immintrin.h>
#endif

#include <faiss/IndexIVF.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/utils/fp16.h>
#include <faiss/utils/utils.h>

namespace faiss {

/*******************************************************************
 * ScalarQuantizer implementation
 *
 * The main source of complexity is to support combinations of 4
 * variants without incurring runtime tests or virtual function calls:
 *
 * - 4 / 8 bits per code component
 * - uniform / non-uniform
 * - IP / L2 distance search
 * - scalar / AVX distance computation
 *
 * The appropriate Quantizer object is returned via select_quantizer
 * that hides the template mess.
 ********************************************************************/

#ifdef __AVX2__
#ifdef __F16C__
#define USE_F16C
#else
#warning \
        "Cannot enable AVX optimizations in scalar quantizer if -mf16c is not set as well"
#endif
#endif

namespace {

typedef ScalarQuantizer::QuantizerType QuantizerType;
typedef ScalarQuantizer::RangeStat RangeStat;
using SQDistanceComputer = ScalarQuantizer::SQDistanceComputer;

/*******************************************************************
 * Codec: converts between values in [0, 1] and an index in a code
 * array. The "i" parameter is the vector component index (not byte
 * index).
 */

struct Codec8bit {
    static void encode_component(float x, uint8_t* code, int i) {
        code[i] = (int)(255 * x);
    }

    static float decode_component(const uint8_t* code, int i) {
        return (code[i] + 0.5f) / 255.0f;
    }

#ifdef __AVX2__
    static __m256 decode_8_components(const uint8_t* code, int i) {
        uint64_t c8 = *(uint64_t*)(code + i);
        __m128i c4lo = _mm_cvtepu8_epi32(_mm_set1_epi32(c8));
        __m128i c4hi = _mm_cvtepu8_epi32(_mm_set1_epi32(c8 >> 32));
        // __m256i i8 = _mm256_set_m128i(c4lo, c4hi);
        __m256i i8 = _mm256_castsi128_si256(c4lo);
        i8 = _mm256_insertf128_si256(i8, c4hi, 1);
        __m256 f8 = _mm256_cvtepi32_ps(i8);
        __m256 half = _mm256_set1_ps(0.5f);
        f8 = _mm256_add_ps(f8, half);
        __m256 one_255 = _mm256_set1_ps(1.f / 255.f);
        return _mm256_mul_ps(f8, one_255);
    }
#endif
};

struct Codec4bit {
    static void encode_component(float x, uint8_t* code, int i) {
        code[i / 2] |= (int)(x * 15.0) << ((i & 1) << 2);
    }

    static float decode_component(const uint8_t* code, int i) {
        return (((code[i / 2] >> ((i & 1) << 2)) & 0xf) + 0.5f) / 15.0f;
    }

#ifdef __AVX2__
    static __m256 decode_8_components(const uint8_t* code, int i) {
        uint32_t c4 = *(uint32_t*)(code + (i >> 1));
        uint32_t mask = 0x0f0f0f0f;
        uint32_t c4ev = c4 & mask;
        uint32_t c4od = (c4 >> 4) & mask;

        // the 8 lower bytes of c8 contain the values
        __m128i c8 =
                _mm_unpacklo_epi8(_mm_set1_epi32(c4ev), _mm_set1_epi32(c4od));
        __m128i c4lo = _mm_cvtepu8_epi32(c8);
        __m128i c4hi = _mm_cvtepu8_epi32(_mm_srli_si128(c8, 4));
        __m256i i8 = _mm256_castsi128_si256(c4lo);
        i8 = _mm256_insertf128_si256(i8, c4hi, 1);
        __m256 f8 = _mm256_cvtepi32_ps(i8);
        __m256 half = _mm256_set1_ps(0.5f);
        f8 = _mm256_add_ps(f8, half);
        __m256 one_255 = _mm256_set1_ps(1.f / 15.f);
        return _mm256_mul_ps(f8, one_255);
    }
#endif
};

struct Codec6bit {
    static void encode_component(float x, uint8_t* code, int i) {
        int bits = (int)(x * 63.0);
        code += (i >> 2) * 3;
        switch (i & 3) {
            case 0:
                code[0] |= bits;
                break;
            case 1:
                code[0] |= bits << 6;
                code[1] |= bits >> 2;
                break;
            case 2:
                code[1] |= bits << 4;
                code[2] |= bits >> 4;
                break;
            case 3:
                code[2] |= bits << 2;
                break;
        }
    }

    static float decode_component(const uint8_t* code, int i) {
        uint8_t bits;
        code += (i >> 2) * 3;
        switch (i & 3) {
            case 0:
                bits = code[0] & 0x3f;
                break;
            case 1:
                bits = code[0] >> 6;
                bits |= (code[1] & 0xf) << 2;
                break;
            case 2:
                bits = code[1] >> 4;
                bits |= (code[2] & 3) << 4;
                break;
            case 3:
                bits = code[2] >> 2;
                break;
        }
        return (bits + 0.5f) / 63.0f;
    }

#ifdef __AVX2__

    /* Load 6 bytes that represent 8 6-bit values, return them as a
     * 8*32 bit vector register */
    static __m256i load6(const uint16_t* code16) {
        const __m128i perm = _mm_set_epi8(
                -1, 5, 5, 4, 4, 3, -1, 3, -1, 2, 2, 1, 1, 0, -1, 0);
        const __m256i shifts = _mm256_set_epi32(2, 4, 6, 0, 2, 4, 6, 0);

        // load 6 bytes
        __m128i c1 =
                _mm_set_epi16(0, 0, 0, 0, 0, code16[2], code16[1], code16[0]);

        // put in 8 * 32 bits
        __m128i c2 = _mm_shuffle_epi8(c1, perm);
        __m256i c3 = _mm256_cvtepi16_epi32(c2);

        // shift and mask out useless bits
        __m256i c4 = _mm256_srlv_epi32(c3, shifts);
        __m256i c5 = _mm256_and_si256(_mm256_set1_epi32(63), c4);
        return c5;
    }

    static __m256 decode_8_components(const uint8_t* code, int i) {
        __m256i i8 = load6((const uint16_t*)(code + (i >> 2) * 3));
        __m256 f8 = _mm256_cvtepi32_ps(i8);
        // this could also be done with bit manipulations but it is
        // not obviously faster
        __m256 half = _mm256_set1_ps(0.5f);
        f8 = _mm256_add_ps(f8, half);
        __m256 one_63 = _mm256_set1_ps(1.f / 63.f);
        return _mm256_mul_ps(f8, one_63);
    }

#endif
};

/*******************************************************************
 * Quantizer: normalizes scalar vector components, then passes them
 * through a codec
 *******************************************************************/

template <class Codec, bool uniform, int SIMD>
struct QuantizerTemplate {};

template <class Codec>
struct QuantizerTemplate<Codec, true, 1> : ScalarQuantizer::SQuantizer {
    const size_t d;
    const float vmin, vdiff;

    QuantizerTemplate(size_t d, const std::vector<float>& trained)
            : d(d), vmin(trained[0]), vdiff(trained[1]) {}

    void encode_vector(const float* x, uint8_t* code) const final {
        for (size_t i = 0; i < d; i++) {
            float xi = 0;
            if (vdiff != 0) {
                xi = (x[i] - vmin) / vdiff;
                if (xi < 0) {
                    xi = 0;
                }
                if (xi > 1.0) {
                    xi = 1.0;
                }
            }
            Codec::encode_component(xi, code, i);
        }
    }

    void decode_vector(const uint8_t* code, float* x) const final {
        for (size_t i = 0; i < d; i++) {
            float xi = Codec::decode_component(code, i);
            x[i] = vmin + xi * vdiff;
        }
    }

    float reconstruct_component(const uint8_t* code, int i) const {
        float xi = Codec::decode_component(code, i);
        return vmin + xi * vdiff;
    }
};

#ifdef __AVX2__

template <class Codec>
struct QuantizerTemplate<Codec, true, 8> : QuantizerTemplate<Codec, true, 1> {
    QuantizerTemplate(size_t d, const std::vector<float>& trained)
            : QuantizerTemplate<Codec, true, 1>(d, trained) {}

    __m256 reconstruct_8_components(const uint8_t* code, int i) const {
        __m256 xi = Codec::decode_8_components(code, i);
        return _mm256_add_ps(
                _mm256_set1_ps(this->vmin),
                _mm256_mul_ps(xi, _mm256_set1_ps(this->vdiff)));
    }
};

#endif

template <class Codec>
struct QuantizerTemplate<Codec, false, 1> : ScalarQuantizer::SQuantizer {
    const size_t d;
    const float *vmin, *vdiff;

    QuantizerTemplate(size_t d, const std::vector<float>& trained)
            : d(d), vmin(trained.data()), vdiff(trained.data() + d) {}

    void encode_vector(const float* x, uint8_t* code) const final {
        for (size_t i = 0; i < d; i++) {
            float xi = 0;
            if (vdiff[i] != 0) {
                xi = (x[i] - vmin[i]) / vdiff[i];
                if (xi < 0) {
                    xi = 0;
                }
                if (xi > 1.0) {
                    xi = 1.0;
                }
            }
            Codec::encode_component(xi, code, i);
        }
    }

    void decode_vector(const uint8_t* code, float* x) const final {
        for (size_t i = 0; i < d; i++) {
            float xi = Codec::decode_component(code, i);
            x[i] = vmin[i] + xi * vdiff[i];
        }
    }

    float reconstruct_component(const uint8_t* code, int i) const {
        float xi = Codec::decode_component(code, i);
        return vmin[i] + xi * vdiff[i];
    }
};

#ifdef __AVX2__

template <class Codec>
struct QuantizerTemplate<Codec, false, 8> : QuantizerTemplate<Codec, false, 1> {
    QuantizerTemplate(size_t d, const std::vector<float>& trained)
            : QuantizerTemplate<Codec, false, 1>(d, trained) {}

    __m256 reconstruct_8_components(const uint8_t* code, int i) const {
        __m256 xi = Codec::decode_8_components(code, i);
        return _mm256_add_ps(
                _mm256_loadu_ps(this->vmin + i),
                _mm256_mul_ps(xi, _mm256_loadu_ps(this->vdiff + i)));
    }
};

#endif

/*******************************************************************
 * FP16 quantizer
 *******************************************************************/

template <int SIMDWIDTH>
struct QuantizerFP16 {};

template <>
struct QuantizerFP16<1> : ScalarQuantizer::SQuantizer {
    const size_t d;

    QuantizerFP16(size_t d, const std::vector<float>& /* unused */) : d(d) {}

    void encode_vector(const float* x, uint8_t* code) const final {
        for (size_t i = 0; i < d; i++) {
            ((uint16_t*)code)[i] = encode_fp16(x[i]);
        }
    }

    void decode_vector(const uint8_t* code, float* x) const final {
        for (size_t i = 0; i < d; i++) {
            x[i] = decode_fp16(((uint16_t*)code)[i]);
        }
    }

    float reconstruct_component(const uint8_t* code, int i) const {
        return decode_fp16(((uint16_t*)code)[i]);
    }
};

#ifdef USE_F16C

template <>
struct QuantizerFP16<8> : QuantizerFP16<1> {
    QuantizerFP16(size_t d, const std::vector<float>& trained)
            : QuantizerFP16<1>(d, trained) {}

    __m256 reconstruct_8_components(const uint8_t* code, int i) const {
        __m128i codei = _mm_loadu_si128((const __m128i*)(code + 2 * i));
        return _mm256_cvtph_ps(codei);
    }
};

#endif

/*******************************************************************
 * 8bit_direct quantizer
 *******************************************************************/

template <int SIMDWIDTH>
struct Quantizer8bitDirect {};

template <>
struct Quantizer8bitDirect<1> : ScalarQuantizer::SQuantizer {
    const size_t d;

    Quantizer8bitDirect(size_t d, const std::vector<float>& /* unused */)
            : d(d) {}

    void encode_vector(const float* x, uint8_t* code) const final {
        for (size_t i = 0; i < d; i++) {
            code[i] = (uint8_t)x[i];
        }
    }

    void decode_vector(const uint8_t* code, float* x) const final {
        for (size_t i = 0; i < d; i++) {
            x[i] = code[i];
        }
    }

    float reconstruct_component(const uint8_t* code, int i) const {
        return code[i];
    }
};

#ifdef __AVX2__

template <>
struct Quantizer8bitDirect<8> : Quantizer8bitDirect<1> {
    Quantizer8bitDirect(size_t d, const std::vector<float>& trained)
            : Quantizer8bitDirect<1>(d, trained) {}

    __m256 reconstruct_8_components(const uint8_t* code, int i) const {
        __m128i x8 = _mm_loadl_epi64((__m128i*)(code + i)); // 8 * int8
        __m256i y8 = _mm256_cvtepu8_epi32(x8);              // 8 * int32
        return _mm256_cvtepi32_ps(y8);                      // 8 * float32
    }
};

#endif

template <int SIMDWIDTH>
ScalarQuantizer::SQuantizer* select_quantizer_1(
        QuantizerType qtype,
        size_t d,
        const std::vector<float>& trained) {
    switch (qtype) {
        case ScalarQuantizer::QT_8bit:
            return new QuantizerTemplate<Codec8bit, false, SIMDWIDTH>(
                    d, trained);
        case ScalarQuantizer::QT_6bit:
            return new QuantizerTemplate<Codec6bit, false, SIMDWIDTH>(
                    d, trained);
        case ScalarQuantizer::QT_4bit:
            return new QuantizerTemplate<Codec4bit, false, SIMDWIDTH>(
                    d, trained);
        case ScalarQuantizer::QT_8bit_uniform:
            return new QuantizerTemplate<Codec8bit, true, SIMDWIDTH>(
                    d, trained);
        case ScalarQuantizer::QT_4bit_uniform:
            return new QuantizerTemplate<Codec4bit, true, SIMDWIDTH>(
                    d, trained);
        case ScalarQuantizer::QT_fp16:
            return new QuantizerFP16<SIMDWIDTH>(d, trained);
        case ScalarQuantizer::QT_8bit_direct:
            return new Quantizer8bitDirect<SIMDWIDTH>(d, trained);
    }
    FAISS_THROW_MSG("unknown qtype");
}

/*******************************************************************
 * Quantizer range training
 */

static float sqr(float x) {
    return x * x;
}

void train_Uniform(
        RangeStat rs,
        float rs_arg,
        idx_t n,
        int k,
        const float* x,
        std::vector<float>& trained) {
    trained.resize(2);
    float& vmin = trained[0];
    float& vmax = trained[1];

    if (rs == ScalarQuantizer::RS_minmax) {
        vmin = HUGE_VAL;
        vmax = -HUGE_VAL;
        for (size_t i = 0; i < n; i++) {
            if (x[i] < vmin)
                vmin = x[i];
            if (x[i] > vmax)
                vmax = x[i];
        }
        float vexp = (vmax - vmin) * rs_arg;
        vmin -= vexp;
        vmax += vexp;
    } else if (rs == ScalarQuantizer::RS_meanstd) {
        double sum = 0, sum2 = 0;
        for (size_t i = 0; i < n; i++) {
            sum += x[i];
            sum2 += x[i] * x[i];
        }
        float mean = sum / n;
        float var = sum2 / n - mean * mean;
        float std = var <= 0 ? 1.0 : sqrt(var);

        vmin = mean - std * rs_arg;
        vmax = mean + std * rs_arg;
    } else if (rs == ScalarQuantizer::RS_quantiles) {
        std::vector<float> x_copy(n);
        memcpy(x_copy.data(), x, n * sizeof(*x));
        // TODO just do a qucikselect
        std::sort(x_copy.begin(), x_copy.end());
        int o = int(rs_arg * n);
        if (o < 0)
            o = 0;
        if (o > n - o)
            o = n / 2;
        vmin = x_copy[o];
        vmax = x_copy[n - 1 - o];

    } else if (rs == ScalarQuantizer::RS_optim) {
        float a, b;
        float sx = 0;
        {
            vmin = HUGE_VAL, vmax = -HUGE_VAL;
            for (size_t i = 0; i < n; i++) {
                if (x[i] < vmin)
                    vmin = x[i];
                if (x[i] > vmax)
                    vmax = x[i];
                sx += x[i];
            }
            b = vmin;
            a = (vmax - vmin) / (k - 1);
        }
        int verbose = false;
        int niter = 2000;
        float last_err = -1;
        int iter_last_err = 0;
        for (int it = 0; it < niter; it++) {
            float sn = 0, sn2 = 0, sxn = 0, err1 = 0;

            for (idx_t i = 0; i < n; i++) {
                float xi = x[i];
                float ni = floor((xi - b) / a + 0.5);
                if (ni < 0)
                    ni = 0;
                if (ni >= k)
                    ni = k - 1;
                err1 += sqr(xi - (ni * a + b));
                sn += ni;
                sn2 += ni * ni;
                sxn += ni * xi;
            }

            if (err1 == last_err) {
                iter_last_err++;
                if (iter_last_err == 16)
                    break;
            } else {
                last_err = err1;
                iter_last_err = 0;
            }

            float det = sqr(sn) - sn2 * n;

            b = (sn * sxn - sn2 * sx) / det;
            a = (sn * sx - n * sxn) / det;
            if (verbose) {
                printf("it %d, err1=%g            \r", it, err1);
                fflush(stdout);
            }
        }
        if (verbose)
            printf("\n");

        vmin = b;
        vmax = b + a * (k - 1);

    } else {
        FAISS_THROW_MSG("Invalid qtype");
    }
    vmax -= vmin;
}

void train_NonUniform(
        RangeStat rs,
        float rs_arg,
        idx_t n,
        int d,
        int k,
        const float* x,
        std::vector<float>& trained) {
    trained.resize(2 * d);
    float* vmin = trained.data();
    float* vmax = trained.data() + d;
    if (rs == ScalarQuantizer::RS_minmax) {
        memcpy(vmin, x, sizeof(*x) * d);
        memcpy(vmax, x, sizeof(*x) * d);
        for (size_t i = 1; i < n; i++) {
            const float* xi = x + i * d;
            for (size_t j = 0; j < d; j++) {
                if (xi[j] < vmin[j])
                    vmin[j] = xi[j];
                if (xi[j] > vmax[j])
                    vmax[j] = xi[j];
            }
        }
        float* vdiff = vmax;
        for (size_t j = 0; j < d; j++) {
            float vexp = (vmax[j] - vmin[j]) * rs_arg;
            vmin[j] -= vexp;
            vmax[j] += vexp;
            vdiff[j] = vmax[j] - vmin[j];
        }
    } else {
        // transpose
        std::vector<float> xt(n * d);
        for (size_t i = 1; i < n; i++) {
            const float* xi = x + i * d;
            for (size_t j = 0; j < d; j++) {
                xt[j * n + i] = xi[j];
            }
        }
        std::vector<float> trained_d(2);
#pragma omp parallel for
        for (int j = 0; j < d; j++) {
            train_Uniform(rs, rs_arg, n, k, xt.data() + j * n, trained_d);
            vmin[j] = trained_d[0];
            vmax[j] = trained_d[1];
        }
    }
}

/*******************************************************************
 * Similarity: gets vector components and computes a similarity wrt. a
 * query vector stored in the object. The data fields just encapsulate
 * an accumulator.
 */

template <int SIMDWIDTH>
struct SimilarityL2 {};

template <>
struct SimilarityL2<1> {
    static constexpr int simdwidth = 1;
    static constexpr MetricType metric_type = METRIC_L2;

    const float *y, *yi;

    explicit SimilarityL2(const float* y) : y(y) {}

    /******* scalar accumulator *******/

    float accu;

    void begin() {
        accu = 0;
        yi = y;
    }

    void add_component(float x) {
        float tmp = *yi++ - x;
        accu += tmp * tmp;
    }

    void add_component_2(float x1, float x2) {
        float tmp = x1 - x2;
        accu += tmp * tmp;
    }

    float result() {
        return accu;
    }
};

#ifdef __AVX2__
template <>
struct SimilarityL2<8> {
    static constexpr int simdwidth = 8;
    static constexpr MetricType metric_type = METRIC_L2;

    const float *y, *yi;

    explicit SimilarityL2(const float* y) : y(y) {}
    __m256 accu8;

    void begin_8() {
        accu8 = _mm256_setzero_ps();
        yi = y;
    }

    void add_8_components(__m256 x) {
        __m256 yiv = _mm256_loadu_ps(yi);
        yi += 8;
        __m256 tmp = _mm256_sub_ps(yiv, x);
        accu8 = _mm256_add_ps(accu8, _mm256_mul_ps(tmp, tmp));
    }

    void add_8_components_2(__m256 x, __m256 y) {
        __m256 tmp = _mm256_sub_ps(y, x);
        accu8 = _mm256_add_ps(accu8, _mm256_mul_ps(tmp, tmp));
    }

    float result_8() {
        __m256 sum = _mm256_hadd_ps(accu8, accu8);
        __m256 sum2 = _mm256_hadd_ps(sum, sum);
        // now add the 0th and 4th component
        return _mm_cvtss_f32(_mm256_castps256_ps128(sum2)) +
                _mm_cvtss_f32(_mm256_extractf128_ps(sum2, 1));
    }
};

#endif

template <int SIMDWIDTH>
struct SimilarityIP {};

template <>
struct SimilarityIP<1> {
    static constexpr int simdwidth = 1;
    static constexpr MetricType metric_type = METRIC_INNER_PRODUCT;
    const float *y, *yi;

    float accu;

    explicit SimilarityIP(const float* y) : y(y) {}

    void begin() {
        accu = 0;
        yi = y;
    }

    void add_component(float x) {
        accu += *yi++ * x;
    }

    void add_component_2(float x1, float x2) {
        accu += x1 * x2;
    }

    float result() {
        return accu;
    }
};

#ifdef __AVX2__

template <>
struct SimilarityIP<8> {
    static constexpr int simdwidth = 8;
    static constexpr MetricType metric_type = METRIC_INNER_PRODUCT;

    const float *y, *yi;

    float accu;

    explicit SimilarityIP(const float* y) : y(y) {}

    __m256 accu8;

    void begin_8() {
        accu8 = _mm256_setzero_ps();
        yi = y;
    }

    void add_8_components(__m256 x) {
        __m256 yiv = _mm256_loadu_ps(yi);
        yi += 8;
        accu8 = _mm256_add_ps(accu8, _mm256_mul_ps(yiv, x));
    }

    void add_8_components_2(__m256 x1, __m256 x2) {
        accu8 = _mm256_add_ps(accu8, _mm256_mul_ps(x1, x2));
    }

    float result_8() {
        __m256 sum = _mm256_hadd_ps(accu8, accu8);
        __m256 sum2 = _mm256_hadd_ps(sum, sum);
        // now add the 0th and 4th component
        return _mm_cvtss_f32(_mm256_castps256_ps128(sum2)) +
                _mm_cvtss_f32(_mm256_extractf128_ps(sum2, 1));
    }
};
#endif

/*******************************************************************
 * DistanceComputer: combines a similarity and a quantizer to do
 * code-to-vector or code-to-code comparisons
 *******************************************************************/

template <class Quantizer, class Similarity, int SIMDWIDTH>
struct DCTemplate : SQDistanceComputer {};

template <class Quantizer, class Similarity>
struct DCTemplate<Quantizer, Similarity, 1> : SQDistanceComputer {
    using Sim = Similarity;

    Quantizer quant;

    DCTemplate(size_t d, const std::vector<float>& trained)
            : quant(d, trained) {}

    float compute_distance(const float* x, const uint8_t* code) const {
        Similarity sim(x);
        sim.begin();
        for (size_t i = 0; i < quant.d; i++) {
            float xi = quant.reconstruct_component(code, i);
            sim.add_component(xi);
        }
        return sim.result();
    }

    float compute_code_distance(const uint8_t* code1, const uint8_t* code2)
            const {
        Similarity sim(nullptr);
        sim.begin();
        for (size_t i = 0; i < quant.d; i++) {
            float x1 = quant.reconstruct_component(code1, i);
            float x2 = quant.reconstruct_component(code2, i);
            sim.add_component_2(x1, x2);
        }
        return sim.result();
    }

    void set_query(const float* x) final {
        q = x;
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return compute_code_distance(
                codes + i * code_size, codes + j * code_size);
    }

    float query_to_code(const uint8_t* code) const final {
        return compute_distance(q, code);
    }
};

#ifdef USE_F16C

template <class Quantizer, class Similarity>
struct DCTemplate<Quantizer, Similarity, 8> : SQDistanceComputer {
    using Sim = Similarity;

    Quantizer quant;

    DCTemplate(size_t d, const std::vector<float>& trained)
            : quant(d, trained) {}

    float compute_distance(const float* x, const uint8_t* code) const {
        Similarity sim(x);
        sim.begin_8();
        for (size_t i = 0; i < quant.d; i += 8) {
            __m256 xi = quant.reconstruct_8_components(code, i);
            sim.add_8_components(xi);
        }
        return sim.result_8();
    }

    float compute_code_distance(const uint8_t* code1, const uint8_t* code2)
            const {
        Similarity sim(nullptr);
        sim.begin_8();
        for (size_t i = 0; i < quant.d; i += 8) {
            __m256 x1 = quant.reconstruct_8_components(code1, i);
            __m256 x2 = quant.reconstruct_8_components(code2, i);
            sim.add_8_components_2(x1, x2);
        }
        return sim.result_8();
    }

    void set_query(const float* x) final {
        q = x;
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return compute_code_distance(
                codes + i * code_size, codes + j * code_size);
    }

    float query_to_code(const uint8_t* code) const final {
        return compute_distance(q, code);
    }
};

#endif

/*******************************************************************
 * DistanceComputerByte: computes distances in the integer domain
 *******************************************************************/

template <class Similarity, int SIMDWIDTH>
struct DistanceComputerByte : SQDistanceComputer {};

template <class Similarity>
struct DistanceComputerByte<Similarity, 1> : SQDistanceComputer {
    using Sim = Similarity;

    int d;
    std::vector<uint8_t> tmp;

    DistanceComputerByte(int d, const std::vector<float>&) : d(d), tmp(d) {}

    int compute_code_distance(const uint8_t* code1, const uint8_t* code2)
            const {
        int accu = 0;
        for (int i = 0; i < d; i++) {
            if (Sim::metric_type == METRIC_INNER_PRODUCT) {
                accu += int(code1[i]) * code2[i];
            } else {
                int diff = int(code1[i]) - code2[i];
                accu += diff * diff;
            }
        }
        return accu;
    }

    void set_query(const float* x) final {
        for (int i = 0; i < d; i++) {
            tmp[i] = int(x[i]);
        }
    }

    int compute_distance(const float* x, const uint8_t* code) {
        set_query(x);
        return compute_code_distance(tmp.data(), code);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return compute_code_distance(
                codes + i * code_size, codes + j * code_size);
    }

    float query_to_code(const uint8_t* code) const final {
        return compute_code_distance(tmp.data(), code);
    }
};

#ifdef __AVX2__

template <class Similarity>
struct DistanceComputerByte<Similarity, 8> : SQDistanceComputer {
    using Sim = Similarity;

    int d;
    std::vector<uint8_t> tmp;

    DistanceComputerByte(int d, const std::vector<float>&) : d(d), tmp(d) {}

    int compute_code_distance(const uint8_t* code1, const uint8_t* code2)
            const {
        // __m256i accu = _mm256_setzero_ps ();
        __m256i accu = _mm256_setzero_si256();
        for (int i = 0; i < d; i += 16) {
            // load 16 bytes, convert to 16 uint16_t
            __m256i c1 = _mm256_cvtepu8_epi16(
                    _mm_loadu_si128((__m128i*)(code1 + i)));
            __m256i c2 = _mm256_cvtepu8_epi16(
                    _mm_loadu_si128((__m128i*)(code2 + i)));
            __m256i prod32;
            if (Sim::metric_type == METRIC_INNER_PRODUCT) {
                prod32 = _mm256_madd_epi16(c1, c2);
            } else {
                __m256i diff = _mm256_sub_epi16(c1, c2);
                prod32 = _mm256_madd_epi16(diff, diff);
            }
            accu = _mm256_add_epi32(accu, prod32);
        }
        __m128i sum = _mm256_extractf128_si256(accu, 0);
        sum = _mm_add_epi32(sum, _mm256_extractf128_si256(accu, 1));
        sum = _mm_hadd_epi32(sum, sum);
        sum = _mm_hadd_epi32(sum, sum);
        return _mm_cvtsi128_si32(sum);
    }

    void set_query(const float* x) final {
        /*
        for (int i = 0; i < d; i += 8) {
            __m256 xi = _mm256_loadu_ps (x + i);
            __m256i ci = _mm256_cvtps_epi32(xi);
        */
        for (int i = 0; i < d; i++) {
            tmp[i] = int(x[i]);
        }
    }

    int compute_distance(const float* x, const uint8_t* code) {
        set_query(x);
        return compute_code_distance(tmp.data(), code);
    }

    float symmetric_dis(idx_t i, idx_t j) override {
        return compute_code_distance(
                codes + i * code_size, codes + j * code_size);
    }

    float query_to_code(const uint8_t* code) const final {
        return compute_code_distance(tmp.data(), code);
    }
};

#endif

/*******************************************************************
 * select_distance_computer: runtime selection of template
 * specialization
 *******************************************************************/

template <class Sim>
SQDistanceComputer* select_distance_computer(
        QuantizerType qtype,
        size_t d,
        const std::vector<float>& trained) {
    constexpr int SIMDWIDTH = Sim::simdwidth;
    switch (qtype) {
        case ScalarQuantizer::QT_8bit_uniform:
            return new DCTemplate<
                    QuantizerTemplate<Codec8bit, true, SIMDWIDTH>,
                    Sim,
                    SIMDWIDTH>(d, trained);

        case ScalarQuantizer::QT_4bit_uniform:
            return new DCTemplate<
                    QuantizerTemplate<Codec4bit, true, SIMDWIDTH>,
                    Sim,
                    SIMDWIDTH>(d, trained);

        case ScalarQuantizer::QT_8bit:
            return new DCTemplate<
                    QuantizerTemplate<Codec8bit, false, SIMDWIDTH>,
                    Sim,
                    SIMDWIDTH>(d, trained);

        case ScalarQuantizer::QT_6bit:
            return new DCTemplate<
                    QuantizerTemplate<Codec6bit, false, SIMDWIDTH>,
                    Sim,
                    SIMDWIDTH>(d, trained);

        case ScalarQuantizer::QT_4bit:
            return new DCTemplate<
                    QuantizerTemplate<Codec4bit, false, SIMDWIDTH>,
                    Sim,
                    SIMDWIDTH>(d, trained);

        case ScalarQuantizer::QT_fp16:
            return new DCTemplate<QuantizerFP16<SIMDWIDTH>, Sim, SIMDWIDTH>(
                    d, trained);

        case ScalarQuantizer::QT_8bit_direct:
            if (d % 16 == 0) {
                return new DistanceComputerByte<Sim, SIMDWIDTH>(d, trained);
            } else {
                return new DCTemplate<
                        Quantizer8bitDirect<SIMDWIDTH>,
                        Sim,
                        SIMDWIDTH>(d, trained);
            }
    }
    FAISS_THROW_MSG("unknown qtype");
    return nullptr;
}

} // anonymous namespace

/*******************************************************************
 * ScalarQuantizer implementation
 ********************************************************************/

ScalarQuantizer::ScalarQuantizer(size_t d, QuantizerType qtype)
        : Quantizer(d), qtype(qtype), rangestat(RS_minmax), rangestat_arg(0) {
    set_derived_sizes();
}

ScalarQuantizer::ScalarQuantizer()
        : qtype(QT_8bit), rangestat(RS_minmax), rangestat_arg(0), bits(0) {}

void ScalarQuantizer::set_derived_sizes() {
    switch (qtype) {
        case QT_8bit:
        case QT_8bit_uniform:
        case QT_8bit_direct:
            code_size = d;
            bits = 8;
            break;
        case QT_4bit:
        case QT_4bit_uniform:
            code_size = (d + 1) / 2;
            bits = 4;
            break;
        case QT_6bit:
            code_size = (d * 6 + 7) / 8;
            bits = 6;
            break;
        case QT_fp16:
            code_size = d * 2;
            bits = 16;
            break;
    }
}

void ScalarQuantizer::train(size_t n, const float* x) {
    int bit_per_dim = qtype == QT_4bit_uniform ? 4
            : qtype == QT_4bit                 ? 4
            : qtype == QT_6bit                 ? 6
            : qtype == QT_8bit_uniform         ? 8
            : qtype == QT_8bit                 ? 8
                                               : -1;

    switch (qtype) {
        case QT_4bit_uniform:
        case QT_8bit_uniform:
            train_Uniform(
                    rangestat,
                    rangestat_arg,
                    n * d,
                    1 << bit_per_dim,
                    x,
                    trained);
            break;
        case QT_4bit:
        case QT_8bit:
        case QT_6bit:
            train_NonUniform(
                    rangestat,
                    rangestat_arg,
                    n,
                    d,
                    1 << bit_per_dim,
                    x,
                    trained);
            break;
        case QT_fp16:
        case QT_8bit_direct:
            // no training necessary
            break;
    }
}

void ScalarQuantizer::train_residual(
        size_t n,
        const float* x,
        Index* quantizer,
        bool by_residual,
        bool verbose) {
    const float* x_in = x;

    // 100k points more than enough
    x = fvecs_maybe_subsample(d, (size_t*)&n, 100000, x, verbose, 1234);

    ScopeDeleter<float> del_x(x_in == x ? nullptr : x);

    if (by_residual) {
        std::vector<idx_t> idx(n);
        quantizer->assign(n, x, idx.data());

        std::vector<float> residuals(n * d);
        quantizer->compute_residual_n(n, x, residuals.data(), idx.data());

        train(n, residuals.data());
    } else {
        train(n, x);
    }
}

ScalarQuantizer::SQuantizer* ScalarQuantizer::select_quantizer() const {
#ifdef USE_F16C
    if (d % 8 == 0) {
        return select_quantizer_1<8>(qtype, d, trained);
    } else
#endif
    {
        return select_quantizer_1<1>(qtype, d, trained);
    }
}

void ScalarQuantizer::compute_codes(const float* x, uint8_t* codes, size_t n)
        const {
    std::unique_ptr<SQuantizer> squant(select_quantizer());

    memset(codes, 0, code_size * n);
#pragma omp parallel for
    for (int64_t i = 0; i < n; i++)
        squant->encode_vector(x + i * d, codes + i * code_size);
}

void ScalarQuantizer::decode(const uint8_t* codes, float* x, size_t n) const {
    std::unique_ptr<SQuantizer> squant(select_quantizer());

#pragma omp parallel for
    for (int64_t i = 0; i < n; i++)
        squant->decode_vector(codes + i * code_size, x + i * d);
}

SQDistanceComputer* ScalarQuantizer::get_distance_computer(
        MetricType metric) const {
    FAISS_THROW_IF_NOT(metric == METRIC_L2 || metric == METRIC_INNER_PRODUCT);
#ifdef USE_F16C
    if (d % 8 == 0) {
        if (metric == METRIC_L2) {
            return select_distance_computer<SimilarityL2<8>>(qtype, d, trained);
        } else {
            return select_distance_computer<SimilarityIP<8>>(qtype, d, trained);
        }
    } else
#endif
    {
        if (metric == METRIC_L2) {
            return select_distance_computer<SimilarityL2<1>>(qtype, d, trained);
        } else {
            return select_distance_computer<SimilarityIP<1>>(qtype, d, trained);
        }
    }
}

/*******************************************************************
 * IndexScalarQuantizer/IndexIVFScalarQuantizer scanner object
 *
 * It is an InvertedListScanner, but is designed to work with
 * IndexScalarQuantizer as well.
 ********************************************************************/

namespace {

template <class DCClass, int use_sel>
struct IVFSQScannerIP : InvertedListScanner {
    DCClass dc;
    bool by_residual;
    const IDSelector* sel;

    float accu0; /// added to all distances

    IVFSQScannerIP(
            int d,
            const std::vector<float>& trained,
            size_t code_size,
            bool store_pairs,
            const IDSelector* sel,
            bool by_residual)
            : dc(d, trained), by_residual(by_residual), sel(sel), accu0(0) {
        this->store_pairs = store_pairs;
        this->code_size = code_size;
    }

    void set_query(const float* query) override {
        dc.set_query(query);
    }

    void set_list(idx_t list_no, float coarse_dis) override {
        this->list_no = list_no;
        accu0 = by_residual ? coarse_dis : 0;
    }

    float distance_to_code(const uint8_t* code) const final {
        return accu0 + dc.query_to_code(code);
    }

    size_t scan_codes(
            size_t list_size,
            const uint8_t* codes,
            const idx_t* ids,
            float* simi,
            idx_t* idxi,
            size_t k) const override {
        size_t nup = 0;

        for (size_t j = 0; j < list_size; j++, codes += code_size) {
            if (use_sel && !sel->is_member(use_sel == 1 ? ids[j] : j)) {
                continue;
            }

            float accu = accu0 + dc.query_to_code(codes);

            if (accu > simi[0]) {
                int64_t id = store_pairs ? (list_no << 32 | j) : ids[j];
                minheap_replace_top(k, simi, idxi, accu, id);
                nup++;
            }
        }
        return nup;
    }

    void scan_codes_range(
            size_t list_size,
            const uint8_t* codes,
            const idx_t* ids,
            float radius,
            RangeQueryResult& res) const override {
        for (size_t j = 0; j < list_size; j++, codes += code_size) {
            if (use_sel && !sel->is_member(use_sel == 1 ? ids[j] : j)) {
                continue;
            }

            float accu = accu0 + dc.query_to_code(codes);
            if (accu > radius) {
                int64_t id = store_pairs ? (list_no << 32 | j) : ids[j];
                res.add(accu, id);
            }
        }
    }
};

/* use_sel = 0: don't check selector
 * = 1: check on ids[j]
 * = 2: check in j directly (normally ids is nullptr and store_pairs)
 */
template <class DCClass, int use_sel>
struct IVFSQScannerL2 : InvertedListScanner {
    DCClass dc;

    bool by_residual;
    const Index* quantizer;
    const IDSelector* sel;
    const float* x; /// current query

    std::vector<float> tmp;

    IVFSQScannerL2(
            int d,
            const std::vector<float>& trained,
            size_t code_size,
            const Index* quantizer,
            bool store_pairs,
            const IDSelector* sel,
            bool by_residual)
            : dc(d, trained),
              by_residual(by_residual),
              quantizer(quantizer),
              sel(sel),
              x(nullptr),
              tmp(d) {
        this->store_pairs = store_pairs;
        this->code_size = code_size;
    }

    void set_query(const float* query) override {
        x = query;
        if (!quantizer) {
            dc.set_query(query);
        }
    }

    void set_list(idx_t list_no, float /*coarse_dis*/) override {
        this->list_no = list_no;
        if (by_residual) {
            // shift of x_in wrt centroid
            quantizer->compute_residual(x, tmp.data(), list_no);
            dc.set_query(tmp.data());
        } else {
            dc.set_query(x);
        }
    }

    float distance_to_code(const uint8_t* code) const final {
        return dc.query_to_code(code);
    }

    size_t scan_codes(
            size_t list_size,
            const uint8_t* codes,
            const idx_t* ids,
            float* simi,
            idx_t* idxi,
            size_t k) const override {
        size_t nup = 0;
        for (size_t j = 0; j < list_size; j++, codes += code_size) {
            if (use_sel && !sel->is_member(use_sel == 1 ? ids[j] : j)) {
                continue;
            }

            float dis = dc.query_to_code(codes);

            if (dis < simi[0]) {
                int64_t id = store_pairs ? (list_no << 32 | j) : ids[j];
                maxheap_replace_top(k, simi, idxi, dis, id);
                nup++;
            }
        }
        return nup;
    }

    void scan_codes_range(
            size_t list_size,
            const uint8_t* codes,
            const idx_t* ids,
            float radius,
            RangeQueryResult& res) const override {
        for (size_t j = 0; j < list_size; j++, codes += code_size) {
            if (use_sel && !sel->is_member(use_sel == 1 ? ids[j] : j)) {
                continue;
            }

            float dis = dc.query_to_code(codes);
            if (dis < radius) {
                int64_t id = store_pairs ? (list_no << 32 | j) : ids[j];
                res.add(dis, id);
            }
        }
    }
};

template <class DCClass, int use_sel>
InvertedListScanner* sel3_InvertedListScanner(
        const ScalarQuantizer* sq,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool r) {
    if (DCClass::Sim::metric_type == METRIC_L2) {
        return new IVFSQScannerL2<DCClass, use_sel>(
                sq->d,
                sq->trained,
                sq->code_size,
                quantizer,
                store_pairs,
                sel,
                r);
    } else if (DCClass::Sim::metric_type == METRIC_INNER_PRODUCT) {
        return new IVFSQScannerIP<DCClass, use_sel>(
                sq->d, sq->trained, sq->code_size, store_pairs, sel, r);
    } else {
        FAISS_THROW_MSG("unsupported metric type");
    }
}

template <class DCClass>
InvertedListScanner* sel2_InvertedListScanner(
        const ScalarQuantizer* sq,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool r) {
    if (sel) {
        if (store_pairs) {
            return sel3_InvertedListScanner<DCClass, 2>(
                    sq, quantizer, store_pairs, sel, r);
        } else {
            return sel3_InvertedListScanner<DCClass, 1>(
                    sq, quantizer, store_pairs, sel, r);
        }
    } else {
        return sel3_InvertedListScanner<DCClass, 0>(
                sq, quantizer, store_pairs, sel, r);
    }
}

template <class Similarity, class Codec, bool uniform>
InvertedListScanner* sel12_InvertedListScanner(
        const ScalarQuantizer* sq,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool r) {
    constexpr int SIMDWIDTH = Similarity::simdwidth;
    using QuantizerClass = QuantizerTemplate<Codec, uniform, SIMDWIDTH>;
    using DCClass = DCTemplate<QuantizerClass, Similarity, SIMDWIDTH>;
    return sel2_InvertedListScanner<DCClass>(
            sq, quantizer, store_pairs, sel, r);
}

template <class Similarity>
InvertedListScanner* sel1_InvertedListScanner(
        const ScalarQuantizer* sq,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool r) {
    constexpr int SIMDWIDTH = Similarity::simdwidth;
    switch (sq->qtype) {
        case ScalarQuantizer::QT_8bit_uniform:
            return sel12_InvertedListScanner<Similarity, Codec8bit, true>(
                    sq, quantizer, store_pairs, sel, r);
        case ScalarQuantizer::QT_4bit_uniform:
            return sel12_InvertedListScanner<Similarity, Codec4bit, true>(
                    sq, quantizer, store_pairs, sel, r);
        case ScalarQuantizer::QT_8bit:
            return sel12_InvertedListScanner<Similarity, Codec8bit, false>(
                    sq, quantizer, store_pairs, sel, r);
        case ScalarQuantizer::QT_4bit:
            return sel12_InvertedListScanner<Similarity, Codec4bit, false>(
                    sq, quantizer, store_pairs, sel, r);
        case ScalarQuantizer::QT_6bit:
            return sel12_InvertedListScanner<Similarity, Codec6bit, false>(
                    sq, quantizer, store_pairs, sel, r);
        case ScalarQuantizer::QT_fp16:
            return sel2_InvertedListScanner<DCTemplate<
                    QuantizerFP16<SIMDWIDTH>,
                    Similarity,
                    SIMDWIDTH>>(sq, quantizer, store_pairs, sel, r);
        case ScalarQuantizer::QT_8bit_direct:
            if (sq->d % 16 == 0) {
                return sel2_InvertedListScanner<
                        DistanceComputerByte<Similarity, SIMDWIDTH>>(
                        sq, quantizer, store_pairs, sel, r);
            } else {
                return sel2_InvertedListScanner<DCTemplate<
                        Quantizer8bitDirect<SIMDWIDTH>,
                        Similarity,
                        SIMDWIDTH>>(sq, quantizer, store_pairs, sel, r);
            }
    }

    FAISS_THROW_MSG("unknown qtype");
    return nullptr;
}

template <int SIMDWIDTH>
InvertedListScanner* sel0_InvertedListScanner(
        MetricType mt,
        const ScalarQuantizer* sq,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool by_residual) {
    if (mt == METRIC_L2) {
        return sel1_InvertedListScanner<SimilarityL2<SIMDWIDTH>>(
                sq, quantizer, store_pairs, sel, by_residual);
    } else if (mt == METRIC_INNER_PRODUCT) {
        return sel1_InvertedListScanner<SimilarityIP<SIMDWIDTH>>(
                sq, quantizer, store_pairs, sel, by_residual);
    } else {
        FAISS_THROW_MSG("unsupported metric type");
    }
}

} // anonymous namespace

InvertedListScanner* ScalarQuantizer::select_InvertedListScanner(
        MetricType mt,
        const Index* quantizer,
        bool store_pairs,
        const IDSelector* sel,
        bool by_residual) const {
#ifdef USE_F16C
    if (d % 8 == 0) {
        return sel0_InvertedListScanner<8>(
                mt, this, quantizer, store_pairs, sel, by_residual);
    } else
#endif
    {
        return sel0_InvertedListScanner<1>(
                mt, this, quantizer, store_pairs, sel, by_residual);
    }
}

} // namespace faiss
