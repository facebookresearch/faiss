/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "IndexScalarQuantizer.h"

#include <cstdio>
#include <algorithm>

#include <omp.h>

#include <immintrin.h>

#include "utils.h"

#include "FaissAssert.h"

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

#ifdef __AVX__
#define USE_AVX
#endif


namespace {

typedef Index::idx_t idx_t;
typedef ScalarQuantizer::QuantizerType QuantizerType;
typedef ScalarQuantizer::RangeStat RangeStat;
using DistanceComputer = ScalarQuantizer::DistanceComputer;


/*******************************************************************
 * Codec: converts between values in [0, 1] and an index in a code
 * array. The "i" parameter is the vector component index (not byte
 * index).
 */

struct Codec8bit {

    static void encode_component (float x, uint8_t *code, int i) {
        code[i] = (int)(255 * x);
    }

    static float decode_component (const uint8_t *code, int i) {
        return (code[i] + 0.5f) / 255.0f;
    }

#ifdef USE_AVX
    static __m256 decode_8_components (const uint8_t *code, int i) {
        uint64_t c8 = *(uint64_t*)(code + i);
        __m128i c4lo = _mm_cvtepu8_epi32 (_mm_set1_epi32(c8));
        __m128i c4hi = _mm_cvtepu8_epi32 (_mm_set1_epi32(c8 >> 32));
        // __m256i i8 = _mm256_set_m128i(c4lo, c4hi);
        __m256i i8 = _mm256_castsi128_si256 (c4lo);
        i8 = _mm256_insertf128_si256 (i8, c4hi, 1);
        __m256 f8 = _mm256_cvtepi32_ps (i8);
        __m256 half = _mm256_set1_ps (0.5f);
        f8 += half;
        __m256 one_255 = _mm256_set1_ps (1.f / 255.f);
        return f8 * one_255;
    }
#endif
};


struct Codec4bit {

    static void encode_component (float x, uint8_t *code, int i) {
        code [i / 2] |= (int)(x * 15.0) << ((i & 1) << 2);
    }

    static float decode_component (const uint8_t *code, int i) {
        return (((code[i / 2] >> ((i & 1) << 2)) & 0xf) + 0.5f) / 15.0f;
    }


#ifdef USE_AVX
    static __m256 decode_8_components (const uint8_t *code, int i) {
        uint32_t c4 = *(uint32_t*)(code + (i >> 1));
        uint32_t mask = 0x0f0f0f0f;
        uint32_t c4ev = c4 & mask;
        uint32_t c4od = (c4 >> 4) & mask;

        // the 8 lower bytes of c8 contain the values
        __m128i c8 = _mm_unpacklo_epi8 (_mm_set1_epi32(c4ev),
                                        _mm_set1_epi32(c4od));
        __m128i c4lo = _mm_cvtepu8_epi32 (c8);
        __m128i c4hi = _mm_cvtepu8_epi32 (_mm_srli_si128(c8, 4));
        __m256i i8 = _mm256_castsi128_si256 (c4lo);
        i8 = _mm256_insertf128_si256 (i8, c4hi, 1);
        __m256 f8 = _mm256_cvtepi32_ps (i8);
        __m256 half = _mm256_set1_ps (0.5f);
        f8 += half;
        __m256 one_255 = _mm256_set1_ps (1.f / 15.f);
        return f8 * one_255;
    }
#endif
};



/*******************************************************************
 * Quantizer: normalizes scalar vector components, then passes them
 * through a codec
 */



struct Quantizer {
    virtual void encode_vector(const float *x, uint8_t *code) const = 0;
    virtual void decode_vector(const uint8_t *code, float *x) const = 0;


    virtual ~Quantizer() {}
};





template<class Codec>
struct QuantizerUniform: Quantizer {
    const size_t d;
    const float vmin, vdiff;

    QuantizerUniform(size_t d, const std::vector<float> &trained):
        d(d), vmin(trained[0]), vdiff(trained[1])
    {
    }

    void encode_vector(const float* x, uint8_t* code) const override {
      for (size_t i = 0; i < d; i++) {
        float xi = (x[i] - vmin) / vdiff;
        if (xi < 0)
          xi = 0;
        if (xi > 1.0)
          xi = 1.0;
        Codec::encode_component(xi, code, i);
      }
    }

    void decode_vector(const uint8_t* code, float* x) const override {
      for (size_t i = 0; i < d; i++) {
        float xi = Codec::decode_component(code, i);
        x[i] = vmin + xi * vdiff;
      }
    }

    float reconstruct_component (const uint8_t * code, int i) const
    {
        float xi = Codec::decode_component (code, i);
        return vmin + xi * vdiff;
    }

};



#ifdef USE_AVX

template<class Codec>
struct QuantizerUniform8: QuantizerUniform<Codec> {

    QuantizerUniform8 (size_t d, const std::vector<float> &trained):
        QuantizerUniform<Codec> (d, trained) {}

    __m256 reconstruct_8_components (const uint8_t * code, int i) const
    {
        __m256 xi = Codec::decode_8_components (code, i);
        return _mm256_set1_ps(this->vmin) + xi * _mm256_set1_ps (this->vdiff);
    }

};

#endif



template<class Codec>
struct QuantizerNonUniform: Quantizer {
    const size_t d;
    const float *vmin, *vdiff;

    QuantizerNonUniform(size_t d, const std::vector<float> &trained):
        d(d), vmin(trained.data()), vdiff(trained.data() + d) {}

    void encode_vector(const float* x, uint8_t* code) const override {
      for (size_t i = 0; i < d; i++) {
        float xi = (x[i] - vmin[i]) / vdiff[i];
        if (xi < 0)
          xi = 0;
        if (xi > 1.0)
          xi = 1.0;
        Codec::encode_component(xi, code, i);
      }
    }

    void decode_vector(const uint8_t* code, float* x) const override {
      for (size_t i = 0; i < d; i++) {
        float xi = Codec::decode_component(code, i);
        x[i] = vmin[i] + xi * vdiff[i];
      }
    }

    float reconstruct_component (const uint8_t * code, int i) const
    {
        float xi = Codec::decode_component (code, i);
        return vmin[i] + xi * vdiff[i];
    }

};


#ifdef USE_AVX

template<class Codec>
struct QuantizerNonUniform8: QuantizerNonUniform<Codec> {

    QuantizerNonUniform8 (size_t d, const std::vector<float> &trained):
        QuantizerNonUniform<Codec> (d, trained) {}

    __m256 reconstruct_8_components (const uint8_t * code, int i) const
    {
        __m256 xi = Codec::decode_8_components (code, i);
        return _mm256_loadu_ps (this->vmin + i) + xi * _mm256_loadu_ps (this->vdiff + i);
    }


};

#endif

Quantizer *select_quantizer (
          QuantizerType qtype,
          size_t d, const std::vector<float> & trained)
{
#ifdef USE_AVX
    if (d % 8 == 0) {
        switch(qtype) {
        case ScalarQuantizer::QT_8bit:
            return new QuantizerNonUniform8<Codec8bit>(d, trained);
        case ScalarQuantizer::QT_4bit:
            return new QuantizerNonUniform8<Codec4bit>(d, trained);
        case ScalarQuantizer::QT_8bit_uniform:
            return new QuantizerUniform8<Codec8bit>(d, trained);
        case ScalarQuantizer::QT_4bit_uniform:
            return new QuantizerUniform8<Codec4bit>(d, trained);
        }
    } else
#endif
    {
        switch(qtype) {
        case ScalarQuantizer::QT_8bit:
            return new QuantizerNonUniform<Codec8bit>(d, trained);
        case ScalarQuantizer::QT_4bit:
            return new QuantizerNonUniform<Codec4bit>(d, trained);
        case ScalarQuantizer::QT_8bit_uniform:
            return new QuantizerUniform<Codec8bit>(d, trained);
        case ScalarQuantizer::QT_4bit_uniform:
            return new QuantizerUniform<Codec4bit>(d, trained);
        }
    }
    FAISS_THROW_MSG ("unknown qtype");
    return nullptr;
}


Quantizer *select_quantizer (const ScalarQuantizer &sq)
{
    return select_quantizer (sq.qtype, sq.d, sq.trained);
}




/*******************************************************************
 * Quantizer range training
 */

static float sqr (float x) {
    return x * x;
}


void train_Uniform(RangeStat rs, float rs_arg,
                   idx_t n, int k, const float *x,
                   std::vector<float> & trained)
{
    trained.resize (2);
    float & vmin = trained[0];
    float & vmax = trained[1];

    if (rs == ScalarQuantizer::RS_minmax) {
        vmin = HUGE_VAL; vmax = -HUGE_VAL;
        for (size_t i = 0; i < n; i++) {
            if (x[i] < vmin) vmin = x[i];
            if (x[i] > vmax) vmax = x[i];
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

        vmin = mean - std * rs_arg ;
        vmax = mean + std * rs_arg ;
    } else if (rs == ScalarQuantizer::RS_quantiles) {
        std::vector<float> x_copy(n);
        memcpy(x_copy.data(), x, n * sizeof(*x));
        // TODO just do a qucikselect
        std::sort(x_copy.begin(), x_copy.end());
        int o = int(rs_arg * n);
        if (o < 0) o = 0;
        if (o > n - o) o = n / 2;
        vmin = x_copy[o];
        vmax = x_copy[n - 1 - o];

    } else if (rs == ScalarQuantizer::RS_optim) {
        float a, b;
        float sx = 0;
        {
            vmin = HUGE_VAL, vmax = -HUGE_VAL;
            for (size_t i = 0; i < n; i++) {
                if (x[i] < vmin) vmin = x[i];
                if (x[i] > vmax) vmax = x[i];
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
                float ni = floor ((xi - b) / a + 0.5);
                if (ni < 0) ni = 0;
                if (ni >= k) ni = k - 1;
                err1 += sqr (xi - (ni * a + b));
                sn  += ni;
                sn2 += ni * ni;
                sxn += ni * xi;
            }

            if (err1 == last_err) {
                iter_last_err ++;
                if (iter_last_err == 16) break;
            } else {
                last_err = err1;
                iter_last_err = 0;
            }

            float det = sqr (sn) - sn2 * n;

            b = (sn * sxn - sn2 * sx) / det;
            a = (sn * sx - n * sxn) / det;
            if (verbose) {
                printf ("it %d, err1=%g            \r", it, err1);
                fflush(stdout);
            }
        }
        if (verbose) printf("\n");

        vmin = b;
        vmax = b + a * (k - 1);

    } else {
        FAISS_THROW_MSG ("Invalid qtype");
    }
    vmax -= vmin;
}

void train_NonUniform(RangeStat rs, float rs_arg,
                      idx_t n, int d, int k, const float *x,
                      std::vector<float> & trained)
{
    trained.resize (2 * d);
    float * vmin = trained.data();
    float * vmax = trained.data() + d;
    if (rs == ScalarQuantizer::RS_minmax) {
        memcpy (vmin, x, sizeof(*x) * d);
        memcpy (vmax, x, sizeof(*x) * d);
        for (size_t i = 1; i < n; i++) {
            const float *xi = x + i * d;
            for (size_t j = 0; j < d; j++) {
                if (xi[j] < vmin[j]) vmin[j] = xi[j];
                if (xi[j] > vmax[j]) vmax[j] = xi[j];
            }
        }
        float *vdiff = vmax;
        for (size_t j = 0; j < d; j++) {
            float vexp = (vmax[j] - vmin[j]) * rs_arg;
            vmin[j] -= vexp;
            vmax[j] += vexp;
            vdiff [j] = vmax[j] - vmin[j];
        }
    } else {
        // transpose
        std::vector<float> xt(n * d);
        for (size_t i = 1; i < n; i++) {
            const float *xi = x + i * d;
            for (size_t j = 0; j < d; j++) {
                xt[j * n + i] = xi[j];
            }
        }
        std::vector<float> trained_d(2);
#pragma omp parallel for
        for (size_t j = 0; j < d; j++) {
            train_Uniform(rs, rs_arg,
                          n, k, xt.data() + j * n,
                          trained_d);
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

struct SimilarityL2 {
    const float *y, *yi;

    explicit SimilarityL2 (const float * y): y(y) {}

    /******* scalar accumulator *******/

    float accu;

    void begin () {
        accu = 0;
        yi = y;
    }

    void add_component (float x) {
        float tmp = *yi++ - x;
        accu += tmp * tmp;
    }

    void add_component_2 (float x1, float x2) {
        float tmp = x1 - x2;
        accu += tmp * tmp;
    }

    float result () {
        return accu;
    }

#ifdef USE_AVX
    __m256 accu8;

    void begin_8 () {
        accu8 = _mm256_setzero_ps();
        yi = y;
    }

    void add_8_components (__m256 x) {
        __m256 yiv = _mm256_loadu_ps (yi);
        yi += 8;
        __m256 tmp = yiv - x;
        accu8 += tmp * tmp;
    }

    void add_8_components_2 (__m256 x, __m256 y) {
        __m256 tmp = y - x;
        accu8 += tmp * tmp;
    }

    float result_8 () {
        __m256 sum = _mm256_hadd_ps(accu8, accu8);
        __m256 sum2 = _mm256_hadd_ps(sum, sum);
        // now add the 0th and 4th component
        return
            _mm_cvtss_f32 (_mm256_castps256_ps128(sum2)) +
            _mm_cvtss_f32 (_mm256_extractf128_ps(sum2, 1));
    }
#endif

};


struct SimilarityIP {
    const float *y, *yi;
    /******* scalar accumulator *******/

    float accu;

    explicit SimilarityIP (const float * y):
        y (y) {}

    void begin () {
        accu = 0;
        yi = y;
    }

    void add_component (float x) {
        accu +=  *yi++ * x;
    }

    void add_component_2 (float x1, float x2) {
        accu +=  x1 * x2;
    }

    float result () {
        return accu;
    }

#ifdef USE_AVX

    __m256 accu8;

    void begin_8 () {
        accu8 = _mm256_setzero_ps();
        yi = y;
    }

    void add_8_components (__m256 x) {
        __m256 yiv = _mm256_loadu_ps (yi);
        yi += 8;
        accu8 += yiv * x;
    }

    void add_8_components_2 (__m256 x1, __m256 x2) {
        accu8 += x1 * x2;
    }

    float result_8 () {
        __m256 sum = _mm256_hadd_ps(accu8, accu8);
        __m256 sum2 = _mm256_hadd_ps(sum, sum);
        // now add the 0th and 4th component
        return
            _mm_cvtss_f32 (_mm256_castps256_ps128(sum2)) +
            _mm_cvtss_f32 (_mm256_extractf128_ps(sum2, 1));
    }
#endif
};


/*******************************************************************
 * DistanceComputer: combines a similarity and a quantizer to do
 * code-to-vector or code-to-code comparisons
 */


template<class Quantizer, class Similarity>
struct DCTemplate : ScalarQuantizer::DistanceComputer {

    Quantizer quant;

    DCTemplate(size_t d, const std::vector<float> &trained):
        quant(d, trained)
    {}

    float compute_distance (const float *x,
                            const uint8_t *code) override
    {
        Similarity sim(x);
        sim.begin();
        for (size_t i = 0; i < quant.d; i ++) {
            float xi = quant.reconstruct_component (code, i);
            sim.add_component (xi);
        }
        return sim.result();
    }

    float compute_code_distance (const uint8_t *code1,
                                 const uint8_t *code2) override
    {
        Similarity sim(nullptr);
        sim.begin ();
        for (size_t i = 0; i < quant.d; i ++) {
            float x1 = quant.reconstruct_component (code1, i);
            float x2 = quant.reconstruct_component (code2, i);
            sim.add_component_2 (x1, x2);
        }
        return sim.result ();
    }

};

#ifdef USE_AVX

template<class Quantizer, class Similarity>
struct DCTemplate_8 : ScalarQuantizer::DistanceComputer {

    Quantizer quant;

    DCTemplate_8(size_t d, const std::vector<float> &trained):
        quant(d, trained)
    {}

    float compute_distance (const float *x,
                            const uint8_t *code) override
    {
        Similarity sim(x);
        sim.begin_8();
        for (size_t i = 0; i < quant.d; i += 8) {
            __m256 xi = quant.reconstruct_8_components (code, i);
            sim.add_8_components (xi);
        }
        return sim.result_8();
    }

    float compute_code_distance (const uint8_t *code1,
                                 const uint8_t *code2) override
    {
        Similarity sim(nullptr);
        sim.begin_8 ();
        for (size_t i = 0; i < quant.d; i += 8) {
            __m256 x1 = quant.reconstruct_8_components (code1, i);
            __m256 x2 = quant.reconstruct_8_components (code2, i);
            sim.add_8_components_2 (x1, x2);
        }
        return sim.result_8 ();
    }

};


#endif



template<class Sim>
DistanceComputer *select_distance_computer (
          QuantizerType qtype,
          size_t d, const std::vector<float> & trained)
{
#ifdef USE_AVX
    if (d % 8 == 0) {
        switch(qtype) {
        case ScalarQuantizer::QT_8bit:
            return new DCTemplate_8<QuantizerNonUniform8
                                    <Codec8bit>, Sim>(d, trained);
        case ScalarQuantizer::QT_4bit:
            return new DCTemplate_8<QuantizerNonUniform8
                                    <Codec4bit>, Sim>(d, trained);
        case ScalarQuantizer::QT_8bit_uniform:
            return new DCTemplate_8<QuantizerUniform8
                                    <Codec8bit>, Sim>(d, trained);
        case ScalarQuantizer::QT_4bit_uniform:
            return new DCTemplate_8<QuantizerUniform8
                                    <Codec4bit>, Sim>(d, trained);
        }
    } else
#endif
    {
        switch(qtype) {
        case ScalarQuantizer::QT_8bit:
            return new DCTemplate<QuantizerNonUniform
                                  <Codec8bit>, Sim>(d, trained);
        case ScalarQuantizer::QT_4bit:
            return new DCTemplate<QuantizerNonUniform
                                  <Codec4bit>, Sim>(d, trained);
        case ScalarQuantizer::QT_8bit_uniform:
            return new DCTemplate<QuantizerUniform
                                  <Codec8bit>, Sim>(d, trained);
        case ScalarQuantizer::QT_4bit_uniform:
            return new DCTemplate<QuantizerUniform
                                  <Codec4bit>, Sim>(d, trained);
        }
    }
    FAISS_THROW_MSG ("unknown qtype");
    return nullptr;
}





} // anonymous namespace



/*******************************************************************
 * ScalarQuantizer implementation
 ********************************************************************/

ScalarQuantizer::ScalarQuantizer
          (size_t d, QuantizerType qtype):
              qtype (qtype), rangestat(RS_minmax), rangestat_arg(0), d (d)
{
    switch (qtype) {
    case QT_8bit: case QT_8bit_uniform:
        code_size = d;
        break;
    case QT_4bit: case QT_4bit_uniform:
        code_size = (d + 1) / 2;
        break;
    }

}

ScalarQuantizer::ScalarQuantizer ():
    qtype(QT_8bit),
    rangestat(RS_minmax), rangestat_arg(0), d (0), code_size(0)
{}

void ScalarQuantizer::train (size_t n, const float *x)
{
    int bit_per_dim =
        qtype == QT_4bit_uniform ? 4 :
        qtype == QT_4bit ? 4 :
        qtype == QT_8bit_uniform ? 8 :
        qtype == QT_8bit ? 8 : -1;

    switch (qtype) {
    case QT_4bit_uniform: case QT_8bit_uniform:
        train_Uniform (rangestat, rangestat_arg,
                       n * d, 1 << bit_per_dim, x, trained);
        break;
    case QT_4bit: case QT_8bit:
        train_NonUniform (rangestat, rangestat_arg,
                          n, d, 1 << bit_per_dim, x, trained);
        break;
    }
}

void ScalarQuantizer::compute_codes (const float * x,
                                     uint8_t * codes,
                                     size_t n) const
{
    Quantizer *squant = select_quantizer (*this);
    ScopeDeleter1<Quantizer> del(squant);
#pragma omp parallel for
    for (size_t i = 0; i < n; i++)
        squant->encode_vector (x + i * d, codes + i * code_size);
}

void ScalarQuantizer::decode (const uint8_t *codes, float *x, size_t n) const
{
    Quantizer *squant = select_quantizer (*this);
    ScopeDeleter1<Quantizer> del(squant);
#pragma omp parallel for
    for (size_t i = 0; i < n; i++)
        squant->decode_vector (codes + i * code_size, x + i * d);
}


ScalarQuantizer::DistanceComputer *ScalarQuantizer::get_distance_computer (
                                         MetricType metric)
    const
{
    if (metric == METRIC_L2) {
        return select_distance_computer<SimilarityL2>(qtype, d, trained);
    } else {
        return select_distance_computer<SimilarityIP>(qtype, d, trained);
    }
}


/*******************************************************************
 * IndexScalarQuantizer implementation
 ********************************************************************/

IndexScalarQuantizer::IndexScalarQuantizer
                      (int d, ScalarQuantizer::QuantizerType qtype,
                       MetricType metric):
          Index(d, metric),
          sq (d, qtype)
{
    is_trained = false;
    code_size = sq.code_size;
}


IndexScalarQuantizer::IndexScalarQuantizer ():
    IndexScalarQuantizer(0, ScalarQuantizer::QT_8bit)
{}

void IndexScalarQuantizer::train(idx_t n, const float* x)
{
    sq.train(n, x);
    is_trained = true;
}

void IndexScalarQuantizer::add(idx_t n, const float* x)
{
    FAISS_THROW_IF_NOT (is_trained);
    codes.resize ((n + ntotal) * code_size);
    sq.compute_codes (x, &codes[ntotal * code_size], n);
    ntotal += n;
}



namespace {

template<class C>
void search_flat_scalar_quantizer(
        const IndexScalarQuantizer & index,
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels)
{
    size_t code_size = index.code_size;
    size_t d = index.d;

#pragma omp parallel
    {
        DistanceComputer *dc =
            index.sq.get_distance_computer(index.metric_type);
        ScopeDeleter1<DistanceComputer> del(dc);

#pragma omp for
        for (size_t i = 0; i < n; i++) {
            idx_t *idxi = labels + i * k;
            float *simi = distances + i * k;
            heap_heapify<C> (k, simi, idxi);

            const float *xi = x + i * d;
            const uint8_t *ci = index.codes.data ();

            for (size_t j = 0; j < index.ntotal; j++) {
                float accu = dc->compute_distance(xi, ci);
                if (C::cmp (simi [0], accu)) {
                    heap_pop<C> (k, simi, idxi);
                    heap_push<C> (k, simi, idxi, accu, j);
                }
                ci += code_size;
            }
            heap_reorder<C> (k, simi, idxi);
        }
    }

};

}

void IndexScalarQuantizer::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels) const
{
    FAISS_THROW_IF_NOT (is_trained);
    if (metric_type == METRIC_L2) {
        search_flat_scalar_quantizer<CMax<float, idx_t> > (*this, n, x, k, distances, labels);
    } else {
        search_flat_scalar_quantizer<CMin<float, idx_t> > (*this, n, x, k, distances, labels);
    }
}

void IndexScalarQuantizer::reset()
{
    codes.clear();
    ntotal = 0;
}

void IndexScalarQuantizer::reconstruct_n(
             idx_t i0, idx_t ni, float* recons) const
{
    Quantizer *squant = select_quantizer (sq);
    ScopeDeleter1<Quantizer> del (squant);
    for (size_t i = 0; i < ni; i++) {
        squant->decode_vector(&codes[(i + i0) * code_size], recons + i * d);
    }
}

void IndexScalarQuantizer::reconstruct(idx_t key, float* recons) const
{
    reconstruct_n(key, 1, recons);
}


/*******************************************************************
 * IndexIVFScalarQuantizer implementation
 ********************************************************************/

IndexIVFScalarQuantizer::IndexIVFScalarQuantizer
          (Index *quantizer, size_t d, size_t nlist,
           QuantizerType qtype, MetricType metric):
              IndexIVF (quantizer, d, nlist, 0, metric),
              sq (d, qtype)
{
    code_size = sq.code_size;
    // was not known at construction time
    invlists->code_size = code_size;
    is_trained = false;
}

IndexIVFScalarQuantizer::IndexIVFScalarQuantizer ():
      IndexIVF ()
{}

void IndexIVFScalarQuantizer::train_residual (idx_t n, const float *x)
{
    long * idx = new long [n];
    ScopeDeleter<long> del (idx);
    quantizer->assign (n, x, idx);
    float *residuals = new float [n * d];
    ScopeDeleter<float> del2 (residuals);

#pragma omp parallel for
    for (idx_t i = 0; i < n; i++) {
        quantizer->compute_residual (x + i * d, residuals + i * d, idx[i]);
    }

    sq.train (n, residuals);

}


void IndexIVFScalarQuantizer::add_with_ids
       (idx_t n, const float * x, const long *xids)
{
    FAISS_THROW_IF_NOT (is_trained);
    long * idx = new long [n];
    ScopeDeleter<long> del (idx);
    quantizer->assign (n, x, idx);
    size_t nadd = 0;
    Quantizer *squant = select_quantizer (sq);
    ScopeDeleter1<Quantizer> del2 (squant);

#pragma omp parallel reduction(+: nadd)
    {
        std::vector<float> residual (d);
        std::vector<uint8_t> one_code (code_size);
        int nt = omp_get_num_threads();
        int rank = omp_get_thread_num();

        // each thread takes care of a subset of lists
        for (size_t i = 0; i < n; i++) {
            long list_no = idx [i];
            if (list_no >= 0 && list_no % nt == rank) {
                long id = xids ? xids[i] : ntotal + i;

                quantizer->compute_residual (
                      x + i * d, residual.data(), list_no);

                squant->encode_vector (residual.data(), one_code.data());

                invlists->add_entry (list_no, id, one_code.data());

                nadd++;

            }
        }
    }
    ntotal += nadd;
}

namespace {


void search_with_probes_ip (const IndexIVFScalarQuantizer & index,
                         const float *x,
                         const idx_t *cent_ids, const float *cent_dis,
                         DistanceComputer & dc,
                         int k, float *simi, idx_t *idxi,
                         bool store_pairs)
{
    int nprobe = index.nprobe;
    size_t code_size = index.code_size;
    size_t d = index.d;
    std::vector<float> decoded(d);
    minheap_heapify (k, simi, idxi);
    size_t nscan = 0;
    for (int i = 0; i < nprobe; i++) {
        idx_t list_no = cent_ids[i];
        if (list_no < 0) break;
        float accu0 = cent_dis[i];

        const size_t list_size = index.invlists->list_size (list_no);
        const uint8_t * codes = index.invlists->get_codes (list_no);
        const idx_t * ids =
            store_pairs ? nullptr : index.invlists->get_ids (list_no);

        SimilarityIP sim(x);

        for (size_t j = 0; j < list_size; j++) {

            float accu = accu0 + dc.compute_distance(x, codes);

            if (accu > simi [0]) {
                minheap_pop (k, simi, idxi);
                long id = store_pairs ? (list_no << 32 | j) : ids[j];
                minheap_push (k, simi, idxi, accu, id);
            }
            codes += code_size;
        }
        nscan += list_size;
        if (index.max_codes && nscan > index.max_codes)
            break;
    }
    minheap_reorder (k, simi, idxi);
}

void search_with_probes_L2 (const IndexIVFScalarQuantizer & index,
                            const float *x_in,
                            const idx_t *cent_ids,
                            const Index *quantizer,
                            DistanceComputer & dc,
                            int k, float *simi, idx_t *idxi,
                            bool store_pairs)
{
    int nprobe = index.nprobe;
    size_t code_size = index.code_size;
    size_t d = index.d;
    std::vector<float> x(d);
    maxheap_heapify (k, simi, idxi);
    size_t nscan = 0;
    for (int i = 0; i < nprobe; i++) {
        idx_t list_no = cent_ids[i];
        if (list_no < 0) break;

        const size_t list_size = index.invlists->list_size (list_no);
        const uint8_t * codes = index.invlists->get_codes (list_no);
        const idx_t * ids =
            store_pairs ? nullptr : index.invlists->get_ids (list_no);

        // shift of x_in wrt centroid
        quantizer->compute_residual (x_in, x.data(), list_no);

        for (size_t j = 0; j < list_size; j++) {

            float dis = dc.compute_distance (x.data(), codes);

            if (dis < simi [0]) {
                maxheap_pop (k, simi, idxi);
                long id = store_pairs ? (list_no << 32 | j) : ids[j];
                maxheap_push (k, simi, idxi, dis, id);
            }
            codes += code_size;
        }
        nscan += list_size;
        if (index.max_codes && nscan > index.max_codes)
            break;
    }
    maxheap_reorder (k, simi, idxi);
}

} // anonymous namespace

void IndexIVFScalarQuantizer::search_preassigned (
                             idx_t n, const float *x, idx_t k,
                             const idx_t *idx,
                             const float *dis,
                             float *distances, idx_t *labels,
                             bool store_pairs) const
{
    FAISS_THROW_IF_NOT (is_trained);


    if (metric_type == METRIC_INNER_PRODUCT) {
#pragma omp parallel
        {
            DistanceComputer *dc = sq.get_distance_computer (metric_type);
            ScopeDeleter1<DistanceComputer> del(dc);
#pragma omp for
            for (size_t i = 0; i < n; i++) {
                search_with_probes_ip (*this, x + i * d,
                                       idx + i * nprobe, dis + i * nprobe, *dc,
                                       k, distances + i * k, labels + i * k,
                                       store_pairs);
            }
        }
    } else {
#pragma omp parallel
        {
            DistanceComputer *dc = sq.get_distance_computer (metric_type);
            ScopeDeleter1<DistanceComputer> del(dc);
#pragma omp for
            for (size_t i = 0; i < n; i++) {
                search_with_probes_L2 (*this, x + i * d,
                                       idx + i * nprobe, quantizer, *dc,
                                       k, distances + i * k, labels + i * k,
                                       store_pairs);
            }
        }
    }

}

void IndexIVFScalarQuantizer::reconstruct_from_offset (long list_no,
                                                       long offset,
                                                       float* recons) const
{
    std::vector<float> centroid(d);
    quantizer->reconstruct (list_no, centroid.data());

    const uint8_t* code = invlists->get_single_code (list_no, offset);
    sq.decode (code, recons, 1);
    for (int i = 0; i < d; ++i) {
      recons[i] += centroid[i];
    }
}

} // namespace faiss
