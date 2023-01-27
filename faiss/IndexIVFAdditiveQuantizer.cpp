/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexIVFAdditiveQuantizer.h>

#include <algorithm>
#include <cmath>
#include <cstring>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/ResidualQuantizer.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/extra_distances.h>
#include <faiss/utils/utils.h>

namespace faiss {

/**************************************************************************************
 * IndexIVFAdditiveQuantizer
 **************************************************************************************/

IndexIVFAdditiveQuantizer::IndexIVFAdditiveQuantizer(
        AdditiveQuantizer* aq,
        Index* quantizer,
        size_t d,
        size_t nlist,
        MetricType metric)
        : IndexIVF(quantizer, d, nlist, 0, metric), aq(aq) {
    by_residual = true;
}

IndexIVFAdditiveQuantizer::IndexIVFAdditiveQuantizer(AdditiveQuantizer* aq)
        : IndexIVF(), aq(aq) {}

void IndexIVFAdditiveQuantizer::train_residual(idx_t n, const float* x) {
    const float* x_in = x;

    size_t max_train_points = 1024 * ((size_t)1 << aq->nbits[0]);
    // we need more data to train LSQ
    if (dynamic_cast<LocalSearchQuantizer*>(aq)) {
        max_train_points = 1024 * aq->M * ((size_t)1 << aq->nbits[0]);
    }

    x = fvecs_maybe_subsample(
            d, (size_t*)&n, max_train_points, x, verbose, 1234);
    ScopeDeleter<float> del_x(x_in == x ? nullptr : x);

    if (by_residual) {
        std::vector<idx_t> idx(n);
        quantizer->assign(n, x, idx.data());

        std::vector<float> residuals(n * d);
        quantizer->compute_residual_n(n, x, residuals.data(), idx.data());

        aq->train(n, residuals.data());
    } else {
        aq->train(n, x);
    }
}

void IndexIVFAdditiveQuantizer::encode_vectors(
        idx_t n,
        const float* x,
        const idx_t* list_nos,
        uint8_t* codes,
        bool include_listnos) const {
    FAISS_THROW_IF_NOT(is_trained);

    // first encode then possibly add listnos

    if (by_residual) {
        // subtract centroids
        std::vector<float> residuals(n * d);

#pragma omp parallel for if (n > 10000)
        for (idx_t i = 0; i < n; i++) {
            quantizer->compute_residual(
                    x + i * d,
                    residuals.data() + i * d,
                    list_nos[i] >= 0 ? list_nos[i] : 0);
        }
        aq->compute_codes(residuals.data(), codes, n);
    } else {
        aq->compute_codes(x, codes, n);
    }

    if (include_listnos) {
        // write back from the end, where there is enough space
        size_t coarse_size = coarse_code_size();
        for (idx_t i = n - 1; i >= 0; i--) {
            uint8_t* code = codes + i * (code_size + coarse_size);
            memmove(code + coarse_size, codes + i * code_size, code_size);
            encode_listno(list_nos[i], code);
        }
    }
}

void IndexIVFAdditiveQuantizer::sa_decode(
        idx_t n,
        const uint8_t* codes,
        float* x) const {
    const size_t coarse_size = coarse_code_size();

#pragma omp parallel if (n > 1000)
    {
        std::vector<float> residual(d);

#pragma omp for
        for (idx_t i = 0; i < n; i++) {
            const uint8_t* code = codes + i * (code_size + coarse_size);
            int64_t list_no = decode_listno(code);
            float* xi = x + i * d;
            aq->decode(code + coarse_size, xi, 1);
            if (by_residual) {
                quantizer->reconstruct(list_no, residual.data());
                for (size_t j = 0; j < d; j++) {
                    xi[j] += residual[j];
                }
            }
        }
    }
}

IndexIVFAdditiveQuantizer::~IndexIVFAdditiveQuantizer() {}

/*********************************************
 * AQInvertedListScanner
 *********************************************/

namespace {

using Search_type_t = AdditiveQuantizer::Search_type_t;

struct AQInvertedListScanner : InvertedListScanner {
    const IndexIVFAdditiveQuantizer& ia;
    const AdditiveQuantizer& aq;
    std::vector<float> tmp;

    AQInvertedListScanner(const IndexIVFAdditiveQuantizer& ia, bool store_pairs)
            : ia(ia), aq(*ia.aq) {
        this->store_pairs = store_pairs;
        this->code_size = ia.code_size;
        keep_max = ia.metric_type == METRIC_INNER_PRODUCT;
        tmp.resize(ia.d);
    }

    const float* q0;

    /// from now on we handle this query.
    void set_query(const float* query_vector) override {
        q0 = query_vector;
    }

    const float* q;
    /// following codes come from this inverted list
    void set_list(idx_t list_no, float coarse_dis) override {
        if (ia.metric_type == METRIC_L2 && ia.by_residual) {
            ia.quantizer->compute_residual(q0, tmp.data(), list_no);
            q = tmp.data();
        } else {
            q = q0;
        }
    }

    ~AQInvertedListScanner() {}
};

template <bool is_IP>
struct AQInvertedListScannerDecompress : AQInvertedListScanner {
    AQInvertedListScannerDecompress(
            const IndexIVFAdditiveQuantizer& ia,
            bool store_pairs)
            : AQInvertedListScanner(ia, store_pairs) {}

    float coarse_dis = 0;

    /// following codes come from this inverted list
    void set_list(idx_t list_no, float coarse_dis) override {
        AQInvertedListScanner::set_list(list_no, coarse_dis);
        if (ia.by_residual) {
            this->coarse_dis = coarse_dis;
        }
    }

    /// compute a single query-to-code distance
    float distance_to_code(const uint8_t* code) const final {
        std::vector<float> b(aq.d);
        aq.decode(code, b.data(), 1);
        FAISS_ASSERT(q);
        FAISS_ASSERT(b.data());

        return is_IP ? coarse_dis + fvec_inner_product(q, b.data(), aq.d)
                     : fvec_L2sqr(q, b.data(), aq.d);
    }

    ~AQInvertedListScannerDecompress() override {}
};

template <bool is_IP, Search_type_t search_type>
struct AQInvertedListScannerLUT : AQInvertedListScanner {
    std::vector<float> LUT, tmp;
    float distance_bias;

    AQInvertedListScannerLUT(
            const IndexIVFAdditiveQuantizer& ia,
            bool store_pairs)
            : AQInvertedListScanner(ia, store_pairs) {
        LUT.resize(aq.total_codebook_size);
        tmp.resize(ia.d);
        distance_bias = 0;
    }

    /// from now on we handle this query.
    void set_query(const float* query_vector) override {
        AQInvertedListScanner::set_query(query_vector);
        if (!is_IP && !ia.by_residual) {
            distance_bias = fvec_norm_L2sqr(query_vector, ia.d);
        }
    }

    /// following codes come from this inverted list
    void set_list(idx_t list_no, float coarse_dis) override {
        AQInvertedListScanner::set_list(list_no, coarse_dis);
        // TODO find a way to provide the nprobes together to do a matmul
        // +  precompute tables
        aq.compute_LUT(1, q, LUT.data());

        if (ia.by_residual) {
            distance_bias = coarse_dis;
        }
    }

    /// compute a single query-to-code distance
    float distance_to_code(const uint8_t* code) const final {
        return distance_bias +
                aq.compute_1_distance_LUT<is_IP, search_type>(code, LUT.data());
    }

    ~AQInvertedListScannerLUT() override {}
};

} // anonymous namespace

InvertedListScanner* IndexIVFAdditiveQuantizer::get_InvertedListScanner(
        bool store_pairs,
        const IDSelector* sel) const {
    FAISS_THROW_IF_NOT(!sel);
    if (metric_type == METRIC_INNER_PRODUCT) {
        if (aq->search_type == AdditiveQuantizer::ST_decompress) {
            return new AQInvertedListScannerDecompress<true>(
                    *this, store_pairs);
        } else {
            return new AQInvertedListScannerLUT<
                    true,
                    AdditiveQuantizer::ST_LUT_nonorm>(*this, store_pairs);
        }
    } else {
        switch (aq->search_type) {
            case AdditiveQuantizer::ST_decompress:
                return new AQInvertedListScannerDecompress<false>(
                        *this, store_pairs);
#define A(st)                                                              \
    case AdditiveQuantizer::st:                                            \
        return new AQInvertedListScannerLUT<false, AdditiveQuantizer::st>( \
                *this, store_pairs);
                A(ST_LUT_nonorm)
                // A(ST_norm_from_LUT)
                A(ST_norm_float)
                A(ST_norm_qint8)
                A(ST_norm_qint4)
                A(ST_norm_cqint4)
            case AdditiveQuantizer::ST_norm_lsq2x4:
            case AdditiveQuantizer::ST_norm_rq2x4:
                A(ST_norm_cqint8)
#undef A
            default:
                FAISS_THROW_FMT(
                        "search type %d not supported", aq->search_type);
        }
    }
}

/**************************************************************************************
 * IndexIVFResidualQuantizer
 **************************************************************************************/

IndexIVFResidualQuantizer::IndexIVFResidualQuantizer(
        Index* quantizer,
        size_t d,
        size_t nlist,
        const std::vector<size_t>& nbits,
        MetricType metric,
        Search_type_t search_type)
        : IndexIVFAdditiveQuantizer(&rq, quantizer, d, nlist, metric),
          rq(d, nbits, search_type) {
    code_size = invlists->code_size = rq.code_size;
}

IndexIVFResidualQuantizer::IndexIVFResidualQuantizer()
        : IndexIVFAdditiveQuantizer(&rq) {}

IndexIVFResidualQuantizer::IndexIVFResidualQuantizer(
        Index* quantizer,
        size_t d,
        size_t nlist,
        size_t M,     /* number of subquantizers */
        size_t nbits, /* number of bit per subvector index */
        MetricType metric,
        Search_type_t search_type)
        : IndexIVFResidualQuantizer(
                  quantizer,
                  d,
                  nlist,
                  std::vector<size_t>(M, nbits),
                  metric,
                  search_type) {}

IndexIVFResidualQuantizer::~IndexIVFResidualQuantizer() {}

/**************************************************************************************
 * IndexIVFLocalSearchQuantizer
 **************************************************************************************/

IndexIVFLocalSearchQuantizer::IndexIVFLocalSearchQuantizer(
        Index* quantizer,
        size_t d,
        size_t nlist,
        size_t M,     /* number of subquantizers */
        size_t nbits, /* number of bit per subvector index */
        MetricType metric,
        Search_type_t search_type)
        : IndexIVFAdditiveQuantizer(&lsq, quantizer, d, nlist, metric),
          lsq(d, M, nbits, search_type) {
    code_size = invlists->code_size = lsq.code_size;
}

IndexIVFLocalSearchQuantizer::IndexIVFLocalSearchQuantizer()
        : IndexIVFAdditiveQuantizer(&lsq) {}

IndexIVFLocalSearchQuantizer::~IndexIVFLocalSearchQuantizer() {}

/**************************************************************************************
 * IndexIVFProductResidualQuantizer
 **************************************************************************************/

IndexIVFProductResidualQuantizer::IndexIVFProductResidualQuantizer(
        Index* quantizer,
        size_t d,
        size_t nlist,
        size_t nsplits,
        size_t Msub,
        size_t nbits,
        MetricType metric,
        Search_type_t search_type)
        : IndexIVFAdditiveQuantizer(&prq, quantizer, d, nlist, metric),
          prq(d, nsplits, Msub, nbits, search_type) {
    code_size = invlists->code_size = prq.code_size;
}

IndexIVFProductResidualQuantizer::IndexIVFProductResidualQuantizer()
        : IndexIVFAdditiveQuantizer(&prq) {}

IndexIVFProductResidualQuantizer::~IndexIVFProductResidualQuantizer() {}

/**************************************************************************************
 * IndexIVFProductLocalSearchQuantizer
 **************************************************************************************/

IndexIVFProductLocalSearchQuantizer::IndexIVFProductLocalSearchQuantizer(
        Index* quantizer,
        size_t d,
        size_t nlist,
        size_t nsplits,
        size_t Msub,
        size_t nbits,
        MetricType metric,
        Search_type_t search_type)
        : IndexIVFAdditiveQuantizer(&plsq, quantizer, d, nlist, metric),
          plsq(d, nsplits, Msub, nbits, search_type) {
    code_size = invlists->code_size = plsq.code_size;
}

IndexIVFProductLocalSearchQuantizer::IndexIVFProductLocalSearchQuantizer()
        : IndexIVFAdditiveQuantizer(&plsq) {}

IndexIVFProductLocalSearchQuantizer::~IndexIVFProductLocalSearchQuantizer() {}

} // namespace faiss
