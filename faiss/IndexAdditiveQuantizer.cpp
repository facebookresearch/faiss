/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexAdditiveQuantizer.h>

#include <algorithm>
#include <cmath>
#include <cstring>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/ResidualQuantizer.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/extra_distances.h>

namespace faiss {

/**************************************************************************************
 * IndexAdditiveQuantizer
 **************************************************************************************/

IndexAdditiveQuantizer::IndexAdditiveQuantizer(
        idx_t d,
        AdditiveQuantizer* aq,
        MetricType metric)
        : IndexFlatCodes(aq->code_size, d, metric), aq(aq) {
    FAISS_THROW_IF_NOT(metric == METRIC_INNER_PRODUCT || metric == METRIC_L2);
}

namespace {

/************************************************************
 * DistanceComputer implementation
 ************************************************************/

template <class VectorDistance>
struct AQDistanceComputerDecompress : FlatCodesDistanceComputer {
    std::vector<float> tmp;
    const AdditiveQuantizer& aq;
    VectorDistance vd;
    size_t d;

    AQDistanceComputerDecompress(
            const IndexAdditiveQuantizer& iaq,
            VectorDistance vd)
            : FlatCodesDistanceComputer(iaq.codes.data(), iaq.code_size),
              tmp(iaq.d * 2),
              aq(*iaq.aq),
              vd(vd),
              d(iaq.d) {}

    const float* q;
    void set_query(const float* x) final {
        q = x;
    }

    float symmetric_dis(idx_t i, idx_t j) final {
        aq.decode(codes + i * d, tmp.data(), 1);
        aq.decode(codes + j * d, tmp.data() + d, 1);
        return vd(tmp.data(), tmp.data() + d);
    }

    float distance_to_code(const uint8_t* code) final {
        aq.decode(code, tmp.data(), 1);
        return vd(q, tmp.data());
    }

    virtual ~AQDistanceComputerDecompress() = default;
};

template <bool is_IP, AdditiveQuantizer::Search_type_t st>
struct AQDistanceComputerLUT : FlatCodesDistanceComputer {
    std::vector<float> LUT;
    const AdditiveQuantizer& aq;
    size_t d;

    explicit AQDistanceComputerLUT(const IndexAdditiveQuantizer& iaq)
            : FlatCodesDistanceComputer(iaq.codes.data(), iaq.code_size),
              LUT(iaq.aq->total_codebook_size + iaq.d * 2),
              aq(*iaq.aq),
              d(iaq.d) {}

    float bias;
    void set_query(const float* x) final {
        // this is quite sub-optimal for multiple queries
        aq.compute_LUT(1, x, LUT.data());
        if (is_IP) {
            bias = 0;
        } else {
            bias = fvec_norm_L2sqr(x, d);
        }
    }

    float symmetric_dis(idx_t i, idx_t j) final {
        float* tmp = LUT.data();
        aq.decode(codes + i * d, tmp, 1);
        aq.decode(codes + j * d, tmp + d, 1);
        return fvec_L2sqr(tmp, tmp + d, d);
    }

    float distance_to_code(const uint8_t* code) final {
        return bias + aq.compute_1_distance_LUT<is_IP, st>(code, LUT.data());
    }

    virtual ~AQDistanceComputerLUT() = default;
};

/************************************************************
 * scanning implementation for search
 ************************************************************/

template <class VectorDistance, class BlockResultHandler>
void search_with_decompress(
        const IndexAdditiveQuantizer& ir,
        const float* xq,
        VectorDistance& vd,
        BlockResultHandler& res) {
    const uint8_t* codes = ir.codes.data();
    size_t ntotal = ir.ntotal;
    size_t code_size = ir.code_size;
    const AdditiveQuantizer* aq = ir.aq;

    using SingleResultHandler =
            typename BlockResultHandler::SingleResultHandler;

#pragma omp parallel for if (res.nq > 100)
    for (int64_t q = 0; q < res.nq; q++) {
        SingleResultHandler resi(res);
        resi.begin(q);
        std::vector<float> tmp(ir.d);
        const float* x = xq + ir.d * q;
        for (size_t i = 0; i < ntotal; i++) {
            aq->decode(codes + i * code_size, tmp.data(), 1);
            float dis = vd(x, tmp.data());
            resi.add_result(dis, i);
        }
        resi.end();
    }
}

template <
        bool is_IP,
        AdditiveQuantizer::Search_type_t st,
        class BlockResultHandler>
void search_with_LUT(
        const IndexAdditiveQuantizer& ir,
        const float* xq,
        BlockResultHandler& res) {
    const AdditiveQuantizer& aq = *ir.aq;
    const uint8_t* codes = ir.codes.data();
    size_t ntotal = ir.ntotal;
    size_t code_size = aq.code_size;
    size_t nq = res.nq;
    size_t d = ir.d;

    using SingleResultHandler =
            typename BlockResultHandler::SingleResultHandler;
    std::unique_ptr<float[]> LUT(new float[nq * aq.total_codebook_size]);

    aq.compute_LUT(nq, xq, LUT.get());

#pragma omp parallel for if (nq > 100)
    for (int64_t q = 0; q < nq; q++) {
        SingleResultHandler resi(res);
        resi.begin(q);
        std::vector<float> tmp(aq.d);
        const float* LUT_q = LUT.get() + aq.total_codebook_size * q;
        float bias = 0;
        if (!is_IP) { // the LUT function returns ||y||^2 - 2 * <x, y>, need to
                      // add ||x||^2
            bias = fvec_norm_L2sqr(xq + q * d, d);
        }
        for (size_t i = 0; i < ntotal; i++) {
            float dis = aq.compute_1_distance_LUT<is_IP, st>(
                    codes + i * code_size, LUT_q);
            resi.add_result(dis + bias, i);
        }
        resi.end();
    }
}

} // anonymous namespace

FlatCodesDistanceComputer* IndexAdditiveQuantizer::
        get_FlatCodesDistanceComputer() const {
    if (aq->search_type == AdditiveQuantizer::ST_decompress) {
        if (metric_type == METRIC_L2) {
            using VD = VectorDistance<METRIC_L2>;
            VD vd = {size_t(d), metric_arg};
            return new AQDistanceComputerDecompress<VD>(*this, vd);
        } else if (metric_type == METRIC_INNER_PRODUCT) {
            using VD = VectorDistance<METRIC_INNER_PRODUCT>;
            VD vd = {size_t(d), metric_arg};
            return new AQDistanceComputerDecompress<VD>(*this, vd);
        } else {
            FAISS_THROW_MSG("unsupported metric");
        }
    } else {
        if (metric_type == METRIC_INNER_PRODUCT) {
            return new AQDistanceComputerLUT<
                    true,
                    AdditiveQuantizer::ST_LUT_nonorm>(*this);
        } else {
            switch (aq->search_type) {
#define DISPATCH(st)                                                           \
    case AdditiveQuantizer::st:                                                \
        return new AQDistanceComputerLUT<false, AdditiveQuantizer::st>(*this); \
        break;
                DISPATCH(ST_norm_float)
                DISPATCH(ST_LUT_nonorm)
                DISPATCH(ST_norm_qint8)
                DISPATCH(ST_norm_qint4)
                DISPATCH(ST_norm_cqint4)
                case AdditiveQuantizer::ST_norm_cqint8:
                case AdditiveQuantizer::ST_norm_lsq2x4:
                case AdditiveQuantizer::ST_norm_rq2x4:
                    return new AQDistanceComputerLUT<
                            false,
                            AdditiveQuantizer::ST_norm_cqint8>(*this);
                    break;
#undef DISPATCH
                default:
                    FAISS_THROW_FMT(
                            "search type %d not supported", aq->search_type);
            }
        }
    }
}

void IndexAdditiveQuantizer::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(
            !params, "search params not supported for this index");

    if (aq->search_type == AdditiveQuantizer::ST_decompress) {
        if (metric_type == METRIC_L2) {
            using VD = VectorDistance<METRIC_L2>;
            VD vd = {size_t(d), metric_arg};
            HeapBlockResultHandler<VD::C> rh(n, distances, labels, k);
            search_with_decompress(*this, x, vd, rh);
        } else if (metric_type == METRIC_INNER_PRODUCT) {
            using VD = VectorDistance<METRIC_INNER_PRODUCT>;
            VD vd = {size_t(d), metric_arg};
            HeapBlockResultHandler<VD::C> rh(n, distances, labels, k);
            search_with_decompress(*this, x, vd, rh);
        }
    } else {
        if (metric_type == METRIC_INNER_PRODUCT) {
            HeapBlockResultHandler<CMin<float, idx_t>> rh(
                    n, distances, labels, k);
            search_with_LUT<true, AdditiveQuantizer::ST_LUT_nonorm>(
                    *this, x, rh);
        } else {
            HeapBlockResultHandler<CMax<float, idx_t>> rh(
                    n, distances, labels, k);
            switch (aq->search_type) {
#define DISPATCH(st)                                                 \
    case AdditiveQuantizer::st:                                      \
        search_with_LUT<false, AdditiveQuantizer::st>(*this, x, rh); \
        break;
                DISPATCH(ST_norm_float)
                DISPATCH(ST_LUT_nonorm)
                DISPATCH(ST_norm_qint8)
                DISPATCH(ST_norm_qint4)
                DISPATCH(ST_norm_cqint4)
                DISPATCH(ST_norm_from_LUT)
                case AdditiveQuantizer::ST_norm_cqint8:
                case AdditiveQuantizer::ST_norm_lsq2x4:
                case AdditiveQuantizer::ST_norm_rq2x4:
                    search_with_LUT<false, AdditiveQuantizer::ST_norm_cqint8>(
                            *this, x, rh);
                    break;
#undef DISPATCH
                default:
                    FAISS_THROW_FMT(
                            "search type %d not supported", aq->search_type);
            }
        }
    }
}

void IndexAdditiveQuantizer::sa_encode(idx_t n, const float* x, uint8_t* bytes)
        const {
    return aq->compute_codes(x, bytes, n);
}

void IndexAdditiveQuantizer::sa_decode(idx_t n, const uint8_t* bytes, float* x)
        const {
    return aq->decode(bytes, x, n);
}

/**************************************************************************************
 * IndexResidualQuantizer
 **************************************************************************************/

IndexResidualQuantizer::IndexResidualQuantizer(
        int d,        ///< dimensionality of the input vectors
        size_t M,     ///< number of subquantizers
        size_t nbits, ///< number of bit per subvector index
        MetricType metric,
        Search_type_t search_type)
        : IndexResidualQuantizer(
                  d,
                  std::vector<size_t>(M, nbits),
                  metric,
                  search_type) {}

IndexResidualQuantizer::IndexResidualQuantizer(
        int d,
        const std::vector<size_t>& nbits,
        MetricType metric,
        Search_type_t search_type)
        : IndexAdditiveQuantizer(d, &rq, metric), rq(d, nbits, search_type) {
    code_size = rq.code_size;
    is_trained = false;
}

IndexResidualQuantizer::IndexResidualQuantizer()
        : IndexResidualQuantizer(0, 0, 0) {}

void IndexResidualQuantizer::train(idx_t n, const float* x) {
    rq.train(n, x);
    is_trained = true;
}

/**************************************************************************************
 * IndexLocalSearchQuantizer
 **************************************************************************************/

IndexLocalSearchQuantizer::IndexLocalSearchQuantizer(
        int d,
        size_t M,     ///< number of subquantizers
        size_t nbits, ///< number of bit per subvector index
        MetricType metric,
        Search_type_t search_type)
        : IndexAdditiveQuantizer(d, &lsq, metric),
          lsq(d, M, nbits, search_type) {
    code_size = lsq.code_size;
    is_trained = false;
}

IndexLocalSearchQuantizer::IndexLocalSearchQuantizer()
        : IndexLocalSearchQuantizer(0, 0, 0) {}

void IndexLocalSearchQuantizer::train(idx_t n, const float* x) {
    lsq.train(n, x);
    is_trained = true;
}

/**************************************************************************************
 * IndexProductResidualQuantizer
 **************************************************************************************/

IndexProductResidualQuantizer::IndexProductResidualQuantizer(
        int d,          ///< dimensionality of the input vectors
        size_t nsplits, ///< number of residual quantizers
        size_t Msub,    ///< number of subquantizers per RQ
        size_t nbits,   ///< number of bit per subvector index
        MetricType metric,
        Search_type_t search_type)
        : IndexAdditiveQuantizer(d, &prq, metric),
          prq(d, nsplits, Msub, nbits, search_type) {
    code_size = prq.code_size;
    is_trained = false;
}

IndexProductResidualQuantizer::IndexProductResidualQuantizer()
        : IndexProductResidualQuantizer(0, 0, 0, 0) {}

void IndexProductResidualQuantizer::train(idx_t n, const float* x) {
    prq.train(n, x);
    is_trained = true;
}

/**************************************************************************************
 * IndexProductLocalSearchQuantizer
 **************************************************************************************/

IndexProductLocalSearchQuantizer::IndexProductLocalSearchQuantizer(
        int d,          ///< dimensionality of the input vectors
        size_t nsplits, ///< number of local search quantizers
        size_t Msub,    ///< number of subquantizers per LSQ
        size_t nbits,   ///< number of bit per subvector index
        MetricType metric,
        Search_type_t search_type)
        : IndexAdditiveQuantizer(d, &plsq, metric),
          plsq(d, nsplits, Msub, nbits, search_type) {
    code_size = plsq.code_size;
    is_trained = false;
}

IndexProductLocalSearchQuantizer::IndexProductLocalSearchQuantizer()
        : IndexProductLocalSearchQuantizer(0, 0, 0, 0) {}

void IndexProductLocalSearchQuantizer::train(idx_t n, const float* x) {
    plsq.train(n, x);
    is_trained = true;
}

/**************************************************************************************
 * AdditiveCoarseQuantizer
 **************************************************************************************/

AdditiveCoarseQuantizer::AdditiveCoarseQuantizer(
        idx_t d,
        AdditiveQuantizer* aq,
        MetricType metric)
        : Index(d, metric), aq(aq) {}

void AdditiveCoarseQuantizer::add(idx_t, const float*) {
    FAISS_THROW_MSG("not applicable");
}

void AdditiveCoarseQuantizer::reconstruct(idx_t key, float* recons) const {
    aq->decode_64bit(key, recons);
}

void AdditiveCoarseQuantizer::reset() {
    FAISS_THROW_MSG("not applicable");
}

void AdditiveCoarseQuantizer::train(idx_t n, const float* x) {
    if (verbose) {
        printf("AdditiveCoarseQuantizer::train: training on %zd vectors\n",
               size_t(n));
    }
    size_t norms_size = sizeof(float) << aq->tot_bits;

    FAISS_THROW_IF_NOT_MSG(
            norms_size <= aq->max_mem_distances,
            "the RCQ norms matrix will become too large, please reduce the number of quantization steps");

    aq->train(n, x);
    is_trained = true;
    ntotal = (idx_t)1 << aq->tot_bits;

    if (metric_type == METRIC_L2) {
        if (verbose) {
            printf("AdditiveCoarseQuantizer::train: computing centroid norms for %zd centroids\n",
                   size_t(ntotal));
        }
        // this is not necessary for the residualcoarsequantizer when
        // using beam search. We'll see if the memory overhead is too high
        centroid_norms.resize(ntotal);
        aq->compute_centroid_norms(centroid_norms.data());
    }
}

void AdditiveCoarseQuantizer::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT_MSG(
            !params, "search params not supported for this index");

    if (metric_type == METRIC_INNER_PRODUCT) {
        aq->knn_centroids_inner_product(n, x, k, distances, labels);
    } else if (metric_type == METRIC_L2) {
        FAISS_THROW_IF_NOT(centroid_norms.size() == ntotal);
        aq->knn_centroids_L2(n, x, k, distances, labels, centroid_norms.data());
    }
}

/**************************************************************************************
 * ResidualCoarseQuantizer
 **************************************************************************************/

ResidualCoarseQuantizer::ResidualCoarseQuantizer(
        int d, ///< dimensionality of the input vectors
        const std::vector<size_t>& nbits,
        MetricType metric)
        : AdditiveCoarseQuantizer(d, &rq, metric), rq(d, nbits) {
    FAISS_THROW_IF_NOT(rq.tot_bits <= 63);
    is_trained = false;
}

ResidualCoarseQuantizer::ResidualCoarseQuantizer(
        int d,
        size_t M,     ///< number of subquantizers
        size_t nbits, ///< number of bit per subvector index
        MetricType metric)
        : ResidualCoarseQuantizer(d, std::vector<size_t>(M, nbits), metric) {}

ResidualCoarseQuantizer::ResidualCoarseQuantizer()
        : ResidualCoarseQuantizer(0, 0, 0) {}

void ResidualCoarseQuantizer::set_beam_factor(float new_beam_factor) {
    beam_factor = new_beam_factor;
    if (new_beam_factor > 0) {
        FAISS_THROW_IF_NOT(new_beam_factor >= 1.0);
        if (rq.codebook_cross_products.size() == 0) {
            rq.compute_codebook_tables();
        }
        return;
    } else {
        // new_beam_factor = -1: exhaustive computation.
        // Does not use the cross_products
        rq.codebook_cross_products.resize(0);
        // but the centroid norms are necessary!
        if (metric_type == METRIC_L2 && ntotal != centroid_norms.size()) {
            if (verbose) {
                printf("AdditiveCoarseQuantizer::train: computing centroid norms for %zd centroids\n",
                       size_t(ntotal));
            }
            centroid_norms.resize(ntotal);
            aq->compute_centroid_norms(centroid_norms.data());
        }
    }
}

void ResidualCoarseQuantizer::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params_in) const {
    float beam_factor_2 = this->beam_factor;
    if (params_in) {
        auto params =
                dynamic_cast<const SearchParametersResidualCoarseQuantizer*>(
                        params_in);
        FAISS_THROW_IF_NOT_MSG(
                params,
                "need SearchParametersResidualCoarseQuantizer parameters");
        beam_factor_2 = params->beam_factor;
    }

    if (beam_factor_2 < 0) {
        AdditiveCoarseQuantizer::search(n, x, k, distances, labels);
        return;
    }

    int beam_size = int(k * beam_factor_2);
    if (beam_size > ntotal) {
        beam_size = ntotal;
    }
    size_t memory_per_point = rq.memory_per_point(beam_size);

    /*

    printf("mem per point %ld n=%d max_mem_distance=%ld mem_kb=%zd\n",
        memory_per_point, int(n), rq.max_mem_distances, get_mem_usage_kb());
    */
    if (n > 1 && memory_per_point * n > rq.max_mem_distances) {
        // then split queries to reduce temp memory
        idx_t bs = rq.max_mem_distances / memory_per_point;
        if (bs == 0) {
            bs = 1; // otherwise we can't do much
        }
        if (verbose) {
            printf("ResidualCoarseQuantizer::search: run %d searches in batches of size %d\n",
                   int(n),
                   int(bs));
        }
        for (idx_t i0 = 0; i0 < n; i0 += bs) {
            idx_t i1 = std::min(n, i0 + bs);
            search(i1 - i0,
                   x + i0 * d,
                   k,
                   distances + i0 * k,
                   labels + i0 * k,
                   params_in);
            InterruptCallback::check();
        }
        return;
    }

    std::vector<int32_t> codes(beam_size * rq.M * n);
    std::vector<float> beam_distances(n * beam_size);

    rq.refine_beam(
            n, 1, x, beam_size, codes.data(), nullptr, beam_distances.data());

    // pack int32 table
#pragma omp parallel for if (n > 4000)
    for (idx_t i = 0; i < n; i++) {
        memcpy(distances + i * k,
               beam_distances.data() + beam_size * i,
               k * sizeof(distances[0]));

        const int32_t* codes_i = codes.data() + beam_size * i * rq.M;
        for (idx_t j = 0; j < k; j++) {
            idx_t l = 0;
            int shift = 0;
            for (int m = 0; m < rq.M; m++) {
                l |= (*codes_i++) << shift;
                shift += rq.nbits[m];
            }
            labels[i * k + j] = l;
        }
    }
}

void ResidualCoarseQuantizer::initialize_from(
        const ResidualCoarseQuantizer& other) {
    FAISS_THROW_IF_NOT(rq.M <= other.rq.M);
    rq.initialize_from(other.rq);
    set_beam_factor(other.beam_factor);
    is_trained = other.is_trained;
    ntotal = (idx_t)1 << aq->tot_bits;
}

/**************************************************************************************
 * LocalSearchCoarseQuantizer
 **************************************************************************************/

LocalSearchCoarseQuantizer::LocalSearchCoarseQuantizer(
        int d,        ///< dimensionality of the input vectors
        size_t M,     ///< number of subquantizers
        size_t nbits, ///< number of bit per subvector index
        MetricType metric)
        : AdditiveCoarseQuantizer(d, &lsq, metric), lsq(d, M, nbits) {
    FAISS_THROW_IF_NOT(lsq.tot_bits <= 63);
    is_trained = false;
}

LocalSearchCoarseQuantizer::LocalSearchCoarseQuantizer() {
    aq = &lsq;
}

} // namespace faiss
