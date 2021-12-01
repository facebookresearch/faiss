/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// quiet the noise
// clang-format off

#include <faiss/IndexAdditiveQuantizer.h>

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
 * IndexAdditiveQuantizer
 **************************************************************************************/

IndexAdditiveQuantizer::IndexAdditiveQuantizer(
            idx_t d,
            AdditiveQuantizer* aq,
            MetricType metric):
        IndexFlatCodes(aq->code_size, d, metric), aq(aq)
{
    FAISS_THROW_IF_NOT(metric == METRIC_INNER_PRODUCT || metric == METRIC_L2);
}


namespace {

template <class VectorDistance, class ResultHandler>
void search_with_decompress(
        const IndexAdditiveQuantizer& ir,
        const float* xq,
        VectorDistance& vd,
        ResultHandler& res) {
    const uint8_t* codes = ir.codes.data();
    size_t ntotal = ir.ntotal;
    size_t code_size = ir.code_size;
    const AdditiveQuantizer *aq = ir.aq;

    using SingleResultHandler = typename ResultHandler::SingleResultHandler;

#pragma omp parallel for if(res.nq > 100)
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

template<bool is_IP, AdditiveQuantizer::Search_type_t st, class ResultHandler>
void search_with_LUT(
        const IndexAdditiveQuantizer& ir,
        const float* xq,
        ResultHandler& res)
{
    const AdditiveQuantizer & aq = *ir.aq;
    const uint8_t* codes = ir.codes.data();
    size_t ntotal = ir.ntotal;
    size_t code_size = aq.code_size;
    size_t nq = res.nq;
    size_t d = ir.d;

    using SingleResultHandler = typename ResultHandler::SingleResultHandler;
    std::unique_ptr<float []> LUT(new float[nq * aq.total_codebook_size]);

    aq.compute_LUT(nq, xq, LUT.get());

#pragma omp parallel for if(nq > 100)
    for (int64_t q = 0; q < nq; q++) {
        SingleResultHandler resi(res);
        resi.begin(q);
        std::vector<float> tmp(aq.d);
        const float *LUT_q = LUT.get() + aq.total_codebook_size * q;
        float bias = 0;
        if (!is_IP) { // the LUT function returns ||y||^2 - 2 * <x, y>, need to add ||x||^2
            bias = fvec_norm_L2sqr(xq + q * d, d);
        }
        for (size_t i = 0; i < ntotal; i++) {
            float dis = aq.compute_1_distance_LUT<is_IP, st>(
                codes + i * code_size,
                LUT_q
            );
            resi.add_result(dis + bias, i);
        }
        resi.end();
    }

}


} // anonymous namespace

void IndexAdditiveQuantizer::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels) const {
    if (aq->search_type == AdditiveQuantizer::ST_decompress) {
        if (metric_type == METRIC_L2) {
            using VD = VectorDistance<METRIC_L2>;
            VD vd = {size_t(d), metric_arg};
            HeapResultHandler<VD::C> rh(n, distances, labels, k);
            search_with_decompress(*this, x, vd, rh);
        } else if (metric_type == METRIC_INNER_PRODUCT) {
            using VD = VectorDistance<METRIC_INNER_PRODUCT>;
            VD vd = {size_t(d), metric_arg};
            HeapResultHandler<VD::C> rh(n, distances, labels, k);
            search_with_decompress(*this, x, vd, rh);
        }
    } else {
        if (metric_type == METRIC_INNER_PRODUCT) {
            HeapResultHandler<CMin<float, idx_t> > rh(n, distances, labels, k);
            search_with_LUT<true, AdditiveQuantizer::ST_LUT_nonorm> (*this, x, rh);
        } else {
            HeapResultHandler<CMax<float, idx_t> > rh(n, distances, labels, k);

            if (aq->search_type == AdditiveQuantizer::ST_norm_float) {
                search_with_LUT<false, AdditiveQuantizer::ST_norm_float> (*this, x, rh);
            } else if (aq->search_type == AdditiveQuantizer::ST_LUT_nonorm) {
                search_with_LUT<false, AdditiveQuantizer::ST_norm_float> (*this, x, rh);
            } else if (aq->search_type == AdditiveQuantizer::ST_norm_qint8) {
                search_with_LUT<false, AdditiveQuantizer::ST_norm_qint8> (*this, x, rh);
            } else if (aq->search_type == AdditiveQuantizer::ST_norm_qint4) {
                search_with_LUT<false, AdditiveQuantizer::ST_norm_qint4> (*this, x, rh);
            } else if (aq->search_type == AdditiveQuantizer::ST_norm_cqint8) {
                search_with_LUT<false, AdditiveQuantizer::ST_norm_cqint8> (*this, x, rh);
            } else if (aq->search_type == AdditiveQuantizer::ST_norm_cqint4) {
                search_with_LUT<false, AdditiveQuantizer::ST_norm_cqint4> (*this, x, rh);
            } else {
                FAISS_THROW_FMT("search type %d not supported", aq->search_type);
            }
        }

    }
}

void IndexAdditiveQuantizer::sa_encode(idx_t n, const float* x, uint8_t* bytes) const {
    return aq->compute_codes(x, bytes, n);
}

void IndexAdditiveQuantizer::sa_decode(idx_t n, const uint8_t* bytes, float* x) const {
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
        : IndexResidualQuantizer(d, std::vector<size_t>(M, nbits), metric, search_type) {
}

IndexResidualQuantizer::IndexResidualQuantizer(
        int d,
        const std::vector<size_t>& nbits,
        MetricType metric,
        Search_type_t search_type)
        : IndexAdditiveQuantizer(d, &rq, metric), rq(d, nbits, search_type) {
    code_size = rq.code_size;
    is_trained = false;
}

IndexResidualQuantizer::IndexResidualQuantizer() : IndexResidualQuantizer(0, 0, 0) {}

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
        : IndexAdditiveQuantizer(d, &lsq, metric), lsq(d, M, nbits, search_type) {
    code_size = lsq.code_size;
    is_trained = false;
}

IndexLocalSearchQuantizer::IndexLocalSearchQuantizer() : IndexLocalSearchQuantizer(0, 0, 0) {}

void IndexLocalSearchQuantizer::train(idx_t n, const float* x) {
    lsq.train(n, x);
    is_trained = true;
}

/**************************************************************************************
 * AdditiveCoarseQuantizer
 **************************************************************************************/

AdditiveCoarseQuantizer::AdditiveCoarseQuantizer(
            idx_t d,
            AdditiveQuantizer* aq,
            MetricType metric):
        Index(d, metric), aq(aq)
{}

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
        printf("AdditiveCoarseQuantizer::train: training on %zd vectors\n", size_t(n));
    }
    aq->train(n, x);
    is_trained = true;
    ntotal = (idx_t)1 << aq->tot_bits;

    if (metric_type == METRIC_L2) {
        if (verbose) {
            printf("AdditiveCoarseQuantizer::train: computing centroid norms for %zd centroids\n", size_t(ntotal));
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
        idx_t* labels) const {
    if (metric_type == METRIC_INNER_PRODUCT) {
        aq->knn_centroids_inner_product(n, x, k, distances, labels);
    } else if (metric_type == METRIC_L2) {
        FAISS_THROW_IF_NOT(centroid_norms.size() == ntotal);
        aq->knn_centroids_L2(
                n, x, k, distances, labels, centroid_norms.data());
    }
}

/**************************************************************************************
 * ResidualCoarseQuantizer
 **************************************************************************************/

ResidualCoarseQuantizer::ResidualCoarseQuantizer(
        int d,        ///< dimensionality of the input vectors
        const std::vector<size_t>& nbits,
        MetricType metric)
        : AdditiveCoarseQuantizer(d, &rq, metric), rq(d, nbits), beam_factor(4.0) {
    FAISS_THROW_IF_NOT(rq.tot_bits <= 63);
    is_trained = false;
}

ResidualCoarseQuantizer::ResidualCoarseQuantizer(
        int d,
        size_t M,     ///< number of subquantizers
        size_t nbits, ///< number of bit per subvector index
        MetricType metric)
        : ResidualCoarseQuantizer(d, std::vector<size_t>(M, nbits), metric) {}

ResidualCoarseQuantizer::ResidualCoarseQuantizer(): ResidualCoarseQuantizer(0, 0, 0) {}



void ResidualCoarseQuantizer::set_beam_factor(float new_beam_factor) {
    beam_factor = new_beam_factor;
    if (new_beam_factor > 0) {
        FAISS_THROW_IF_NOT(new_beam_factor >= 1.0);
        return;
    } else if (metric_type == METRIC_L2 && ntotal != centroid_norms.size()) {
        if (verbose) {
            printf("AdditiveCoarseQuantizer::train: computing centroid norms for %zd centroids\n", size_t(ntotal));
        }
        centroid_norms.resize(ntotal);
        aq->compute_centroid_norms(centroid_norms.data());
    }
}

void ResidualCoarseQuantizer::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels) const {
    if (beam_factor < 0) {
        AdditiveCoarseQuantizer::search(n, x, k, distances, labels);
        return;
    }

    int beam_size = int(k * beam_factor);
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
            search(i1 - i0, x + i0 * d, k, distances + i0 * k, labels + i0 * k);
            InterruptCallback::check();
        }
        return;
    }

    std::vector<int32_t> codes(beam_size * rq.M * n);
    std::vector<float> beam_distances(n * beam_size);

    rq.refine_beam(
            n, 1, x, beam_size, codes.data(), nullptr, beam_distances.data());

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
