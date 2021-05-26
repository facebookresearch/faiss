/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexResidual.h>

#include <algorithm>
#include <cmath>
#include <cstring>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/extra_distances.h>
#include <faiss/utils/utils.h>

namespace faiss {

/**************************************************************************************
 * IndexResidual
 **************************************************************************************/

IndexResidual::IndexResidual(
        int d,        ///< dimensionality of the input vectors
        size_t M,     ///< number of subquantizers
        size_t nbits, ///< number of bit per subvector index
        MetricType metric,
        Search_type_t search_type_in)
        : Index(d, metric), rq(d, M, nbits), search_type(ST_decompress) {
    is_trained = false;
    norm_max = norm_min = NAN;
    set_search_type(search_type_in);
}

IndexResidual::IndexResidual(
        int d,
        const std::vector<size_t>& nbits,
        MetricType metric,
        Search_type_t search_type_in)
        : Index(d, metric), rq(d, nbits), search_type(ST_decompress) {
    is_trained = false;
    norm_max = norm_min = NAN;
    set_search_type(search_type_in);
}

IndexResidual::IndexResidual() : IndexResidual(0, 0, 0) {}

void IndexResidual::set_search_type(Search_type_t new_search_type) {
    int norm_bits = new_search_type == ST_norm_float ? 32
            : new_search_type == ST_norm_qint8       ? 8
                                                     : 0;

    FAISS_THROW_IF_NOT(ntotal == 0);

    search_type = new_search_type;
    code_size = (rq.tot_bits + norm_bits + 7) / 8;
}

void IndexResidual::train(idx_t n, const float* x) {
    rq.train(n, x);

    std::vector<float> norms(n);
    fvec_norms_L2sqr(norms.data(), x, d, n);

    norm_min = HUGE_VALF;
    norm_max = -HUGE_VALF;
    for (idx_t i = 0; i < n; i++) {
        if (norms[i] < norm_min) {
            norm_min = norms[i];
        }
        if (norms[i] > norm_min) {
            norm_max = norms[i];
        }
    }

    is_trained = true;
}

void IndexResidual::add(idx_t n, const float* x) {
    FAISS_THROW_IF_NOT(is_trained);
    codes.resize((n + ntotal) * rq.code_size);
    if (search_type == ST_decompress || search_type == ST_LUT_nonorm) {
        rq.compute_codes(x, &codes[ntotal * rq.code_size], n);
    } else {
        // should compute codes + compute and quantize norms
        FAISS_THROW_MSG("not implemented");
    }
    ntotal += n;
}

namespace {

template <class VectorDistance, class ResultHandler>
void search_with_decompress(
        const IndexResidual& ir,
        const float* xq,
        VectorDistance& vd,
        ResultHandler& res) {
    const uint8_t* codes = ir.codes.data();
    size_t ntotal = ir.ntotal;
    size_t code_size = ir.code_size;

    using SingleResultHandler = typename ResultHandler::SingleResultHandler;

#pragma omp parallel for
    for (int64_t q = 0; q < res.nq; q++) {
        SingleResultHandler resi(res);
        resi.begin(q);
        std::vector<float> tmp(ir.d);
        const float* x = xq + ir.d * q;
        for (size_t i = 0; i < ntotal; i++) {
            ir.rq.decode(codes + i * code_size, tmp.data(), 1);
            float dis = vd(x, tmp.data());
            resi.add_result(dis, i);
        }
        resi.end();
    }
}

} // anonymous namespace

void IndexResidual::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels) const {
    if (search_type == ST_decompress) {
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
        FAISS_THROW_MSG("not implemented");
    }
}

void IndexResidual::reset() {
    codes.clear();
    ntotal = 0;
}

size_t IndexResidual::sa_code_size() const {
    return code_size;
}

void IndexResidual::sa_encode(idx_t n, const float* x, uint8_t* bytes) const {
    return rq.compute_codes(x, bytes, n);
}

void IndexResidual::sa_decode(idx_t n, const uint8_t* bytes, float* x) const {
    return rq.decode(bytes, x, n);
}

/**************************************************************************************
 * ResidualCoarseQuantizer
 **************************************************************************************/

ResidualCoarseQuantizer::ResidualCoarseQuantizer(
        int d,        ///< dimensionality of the input vectors
        size_t M,     ///< number of subquantizers
        size_t nbits, ///< number of bit per subvector index
        MetricType metric)
        : Index(d, metric), rq(d, M, nbits), beam_factor(4.0) {
    FAISS_THROW_IF_NOT(rq.tot_bits <= 63);
    is_trained = false;
}

ResidualCoarseQuantizer::ResidualCoarseQuantizer(
        int d,
        const std::vector<size_t>& nbits,
        MetricType metric)
        : Index(d, metric), rq(d, nbits), beam_factor(4.0) {
    FAISS_THROW_IF_NOT(rq.tot_bits <= 63);
    is_trained = false;
}

ResidualCoarseQuantizer::ResidualCoarseQuantizer() {}

void ResidualCoarseQuantizer::train(idx_t n, const float* x) {
    rq.train(n, x);
    is_trained = true;
    ntotal = (idx_t)1 << rq.tot_bits;
}

void ResidualCoarseQuantizer::add(idx_t, const float*) {
    FAISS_THROW_MSG("not applicable");
}

void ResidualCoarseQuantizer::set_beam_factor(float new_beam_factor) {
    centroid_norms.resize(0);
    beam_factor = new_beam_factor;
    if (new_beam_factor > 0) {
        FAISS_THROW_IF_NOT(new_beam_factor >= 1.0);
        return;
    }

    if (metric_type == METRIC_L2) {
        centroid_norms.resize((size_t)1 << rq.tot_bits);
        rq.compute_centroid_norms(centroid_norms.data());
    }
}

void ResidualCoarseQuantizer::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels) const {
    if (beam_factor < 0) {
        if (metric_type == METRIC_INNER_PRODUCT) {
            rq.knn_exact_inner_product(n, x, k, distances, labels);
        } else if (metric_type == METRIC_L2) {
            FAISS_THROW_IF_NOT(centroid_norms.size() == ntotal);
            rq.knn_exact_L2(n, x, k, distances, labels, centroid_norms.data());
        }
        return;
    }

    int beam_size = int(k * beam_factor);

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

void ResidualCoarseQuantizer::reconstruct(idx_t key, float* recons) const {
    rq.decode_64bit(key, recons);
}

void ResidualCoarseQuantizer::reset() {
    FAISS_THROW_MSG("not applicable");
}

} // namespace faiss
