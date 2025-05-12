/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexIVFRaBitQ.h>

#include <omp.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/RaBitQuantizer.h>

namespace faiss {

IndexIVFRaBitQ::IndexIVFRaBitQ(
        Index* quantizer,
        const size_t d,
        const size_t nlist,
        MetricType metric)
        : IndexIVF(quantizer, d, nlist, 0, metric), rabitq(d, metric) {
    code_size = rabitq.code_size;
    invlists->code_size = code_size;
    is_trained = false;

    by_residual = true;
}

IndexIVFRaBitQ::IndexIVFRaBitQ() {
    by_residual = true;
}

void IndexIVFRaBitQ::train_encoder(
        idx_t n,
        const float* x,
        const idx_t* assign) {
    rabitq.train(n, x);
}

void IndexIVFRaBitQ::encode_vectors(
        idx_t n,
        const float* x,
        const idx_t* list_nos,
        uint8_t* codes,
        bool include_listnos) const {
    size_t coarse_size = include_listnos ? coarse_code_size() : 0;
    memset(codes, 0, (code_size + coarse_size) * n);

#pragma omp parallel if (n > 1000)
    {
        std::vector<float> centroid(d);

#pragma omp for
        for (idx_t i = 0; i < n; i++) {
            int64_t list_no = list_nos[i];
            if (list_no >= 0) {
                const float* xi = x + i * d;
                uint8_t* code = codes + i * (code_size + coarse_size);

                // both by_residual and !by_residual lead to the same code
                quantizer->reconstruct(list_no, centroid.data());
                rabitq.compute_codes_core(
                        xi, code + coarse_size, 1, centroid.data());

                if (coarse_size) {
                    encode_listno(list_no, code);
                }
            }
        }
    }
}

void IndexIVFRaBitQ::add_core(
        idx_t n,
        const float* x,
        const idx_t* xids,
        const idx_t* precomputed_idx,
        void* inverted_list_context) {
    FAISS_THROW_IF_NOT(is_trained);

    DirectMapAdd dm_add(direct_map, n, xids);

#pragma omp parallel
    {
        std::vector<uint8_t> one_code(code_size);
        std::vector<float> centroid(d);

        int nt = omp_get_num_threads();
        int rank = omp_get_thread_num();

        // each thread takes care of a subset of lists
        for (size_t i = 0; i < n; i++) {
            int64_t list_no = precomputed_idx[i];
            if (list_no >= 0 && list_no % nt == rank) {
                int64_t id = xids ? xids[i] : ntotal + i;

                const float* xi = x + i * d;

                // both by_residual and !by_residual lead to the same code
                quantizer->reconstruct(list_no, centroid.data());
                rabitq.compute_codes_core(
                        xi, one_code.data(), 1, centroid.data());

                size_t ofs = invlists->add_entry(
                        list_no, id, one_code.data(), inverted_list_context);

                dm_add.add(i, list_no, ofs);

            } else if (rank == 0 && list_no == -1) {
                dm_add.add(i, -1, 0);
            }
        }
    }

    ntotal += n;
}

struct RaBitInvertedListScanner : InvertedListScanner {
    const IndexIVFRaBitQ& ivf_rabitq;

    std::vector<float> reconstructed_centroid;
    std::vector<float> query_vector;

    std::unique_ptr<FlatCodesDistanceComputer> dc;

    uint8_t qb = 0;

    RaBitInvertedListScanner(
            const IndexIVFRaBitQ& ivf_rabitq_in,
            bool store_pairs = false,
            const IDSelector* sel = nullptr,
            uint8_t qb_in = 0)
            : InvertedListScanner(store_pairs, sel),
              ivf_rabitq{ivf_rabitq_in},
              qb{qb_in} {
        keep_max = is_similarity_metric(ivf_rabitq.metric_type);
        code_size = ivf_rabitq.code_size;
    }

    /// from now on we handle this query.
    void set_query(const float* query_vector_in) override {
        query_vector.assign(query_vector_in, query_vector_in + ivf_rabitq.d);

        internal_try_setup_dc();
    }

    /// following codes come from this inverted list
    void set_list(idx_t list_no, float coarse_dis) override {
        this->list_no = list_no;

        reconstructed_centroid.resize(ivf_rabitq.d);
        ivf_rabitq.quantizer->reconstruct(
                list_no, reconstructed_centroid.data());

        internal_try_setup_dc();
    }

    /// compute a single query-to-code distance
    float distance_to_code(const uint8_t* code) const override {
        return dc->distance_to_code(code);
    }

    void internal_try_setup_dc() {
        if (!query_vector.empty() && !reconstructed_centroid.empty()) {
            // both query_vector and centroid are available!
            // set up DistanceComputer
            dc.reset(ivf_rabitq.rabitq.get_distance_computer(
                    qb, reconstructed_centroid.data()));

            dc->set_query(query_vector.data());
        }
    }
};

InvertedListScanner* IndexIVFRaBitQ::get_InvertedListScanner(
        bool store_pairs,
        const IDSelector* sel,
        const IVFSearchParameters* search_params_in) const {
    uint8_t used_qb = qb;
    if (auto params = dynamic_cast<const IVFRaBitQSearchParameters*>(
                search_params_in)) {
        used_qb = params->qb;
    }

    return new RaBitInvertedListScanner(*this, store_pairs, sel, used_qb);
}

void IndexIVFRaBitQ::reconstruct_from_offset(
        int64_t list_no,
        int64_t offset,
        float* recons) const {
    const uint8_t* code = invlists->get_single_code(list_no, offset);

    std::vector<float> centroid(d);
    quantizer->reconstruct(list_no, centroid.data());

    rabitq.decode_core(code, recons, 1, centroid.data());
}

void IndexIVFRaBitQ::sa_decode(idx_t n, const uint8_t* codes, float* x) const {
    size_t coarse_size = coarse_code_size();

#pragma omp parallel
    {
        std::vector<float> centroid(d);

#pragma omp for
        for (idx_t i = 0; i < n; i++) {
            const uint8_t* code = codes + i * (code_size + coarse_size);
            int64_t list_no = decode_listno(code);
            float* xi = x + i * d;

            quantizer->reconstruct(list_no, centroid.data());
            rabitq.decode_core(code + coarse_size, xi, 1, centroid.data());
        }
    }
}

struct IVFRaBitDistanceComputer : DistanceComputer {
    const float* q = nullptr;
    const IndexIVFRaBitQ* parent = nullptr;

    void set_query(const float* x) override;

    float operator()(idx_t i) override;

    float symmetric_dis(idx_t i, idx_t j) override;
};

void IVFRaBitDistanceComputer::set_query(const float* x) {
    q = x;
}

float IVFRaBitDistanceComputer::operator()(idx_t i) {
    // find the appropriate list
    idx_t lo = parent->direct_map.get(i);
    uint64_t list_no = lo_listno(lo);
    uint64_t offset = lo_offset(lo);

    const uint8_t* code = parent->invlists->get_single_code(list_no, offset);

    // ok, we know the appropriate cluster that we need
    std::vector<float> centroid(parent->d);
    parent->quantizer->reconstruct(list_no, centroid.data());

    // compute the distance
    float distance = 0;

    std::unique_ptr<FlatCodesDistanceComputer> dc(
            parent->rabitq.get_distance_computer(parent->qb, centroid.data()));
    dc->set_query(q);
    distance = dc->distance_to_code(code);

    // deallocate
    parent->invlists->release_codes(list_no, code);

    // done
    return distance;
}

float IVFRaBitDistanceComputer::symmetric_dis(idx_t i, idx_t j) {
    FAISS_THROW_MSG("Not implemented");
}

DistanceComputer* IndexIVFRaBitQ::get_distance_computer() const {
    IVFRaBitDistanceComputer* dc = new IVFRaBitDistanceComputer;
    dc->parent = this;
    return dc;
}

} // namespace faiss
