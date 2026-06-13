/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexIVFEDEN.h>

#include <omp.h>

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <vector>

#include <faiss/impl/EDENQuantizer.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/expanded_scanners.h>

namespace faiss {

IndexIVFEDEN::IndexIVFEDEN(
        Index* quantizer_in,
        const size_t d_in,
        const size_t nlist_in,
        MetricType metric,
        bool own_invlists_in,
        uint8_t nb_bits_in,
        EDENScaleType scale_type)
        : IndexIVF(quantizer_in, d_in, nlist_in, 0, metric, own_invlists_in),
          eden(d_in, metric, nb_bits_in, scale_type) {
    code_size = eden.code_size;
    if (own_invlists_in) {
        invlists->code_size = code_size;
    }
    is_trained = false;
    by_residual = true;
}

IndexIVFEDEN::IndexIVFEDEN() {
    by_residual = true;
}

void IndexIVFEDEN::train_encoder(
        idx_t n,
        const float* x,
        const idx_t* /*assign*/) {
    eden.train(n, x);
}

void IndexIVFEDEN::encode_vectors(
        idx_t n,
        const float* x,
        const idx_t* list_nos,
        uint8_t* codes,
        bool include_listnos) const {
    const size_t coarse_size = include_listnos ? coarse_code_size() : 0;
    memset(codes, 0, (code_size + coarse_size) * n);

#pragma omp parallel if (n > 1000)
    {
        std::vector<float> centroid(d);

#pragma omp for
        for (idx_t i = 0; i < n; i++) {
            const int64_t list_no = list_nos[i];
            if (list_no >= 0) {
                const float* xi = x + i * d;
                uint8_t* code = codes + i * (code_size + coarse_size);

                quantizer->reconstruct(list_no, centroid.data());
                eden.compute_codes_core(
                        xi, code + coarse_size, 1, centroid.data());

                if (coarse_size) {
                    encode_listno(list_no, code);
                }
            }
        }
    }
}

void IndexIVFEDEN::decode_vectors(
        idx_t n,
        const uint8_t* codes,
        const idx_t* listnos,
        float* x) const {
#pragma omp parallel
    {
        std::vector<float> centroid(d);

#pragma omp for
        for (idx_t i = 0; i < n; i++) {
            const uint8_t* code = codes + i * code_size;
            const int64_t list_no = listnos[i];
            float* xi = x + i * d;

            quantizer->reconstruct(list_no, centroid.data());
            eden.decode_core(code, xi, 1, centroid.data());
        }
    }
}

void IndexIVFEDEN::add_core(
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

        const int nt = omp_get_num_threads();
        const int rank = omp_get_thread_num();

        for (idx_t i = 0; i < n; i++) {
            const int64_t list_no = precomputed_idx[i];
            if (list_no >= 0 && list_no % nt == rank) {
                const int64_t id = xids ? xids[i] : ntotal + i;
                const float* xi = x + i * d;

                quantizer->reconstruct(list_no, centroid.data());
                eden.compute_codes_core(
                        xi, one_code.data(), 1, centroid.data());

                const size_t ofs = invlists->add_entry(
                        list_no, id, one_code.data(), inverted_list_context);
                dm_add.add(i, list_no, ofs);
            } else if (rank == 0 && list_no == -1) {
                dm_add.add(i, -1, 0);
            }
        }
    }

    ntotal += n;
}

namespace {

struct EDENInvertedListScanner : InvertedListScanner {
    using InvertedListScanner::scan_codes;

    const IndexIVFEDEN& ivf_eden;
    std::vector<float> reconstructed_centroid;
    std::vector<float> query_vector;
    std::unique_ptr<FlatCodesDistanceComputer> dc;

    explicit EDENInvertedListScanner(
            const IndexIVFEDEN& ivf_eden_in,
            bool store_pairs_in = false,
            const IDSelector* sel_in = nullptr)
            : InvertedListScanner(store_pairs_in, sel_in),
              ivf_eden(ivf_eden_in) {
        keep_max = is_similarity_metric(ivf_eden.metric_type);
        code_size = ivf_eden.code_size;
    }

    void set_query(const float* query_vector_in) override {
        query_vector.assign(query_vector_in, query_vector_in + ivf_eden.d);
        internal_try_setup_dc();
    }

    void set_list(idx_t list_no_in, float /*coarse_dis*/) override {
        list_no = list_no_in;

        reconstructed_centroid.resize(ivf_eden.d);
        ivf_eden.quantizer->reconstruct(
                list_no_in, reconstructed_centroid.data());
        internal_try_setup_dc();
    }

    float distance_to_code(const uint8_t* code) const final {
        return dc->distance_to_code(code);
    }

    size_t scan_codes(
            size_t list_size,
            const uint8_t* codes,
            const idx_t* ids,
            ResultHandler& handler) const override {
        return run_scan_codes(*this, list_size, codes, ids, handler);
    }

    void internal_try_setup_dc() {
        if (!query_vector.empty() && !reconstructed_centroid.empty()) {
            dc.reset(ivf_eden.eden.get_distance_computer(
                    reconstructed_centroid.data()));
            dc->set_query(query_vector.data());
        }
    }
};

} // namespace

InvertedListScanner* IndexIVFEDEN::get_InvertedListScanner(
        bool store_pairs,
        const IDSelector* sel,
        const IVFSearchParameters* /*params*/) const {
    return new EDENInvertedListScanner(*this, store_pairs, sel);
}

void IndexIVFEDEN::reconstruct_from_offset(
        int64_t list_no,
        int64_t offset,
        float* recons) const {
    const uint8_t* code = invlists->get_single_code(list_no, offset);

    std::vector<float> centroid(d);
    quantizer->reconstruct(list_no, centroid.data());
    eden.decode_core(code, recons, 1, centroid.data());

    invlists->release_codes(list_no, code);
}

void IndexIVFEDEN::sa_decode(idx_t n, const uint8_t* bytes, float* x) const {
    const size_t coarse_size = coarse_code_size();

#pragma omp parallel
    {
        std::vector<float> centroid(d);

#pragma omp for
        for (idx_t i = 0; i < n; i++) {
            const uint8_t* code = bytes + i * (code_size + coarse_size);
            const int64_t list_no = decode_listno(code);
            float* xi = x + i * d;

            quantizer->reconstruct(list_no, centroid.data());
            eden.decode_core(code + coarse_size, xi, 1, centroid.data());
        }
    }
}

struct IVFEDENDistanceComputer : DistanceComputer {
    const float* q = nullptr;
    const IndexIVFEDEN* parent = nullptr;

    void set_query(const float* x) override {
        q = x;
    }

    float operator()(idx_t i) override {
        const idx_t lo = parent->direct_map.get(i);
        const uint64_t list_no = lo_listno(lo);
        const uint64_t offset = lo_offset(lo);
        const uint8_t* code =
                parent->invlists->get_single_code(list_no, offset);

        std::vector<float> centroid(parent->d);
        parent->quantizer->reconstruct(list_no, centroid.data());

        std::unique_ptr<FlatCodesDistanceComputer> dc(
                parent->eden.get_distance_computer(centroid.data()));
        dc->set_query(q);
        const float distance = dc->distance_to_code(code);

        parent->invlists->release_codes(list_no, code);
        return distance;
    }

    float symmetric_dis(idx_t /*i*/, idx_t /*j*/) override {
        FAISS_THROW_MSG("Not implemented");
    }
};

DistanceComputer* IndexIVFEDEN::get_distance_computer() const {
    IVFEDENDistanceComputer* dc = new IVFEDENDistanceComputer;
    dc->parent = this;
    return dc;
}

} // namespace faiss
