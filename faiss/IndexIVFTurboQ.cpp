/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexIVFTurboQ.h>

#include <omp.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/impl/TurboQuantizer.h>
#include <faiss/impl/expanded_scanners.h>

namespace faiss {

IndexIVFTurboQ::IndexIVFTurboQ(
        Index* quantizer_in,
        size_t d_in,
        size_t nlist_in,
        MetricType metric,
        bool own_invlists_in,
        uint8_t nb_bits_in,
        QJLProjectionType qjl_type_in,
        uint8_t nb_bits_lo_in,
        size_t n_hi_dims_in)
        : IndexIVF(quantizer_in, d_in, nlist_in, 0, metric, own_invlists_in),
          turboq(d_in,
                 metric,
                 nb_bits_in,
                 qjl_type_in,
                 nb_bits_lo_in,
                 n_hi_dims_in) {
    code_size = turboq.code_size;
    if (own_invlists_in) {
        invlists->code_size = code_size;
    }
    is_trained = false;

    // TurboQ operates on raw rotated vectors, not IVF residuals,
    // because its Lloyd-Max codebook is calibrated for unit-sphere
    // coordinates, not residual distributions.
    by_residual = false;
}

IndexIVFTurboQ::IndexIVFTurboQ() {
    by_residual = false;
}

void IndexIVFTurboQ::train_encoder(
        idx_t n,
        const float* x,
        const idx_t* /*assign*/) {
    turboq.train(n, x);
}

void IndexIVFTurboQ::encode_vectors(
        idx_t n,
        const float* x,
        const idx_t* list_nos,
        uint8_t* codes,
        bool include_listnos) const {
    size_t coarse_size = include_listnos ? coarse_code_size() : 0;
    memset(codes, 0, (code_size + coarse_size) * n);

#pragma omp parallel if (n > 1000)
    {
#pragma omp for
        for (idx_t i = 0; i < n; i++) {
            int64_t list_no = list_nos[i];
            if (list_no >= 0) {
                const float* xi = x + i * d;
                uint8_t* code = codes + i * (code_size + coarse_size);

                // by_residual=false: centroid is unused
                turboq.compute_codes_core(xi, code + coarse_size, 1, nullptr);

                if (coarse_size) {
                    encode_listno(list_no, code);
                }
            }
        }
    }
}

void IndexIVFTurboQ::decode_vectors(
        idx_t n,
        const uint8_t* codes,
        const idx_t* /*listnos*/,
        float* x) const {
#pragma omp parallel
    {
#pragma omp for
        for (idx_t i = 0; i < n; i++) {
            const uint8_t* code = codes + i * code_size;
            float* xi = x + i * d;

            turboq.decode_core(code, xi, 1, nullptr);
        }
    }
}

void IndexIVFTurboQ::add_core(
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

        int nt = omp_get_num_threads();
        int rank = omp_get_thread_num();

        // each thread takes care of a subset of lists
        for (idx_t i = 0; i < n; i++) {
            int64_t list_no = precomputed_idx[i];
            if (list_no >= 0 && list_no % nt == rank) {
                int64_t id = xids ? xids[i] : ntotal + i;

                const float* xi = x + i * d;

                turboq.compute_codes_core(xi, one_code.data(), 1, nullptr);

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

namespace {

struct TurboQInvertedListScanner : InvertedListScanner {
    using InvertedListScanner::scan_codes;
    const IndexIVFTurboQ& ivf;

    std::unique_ptr<FlatCodesDistanceComputer> dc;
    TurboQDistanceComputer* turboq_dc = nullptr;

    uint8_t qb = 0;
    bool int_qjl = false;

    explicit TurboQInvertedListScanner(
            const IndexIVFTurboQ& ivf_in,
            bool store_pairs_in = false,
            const IDSelector* sel_in = nullptr,
            uint8_t qb_in = 0,
            bool int_qjl_in = false)
            : InvertedListScanner(store_pairs_in, sel_in),
              ivf{ivf_in},
              qb{qb_in},
              int_qjl{int_qjl_in} {
        keep_max = is_similarity_metric(ivf.metric_type);
        code_size = ivf.code_size;
    }

    void set_query(const float* query_vector_in) override {
        dc.reset(ivf.turboq.get_distance_computer());
        turboq_dc = dynamic_cast<TurboQDistanceComputer*>(dc.get());
        FAISS_THROW_IF_NOT_MSG(
                turboq_dc != nullptr, "TurboQ distance computer cast failed");

        turboq_dc->qb = qb;
        turboq_dc->int_qjl = int_qjl;

        dc->set_query(query_vector_in);
    }

    void set_list(idx_t list_no_in, float /*coarse_dis*/) override {
        this->list_no = list_no_in;
    }

    float distance_to_code(const uint8_t* code) const final {
        return dc->distance_to_code(code);
    }

    size_t scan_codes(
            size_t list_size,
            const uint8_t* codes,
            const idx_t* ids,
            ResultHandler& handler) const override {
        if (turboq_dc != nullptr) {
            turboq_dc->threshold_ptr = &handler.threshold;
            turboq_dc->prescreen_l2 = !keep_max;
        }

        size_t nup = run_scan_codes(*this, list_size, codes, ids, handler);

        if (turboq_dc != nullptr) {
            turboq_dc->threshold_ptr = nullptr;
        }

        return nup;
    }
};

} // anonymous namespace

InvertedListScanner* IndexIVFTurboQ::get_InvertedListScanner(
        bool store_pairs,
        const IDSelector* sel,
        const IVFSearchParameters* search_params_in) const {
    uint8_t used_qb = 0;
    bool used_int_qjl = false;

    if (auto params = dynamic_cast<const IVFTurboQSearchParameters*>(
                search_params_in)) {
        used_qb = params->qb;
        used_int_qjl = params->int_qjl;
    }

    return new TurboQInvertedListScanner(
            *this, store_pairs, sel, used_qb, used_int_qjl);
}

namespace {

struct IVFTurboQDistanceComputer : DistanceComputer {
    const IndexIVFTurboQ* parent = nullptr;
    const float* q = nullptr;
    std::unique_ptr<FlatCodesDistanceComputer> dc;

    void set_query(const float* x) override {
        q = x;
        // Lazily initialize DC -- we need it for operator()
        if (!dc) {
            dc.reset(parent->turboq.get_distance_computer());
        }
        dc->set_query(x);
    }

    float operator()(idx_t i) override {
        // Look up the code from the inverted lists via direct_map
        idx_t lo = parent->direct_map.get(i);
        uint64_t list_no = lo_listno(lo);
        uint64_t offset = lo_offset(lo);

        const uint8_t* code =
                parent->invlists->get_single_code(list_no, offset);

        float distance = dc->distance_to_code(code);

        parent->invlists->release_codes(list_no, code);

        return distance;
    }

    float symmetric_dis(idx_t /*i*/, idx_t /*j*/) override {
        FAISS_THROW_MSG("Not implemented");
    }
};

} // anonymous namespace

void IndexIVFTurboQ::reconstruct_from_offset(
        int64_t list_no,
        int64_t offset,
        float* recons) const {
    const uint8_t* code = invlists->get_single_code(list_no, offset);

    turboq.decode_core(code, recons, 1, nullptr);
}

void IndexIVFTurboQ::sa_decode(idx_t n, const uint8_t* codes, float* x) const {
    size_t coarse_size = coarse_code_size();

#pragma omp parallel
    {
#pragma omp for
        for (idx_t i = 0; i < n; i++) {
            const uint8_t* code = codes + i * (code_size + coarse_size);
            float* xi = x + i * d;

            turboq.decode_core(code + coarse_size, xi, 1, nullptr);
        }
    }
}

DistanceComputer* IndexIVFTurboQ::get_distance_computer() const {
    IVFTurboQDistanceComputer* dc = new IVFTurboQDistanceComputer;
    dc->parent = this;
    return dc;
}

} // namespace faiss
