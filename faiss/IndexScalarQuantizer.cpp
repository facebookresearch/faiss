/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/IndexScalarQuantizer.h>

#include <algorithm>
#include <cstdio>

#include <omp.h>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/ScalarQuantizer.h>
#include <faiss/utils/utils.h>

namespace faiss {

/*******************************************************************
 * IndexScalarQuantizer implementation
 ********************************************************************/

IndexScalarQuantizer::IndexScalarQuantizer(
        int d,
        ScalarQuantizer::QuantizerType qtype,
        MetricType metric)
        : IndexFlatCodes(0, d, metric), sq(d, qtype) {
    is_trained = qtype == ScalarQuantizer::QT_fp16 ||
            qtype == ScalarQuantizer::QT_8bit_direct ||
            qtype == ScalarQuantizer::QT_bf16 ||
            qtype == ScalarQuantizer::QT_8bit_direct_signed;
    code_size = sq.code_size;
}

IndexScalarQuantizer::IndexScalarQuantizer()
        : IndexScalarQuantizer(0, ScalarQuantizer::QT_8bit) {}

void IndexScalarQuantizer::train(idx_t n, const float* x) {
    sq.train(n, x);
    is_trained = true;
}

void IndexScalarQuantizer::train(
        idx_t n,
        const void* x,
        NumericType numeric_type) {
    Index::train(n, x, numeric_type);
}

void IndexScalarQuantizer::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    const IDSelector* sel = params ? params->sel : nullptr;

    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(is_trained);
    FAISS_THROW_IF_NOT(
            metric_type == METRIC_L2 || metric_type == METRIC_INNER_PRODUCT);

#pragma omp parallel
    {
        std::unique_ptr<InvertedListScanner> scanner(
                sq.select_InvertedListScanner(metric_type, nullptr, true, sel));

        scanner->list_no = 0; // directly the list number

#pragma omp for
        for (idx_t i = 0; i < n; i++) {
            float* D = distances + k * i;
            idx_t* I = labels + k * i;
            // re-order heap
            if (metric_type == METRIC_L2) {
                maxheap_heapify(k, D, I);
            } else {
                minheap_heapify(k, D, I);
            }
            scanner->set_query(x + i * d);
            scanner->scan_codes(ntotal, codes.data(), nullptr, D, I, k);

            // re-order heap
            if (metric_type == METRIC_L2) {
                maxheap_reorder(k, D, I);
            } else {
                minheap_reorder(k, D, I);
            }
        }
    }
}

void IndexScalarQuantizer::search(
        idx_t n,
        const void* x,
        NumericType numeric_type,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    Index::search(n, x, numeric_type, k, distances, labels, params);
}

FlatCodesDistanceComputer* IndexScalarQuantizer::get_FlatCodesDistanceComputer()
        const {
    ScalarQuantizer::SQDistanceComputer* dc =
            sq.get_distance_computer(metric_type);
    dc->code_size = sq.code_size;
    dc->codes = codes.data();
    return dc;
}

/* Codec interface */

void IndexScalarQuantizer::sa_encode(idx_t n, const float* x, uint8_t* bytes)
        const {
    FAISS_THROW_IF_NOT(is_trained);
    sq.compute_codes(x, bytes, n);
}

void IndexScalarQuantizer::sa_decode(idx_t n, const uint8_t* bytes, float* x)
        const {
    FAISS_THROW_IF_NOT(is_trained);
    sq.decode(bytes, x, n);
}

/*******************************************************************
 * IndexIVFScalarQuantizer implementation
 ********************************************************************/

IndexIVFScalarQuantizer::IndexIVFScalarQuantizer(
        Index* quantizer,
        size_t d,
        size_t nlist,
        ScalarQuantizer::QuantizerType qtype,
        MetricType metric,
        bool by_residual,
        bool own_invlists)
        : IndexIVF(quantizer, d, nlist, 0, metric, own_invlists), sq(d, qtype) {
    code_size = sq.code_size;
    this->by_residual = by_residual;
    if (invlists) {
        // was not known at construction time
        invlists->code_size = code_size;
    }
    is_trained = false;
}

IndexIVFScalarQuantizer::IndexIVFScalarQuantizer() : IndexIVF() {
    by_residual = true;
}

void IndexIVFScalarQuantizer::train_encoder(
        idx_t n,
        const float* x,
        const idx_t* assign) {
    sq.train(n, x);
}

idx_t IndexIVFScalarQuantizer::train_encoder_num_vectors() const {
    return 100000;
}

void IndexIVFScalarQuantizer::encode_vectors(
        idx_t n,
        const float* x,
        const idx_t* list_nos,
        uint8_t* codes,
        bool include_listnos) const {
    std::unique_ptr<ScalarQuantizer::SQuantizer> squant(sq.select_quantizer());
    size_t coarse_size = include_listnos ? coarse_code_size() : 0;
    memset(codes, 0, (code_size + coarse_size) * n);

#pragma omp parallel if (n > 1000)
    {
        std::vector<float> residual(d);

#pragma omp for
        for (idx_t i = 0; i < n; i++) {
            int64_t list_no = list_nos[i];
            if (list_no >= 0) {
                const float* xi = x + i * d;
                uint8_t* code = codes + i * (code_size + coarse_size);
                if (by_residual) {
                    quantizer->compute_residual(xi, residual.data(), list_no);
                    xi = residual.data();
                }
                if (coarse_size) {
                    encode_listno(list_no, code);
                }
                squant->encode_vector(xi, code + coarse_size);
            }
        }
    }
}

void IndexIVFScalarQuantizer::decode_vectors(
        idx_t n,
        const uint8_t* codes,
        const idx_t*,
        float* x) const {
    FAISS_THROW_IF_NOT(is_trained);
    return sq.decode(codes, x, n);
}

void IndexIVFScalarQuantizer::sa_decode(idx_t n, const uint8_t* codes, float* x)
        const {
    std::unique_ptr<ScalarQuantizer::SQuantizer> squant(sq.select_quantizer());
    size_t coarse_size = coarse_code_size();

#pragma omp parallel if (n > 1000)
    {
        std::vector<float> residual(d);

#pragma omp for
        for (idx_t i = 0; i < n; i++) {
            const uint8_t* code = codes + i * (code_size + coarse_size);
            int64_t list_no = decode_listno(code);
            float* xi = x + i * d;
            squant->decode_vector(code + coarse_size, xi);
            if (by_residual) {
                quantizer->reconstruct(list_no, residual.data());
                for (size_t j = 0; j < d; j++) {
                    xi[j] += residual[j];
                }
            }
        }
    }
}

void IndexIVFScalarQuantizer::add_core(
        idx_t n,
        const float* x,
        const idx_t* xids,
        const idx_t* coarse_idx,
        void* inverted_list_context) {
    FAISS_THROW_IF_NOT(is_trained);

    std::unique_ptr<ScalarQuantizer::SQuantizer> squant(sq.select_quantizer());

    DirectMapAdd dm_add(direct_map, n, xids);

#pragma omp parallel
    {
        std::vector<float> residual(d);
        std::vector<uint8_t> one_code(code_size);
        int nt = omp_get_num_threads();
        int rank = omp_get_thread_num();

        // each thread takes care of a subset of lists
        for (size_t i = 0; i < n; i++) {
            int64_t list_no = coarse_idx[i];
            if (list_no >= 0 && list_no % nt == rank) {
                int64_t id = xids ? xids[i] : ntotal + i;

                const float* xi = x + i * d;
                if (by_residual) {
                    quantizer->compute_residual(xi, residual.data(), list_no);
                    xi = residual.data();
                }

                memset(one_code.data(), 0, code_size);
                squant->encode_vector(xi, one_code.data());

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

InvertedListScanner* IndexIVFScalarQuantizer::get_InvertedListScanner(
        bool store_pairs,
        const IDSelector* sel,
        const IVFSearchParameters*) const {
    return sq.select_InvertedListScanner(
            metric_type, quantizer, store_pairs, sel, by_residual);
}

void IndexIVFScalarQuantizer::reconstruct_from_offset(
        int64_t list_no,
        int64_t offset,
        float* recons) const {
    const uint8_t* code = invlists->get_single_code(list_no, offset);

    if (by_residual) {
        std::vector<float> centroid(d);
        quantizer->reconstruct(list_no, centroid.data());

        sq.decode(code, recons, 1);
        for (int i = 0; i < d; ++i) {
            recons[i] += centroid[i];
        }
    } else {
        sq.decode(code, recons, 1);
    }
}

} // namespace faiss
