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

#include <faiss/impl/AuxIndexStructures.h>
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

IndexScalarQuantizer::IndexScalarQuantizer(
        int d,
        bool is_include_one_attribute,
        ScalarQuantizer::QuantizerType qtype,
        MetricType metric)
        : IndexFlatCodes(0, d, is_include_one_attribute, metric), sq(d, qtype) {
    is_trained = qtype == ScalarQuantizer::QT_fp16 ||
            qtype == ScalarQuantizer::QT_8bit_direct ||
            qtype == ScalarQuantizer::QT_bf16 ||
            qtype == ScalarQuantizer::QT_8bit_direct_signed;
    code_size = sq.code_size;
}

IndexScalarQuantizer::IndexScalarQuantizer(
        int d,
        bool is_include_two_attribute,
        bool mode_two,
        ScalarQuantizer::QuantizerType qtype,
        MetricType metric)
        : IndexFlatCodes(0, d, is_include_two_attribute, mode_two, metric), sq(d, qtype) {
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

void IndexScalarQuantizer::search_with_one_attribute(
        idx_t n,
        const float* x,
        const float lower_attribute,
        const float upper_attribute,
        idx_t k,
        float* distances,
        idx_t* labels,
        float* out_attrs,
        const SearchParameters* params) const {
    const IDSelector* sel = params ? params->sel : nullptr;

    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(is_trained);
    FAISS_THROW_IF_NOT(metric_type == METRIC_L2 || metric_type == METRIC_INNER_PRODUCT);

#pragma omp parallel
    {
        std::unique_ptr<InvertedListScanner> scanner(
                sq.select_InvertedListScanner(metric_type, nullptr, true, sel));

        scanner->list_no = 0; // directly the list number

#pragma omp for
        for (idx_t i = 0; i < n; i++) {
            float* D = distances + k * i;
            idx_t* I = labels + k * i;
            float* ATTR = out_attrs + k * i;
            // re-order heap
            if (metric_type == METRIC_L2) {
                maxheap_heapify_one_attribute(k, D, I, ATTR);
            } else {
                minheap_heapify_one_attribute(k, D, I, ATTR);
            }
            scanner->set_query(x + i * d);
            scanner->scan_codes_with_one_attribute(ntotal, codes.data(), attributes.data(), 
                                                   lower_attribute, upper_attribute, nullptr, D, I, ATTR, k);

            // re-order heap
            if (metric_type == METRIC_L2) {
                maxheap_reorder_one_attribute(k, D, I, ATTR);
            } else {
                minheap_reorder_one_attribute(k, D, I, ATTR);
            }
        }
    }
}

void IndexScalarQuantizer::search_with_two_attribute(
        idx_t n,
        const float* x,
        const float lower_attribute_first,
        const float upper_attribute_first,
        const float lower_attribute_second,
        const float upper_attribute_second,
        idx_t k,
        float* distances,
        idx_t* labels,
        float* out_attrs_first,
        float* out_attrs_second,
        const SearchParameters* params) const {
    const IDSelector* sel = params ? params->sel : nullptr;

    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(is_trained);
    FAISS_THROW_IF_NOT(metric_type == METRIC_L2 || metric_type == METRIC_INNER_PRODUCT);

#pragma omp parallel
    {
        std::unique_ptr<InvertedListScanner> scanner(
                sq.select_InvertedListScanner(metric_type, nullptr, true, sel));

        scanner->list_no = 0; // directly the list number

#pragma omp for
        for (idx_t i = 0; i < n; i++) {
            float* D = distances + k * i;
            idx_t* I = labels + k * i;
            float* ATTRF = out_attrs_first + k * i;
            float* ATTRS = out_attrs_second + k * i;
            // re-order heap
            if (metric_type == METRIC_L2) {
                maxheap_heapify_two_attribute(k, D, I, ATTRF, ATTRS);
            } else {
                minheap_heapify_two_attribute(k, D, I, ATTRF, ATTRS);
            }
            scanner->set_query(x + i * d);
            scanner->scan_codes_with_two_attribute(ntotal, codes.data(), attributes_first.data(), attributes_second.data(),
                                                   lower_attribute_first, upper_attribute_first, 
                                                   lower_attribute_second, upper_attribute_second, nullptr, D, I, ATTRF, ATTRS, k);

            // re-order heap
            if (metric_type == METRIC_L2) {
                maxheap_reorder_two_attribute(k, D, I, ATTRF, ATTRS);
            } else {
                minheap_reorder_two_attribute(k, D, I, ATTRF, ATTRS);
            }
        }
    }
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

void IndexScalarQuantizer::sa_encode(idx_t n, const float* x, uint8_t* bytes) const {
    FAISS_THROW_IF_NOT(is_trained);
    sq.compute_codes(x, bytes, n);
}

void IndexScalarQuantizer::sa_one_attribute_encode(idx_t n, const float* attr, uint8_t* bytes) const {
    if (n > 0) {
        memcpy(bytes, attr, sizeof(float) * n);
    }
}

void IndexScalarQuantizer::sa_two_attribute_encode(idx_t n, const float* attr_first, const float* attr_second, 
                                                            uint8_t* bytes_first, uint8_t* bytes_second) const {
    if (n > 0) {
        memcpy(bytes_first, attr_first, sizeof(float) * n);
        memcpy(bytes_second, attr_second, sizeof(float) * n);
    }
}

void IndexScalarQuantizer::sa_decode(idx_t n, const uint8_t* bytes, float* x) const {
    FAISS_THROW_IF_NOT(is_trained);
    sq.decode(bytes, x, n);
}

void IndexScalarQuantizer::sa_one_attribute_decode(idx_t n, const uint8_t* bytes, float* attr) const {
    if (n > 0) {
        memcpy(attr, bytes, sizeof(float) * n);
    }
}

void IndexScalarQuantizer::sa_two_attribute_decode(idx_t n, const uint8_t* bytes_first, const uint8_t* bytes_second, 
                                                            float* attr_first, float* attr_second) const {
    if (n > 0) {
        memcpy(attr_first, bytes_first, sizeof(float) * n);
        memcpy(attr_second, bytes_second, sizeof(float) * n);
    }
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
        bool by_residual)
        : IndexIVF(quantizer, d, nlist, 0, metric), sq(d, qtype) {
    code_size = sq.code_size;
    this->by_residual = by_residual;
    // was not known at construction time
    invlists->code_size = code_size;
    is_trained = false;
}

IndexIVFScalarQuantizer::IndexIVFScalarQuantizer(
        Index* quantizer,
        size_t d,
        size_t nlist,
        bool is_include_one_attribute,
        ScalarQuantizer::QuantizerType qtype,
        MetricType metric,
        bool by_residual)
        : IndexIVF(quantizer, d, nlist, 0, is_include_one_attribute, metric), sq(d, qtype) {
    code_size = sq.code_size;
    this->by_residual = by_residual;
    // was not known at construction time
    invlists->code_size = code_size;
    is_trained = false;
}

IndexIVFScalarQuantizer::IndexIVFScalarQuantizer(
        Index* quantizer,
        size_t d,
        size_t nlist,
        bool is_include_two_attribute,
        bool mode_two,
        ScalarQuantizer::QuantizerType qtype,
        MetricType metric,
        bool by_residual)
        : IndexIVF(quantizer, d, nlist, 0, is_include_two_attribute, mode_two, metric), sq(d, qtype) {
    code_size = sq.code_size;
    this->by_residual = by_residual;
    // was not known at construction time
    invlists->code_size = code_size;
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

void IndexIVFScalarQuantizer::add_core_with_one_attribute(
        idx_t n,
        const float* x,
        const float* attr,
        const idx_t* xids,
        const idx_t* coarse_idx,
        void* inverted_list_context) {
    FAISS_THROW_IF_NOT(is_trained);
    FAISS_THROW_IF_NOT(invlists->has_one_attribute(0));
    FAISS_THROW_IF_NOT_MSG(invlists->is_include_one_attribute == true, "is_include_one_attribute must be true to add_core_with_one_attribute");
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
                const float* attri = attr + i * 1;
                if (by_residual) {
                    quantizer->compute_residual(xi, residual.data(), list_no);
                    xi = residual.data();
                }

                memset(one_code.data(), 0, code_size);
                squant->encode_vector(xi, one_code.data());

                size_t ofs = invlists->add_entry_with_one_attribute(list_no, id, one_code.data(), (const uint8_t*)attri, inverted_list_context);

                dm_add.add(i, list_no, ofs);

            } else if (rank == 0 && list_no == -1) {
                dm_add.add(i, -1, 0);
            }
        }
    }

    ntotal += n;
}

void IndexIVFScalarQuantizer::add_core_with_two_attribute(
        idx_t n,
        const float* x,
        const float* attr_first,
        const float* attr_second,
        const idx_t* xids,
        const idx_t* coarse_idx,
        void* inverted_list_context) {
    FAISS_THROW_IF_NOT(is_trained);
    FAISS_THROW_IF_NOT(invlists->has_two_attribute(0));
    FAISS_THROW_IF_NOT_MSG(invlists->is_include_two_attribute == true, "is_include_two_attribute must be true to add_core_with_two_attribute");
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
                const float* attrfi = attr_first + i * 1;
                const float* attrsi = attr_second + i * 1;
                if (by_residual) {
                    quantizer->compute_residual(xi, residual.data(), list_no);
                    xi = residual.data();
                }

                memset(one_code.data(), 0, code_size);
                squant->encode_vector(xi, one_code.data());

                size_t ofs = invlists->add_entry_with_two_attribute(list_no, id, one_code.data(), 
                                                                    (const uint8_t*)attrfi, (const uint8_t*)attrsi, inverted_list_context);

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
        const IDSelector* sel) const {
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

void IndexIVFScalarQuantizer::reconstruct_one_attribute_from_offset(
        int64_t list_no,
        int64_t offset,
        float* recons_attr) const {
    memcpy(recons_attr, invlists->get_single_attribute(list_no, offset), attr_size);
}

void IndexIVFScalarQuantizer::reconstruct_two_attribute_from_offset(
        int64_t list_no,
        int64_t offset,
        float* recons_attr_first,
        float* recons_attr_second) const {
    memcpy(recons_attr_first, invlists->get_single_attribute_first(list_no, offset), attr_size);
    memcpy(recons_attr_second, invlists->get_single_attribute_second(list_no, offset), attr_size);
}

} // namespace faiss
