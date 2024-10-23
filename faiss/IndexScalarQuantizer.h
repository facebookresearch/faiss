/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#ifndef FAISS_INDEX_SCALAR_QUANTIZER_H
#define FAISS_INDEX_SCALAR_QUANTIZER_H

#include <stdint.h>
#include <vector>

#include <faiss/IndexFlatCodes.h>
#include <faiss/IndexIVF.h>
#include <faiss/impl/ScalarQuantizer.h>

namespace faiss {

/**
 * Flat index built on a scalar quantizer.
 */
struct IndexScalarQuantizer : IndexFlatCodes {
    /// Used to encode the vectors
    ScalarQuantizer sq;

    /** Constructor.
     *
     * @param d      dimensionality of the input vectors
     * @param M      number of subquantizers
     * @param nbits  number of bit per subvector index
     */
    IndexScalarQuantizer(
            int d,
            ScalarQuantizer::QuantizerType qtype,
            MetricType metric = METRIC_L2);
 
    IndexScalarQuantizer(
            int d,
            bool is_include_one_attribute,
            ScalarQuantizer::QuantizerType qtype,
            MetricType metric = METRIC_L2);

    IndexScalarQuantizer(
            int d,
            bool is_include_two_attribute,
            bool mode_two,
            ScalarQuantizer::QuantizerType qtype,
            MetricType metric = METRIC_L2);            

    IndexScalarQuantizer();

    void train(idx_t n, const float* x) override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels,
            const SearchParameters* params = nullptr) const override;

    
    void search_with_one_attribute(
            idx_t n,
            const float* x,
            const float lower_attribute,
            const float upper_attribute,
            idx_t k,
            float* distances,
            idx_t* labels,
            float* out_attrs,
            const SearchParameters* params = nullptr) const override;

    void search_with_two_attribute(
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
            const SearchParameters* params = nullptr) const override;

    FlatCodesDistanceComputer* get_FlatCodesDistanceComputer() const override;

    /* standalone codec interface */
    void sa_encode(idx_t n, const float* x, uint8_t* bytes) const override;
    void sa_one_attribute_encode(idx_t n, const float* attr, uint8_t* bytes) const override;
    void sa_two_attribute_encode(idx_t n, const float* attr_first, const float* attr_second,
                                 uint8_t* bytes_first, uint8_t* bytes_second) const override;

    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;
    void sa_one_attribute_decode(idx_t n, const uint8_t* bytes, float* attr) const override;
    void sa_two_attribute_decode(idx_t n, const uint8_t* bytes_first, const uint8_t* bytes_second,
                                 float* attr_first, float* attr_second) const override;
};

/** An IVF implementation where the components of the residuals are
 * encoded with a scalar quantizer. All distance computations
 * are asymmetric, so the encoded vectors are decoded and approximate
 * distances are computed.
 */

struct IndexIVFScalarQuantizer : IndexIVF {
    ScalarQuantizer sq;

    IndexIVFScalarQuantizer(
            Index* quantizer,
            size_t d,
            size_t nlist,
            ScalarQuantizer::QuantizerType qtype,
            MetricType metric = METRIC_L2,
            bool by_residual = true);

    IndexIVFScalarQuantizer(
            Index* quantizer,
            size_t d,
            size_t nlist,
            bool is_include_one_attribute,
            ScalarQuantizer::QuantizerType qtype,
            MetricType metric = METRIC_L2,
            bool by_residual = true);

    IndexIVFScalarQuantizer(
            Index* quantizer,
            size_t d,
            size_t nlist,
            bool is_include_two_attribute,
            bool mode_two,
            ScalarQuantizer::QuantizerType qtype,
            MetricType metric = METRIC_L2,
            bool by_residual = true);

    IndexIVFScalarQuantizer();

    void train_encoder(idx_t n, const float* x, const idx_t* assign) override;

    idx_t train_encoder_num_vectors() const override;

    void encode_vectors(
            idx_t n,
            const float* x,
            const idx_t* list_nos,
            uint8_t* codes,
            bool include_listnos = false) const override;

    void add_core(
            idx_t n,
            const float* x,
            const idx_t* xids,
            const idx_t* precomputed_idx,
            void* inverted_list_context = nullptr) override;

    void add_core_with_one_attribute(
            idx_t n,
            const float* x,
            const float* attr,
            const idx_t* xids,
            const idx_t* precomputed_idx,
            void* inverted_list_context = nullptr) override;

    void add_core_with_two_attribute(
            idx_t n,
            const float* x,
            const float* attr_first,
            const float* attr_second,
            const idx_t* xids,
            const idx_t* precomputed_idx,
            void* inverted_list_context = nullptr) override;

    InvertedListScanner* get_InvertedListScanner(
            bool store_pairs,
            const IDSelector* sel) const override;

    void reconstruct_from_offset(int64_t list_no, int64_t offset, float* recons) const override;
    void reconstruct_one_attribute_from_offset(int64_t list_no, int64_t offset, float* recons_attr) const override;
    void reconstruct_two_attribute_from_offset(int64_t list_no, int64_t offset, 
                                               float* recons_attr_first, float* recons_attr_second) const override;

    /* standalone codec interface */
    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;
};

} // namespace faiss

#endif
