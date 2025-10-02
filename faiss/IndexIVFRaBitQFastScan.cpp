/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexIVFRaBitQFastScan.h>

#include <algorithm>
#include <cstdio>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/RaBitQUtils.h>
#include <faiss/impl/simd_result_handlers.h>
#include <faiss/invlists/BlockInvertedLists.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/utils.h>

namespace faiss {

// Import shared utilities from RaBitQUtils
using rabitq_utils::FactorsData;
using rabitq_utils::QueryFactorsData;

inline size_t roundup(size_t a, size_t b) {
    return (a + b - 1) / b * b;
}

/*********************************************************
 * IndexIVFRaBitQFastScan implementation
 *********************************************************/

IndexIVFRaBitQFastScan::IndexIVFRaBitQFastScan() = default;

IndexIVFRaBitQFastScan::IndexIVFRaBitQFastScan(
        Index* quantizer,
        size_t d,
        size_t nlist,
        MetricType metric,
        int bbs,
        bool own_invlists)
        : IndexIVFFastScan(quantizer, d, nlist, 0, metric, own_invlists),
          rabitq(d, metric) {
    FAISS_THROW_IF_NOT_MSG(d > 0, "Dimension must be positive");
    FAISS_THROW_IF_NOT_MSG(
            metric == METRIC_L2 || metric == METRIC_INNER_PRODUCT,
            "RaBitQ only supports L2 and Inner Product metrics");
    FAISS_THROW_IF_NOT_MSG(bbs % 32 == 0, "Batch size must be multiple of 32");
    FAISS_THROW_IF_NOT_MSG(quantizer != nullptr, "Quantizer cannot be null");

    by_residual = true;
    qb = 8; // RaBitQ quantization bits
    centered = false;

    // FastScan-specific parameters: 4 bits per sub-quantizer
    const size_t M_fastscan = (d + 3) / 4;
    constexpr size_t nbits_fastscan = 4;

    this->bbs = bbs;
    this->fine_quantizer = &rabitq;
    this->M = M_fastscan;
    this->nbits = nbits_fastscan;
    this->ksub = (1 << nbits_fastscan);
    this->M2 = roundup(M_fastscan, 2);
    this->code_size = M2 / 2;

    is_trained = false;

    if (own_invlists) {
        replace_invlists(new BlockInvertedLists(nlist, get_CodePacker()), true);
    }

    factors_storage.clear();
    query_factors_storage.clear();
}

// Constructor that converts an existing IndexIVFRaBitQ to FastScan format
IndexIVFRaBitQFastScan::IndexIVFRaBitQFastScan(
        const IndexIVFRaBitQ& orig,
        int /* bbs */)
        : IndexIVFFastScan(
                  orig.quantizer,
                  orig.d,
                  orig.nlist,
                  0,
                  orig.metric_type,
                  false),
          rabitq(orig.rabitq) {}

void IndexIVFRaBitQFastScan::add_with_ids(
        idx_t n,
        const float* x,
        const idx_t* xids) {
    FAISS_THROW_IF_NOT(is_trained);
    FAISS_THROW_IF_NOT(n > 0);
    FAISS_THROW_IF_NOT(x != nullptr);

    size_t start_global_idx = ntotal;

    IndexIVFFastScan::add_with_ids(n, x, xids);

    if (factors_storage.size() < ntotal) {
        try {
            factors_storage.resize(ntotal);
        } catch (const std::bad_alloc&) {
            FAISS_THROW_FMT(
                    "Failed to allocate factors storage for %zu vectors",
                    ntotal);
        }
    }

    std::vector<idx_t> assign(n);
    quantizer->assign(n, x, assign.data());

#pragma omp parallel if (n > 1000)
    {
        std::vector<float> centroid(d);

#pragma omp for
        for (idx_t i = 0; i < n; i++) {
            idx_t list_no = assign[i];
            if (list_no >= 0 && list_no < nlist) {
                quantizer->reconstruct(list_no, centroid.data());

                FactorsData factors_temp = rabitq_utils::compute_vector_factors(
                        x + i * d, d, centroid.data(), rabitq.metric_type);

                size_t global_idx = start_global_idx + i;
                factors_storage[global_idx] = factors_temp;
            }
        }
    }
}

void IndexIVFRaBitQFastScan::train_encoder(
        idx_t n,
        const float* x,
        const idx_t* assign) {
    FAISS_THROW_IF_NOT(n > 0);
    FAISS_THROW_IF_NOT(x != nullptr);
    FAISS_THROW_IF_NOT(assign != nullptr || !by_residual);

    rabitq.train(n, x);
    is_trained = true;
    init_code_packer();
}

void IndexIVFRaBitQFastScan::encode_vectors(
        idx_t n,
        const float* x,
        const idx_t* list_nos,
        uint8_t* codes,
        bool include_listnos) const {
    FAISS_THROW_IF_NOT(n > 0);
    FAISS_THROW_IF_NOT(x != nullptr);
    FAISS_THROW_IF_NOT(list_nos != nullptr);
    FAISS_THROW_IF_NOT(codes != nullptr);
    FAISS_THROW_IF_NOT(is_trained);

    size_t coarse_size = include_listnos ? coarse_code_size() : 0;
    size_t total_code_size = code_size + coarse_size;
    memset(codes, 0, total_code_size * n);

#pragma omp parallel if (n > 1000)
    {
        std::vector<float> centroid(d);

#pragma omp for
        for (idx_t i = 0; i < n; i++) {
            int64_t list_no = list_nos[i];

            if (list_no >= 0) {
                const float* xi = x + i * d;
                uint8_t* code_out = codes + i * total_code_size;
                uint8_t* fastscan_code = code_out + coarse_size;

                // Reconstruct centroid for residual computation
                quantizer->reconstruct(list_no, centroid.data());

                // Encode vector to FastScan format (no factor computation)
                encode_vector_to_fastscan(xi, centroid.data(), fastscan_code);

                // Include coarse codes if requested
                if (include_listnos) {
                    encode_listno(list_no, code_out);
                }
            }
        }
    }
}

void IndexIVFRaBitQFastScan::encode_vector_to_fastscan(
        const float* xi,
        const float* centroid,
        uint8_t* fastscan_code) const {
    memset(fastscan_code, 0, code_size);

    for (size_t j = 0; j < d; j++) {
        const float x_val = xi[j];
        const float centroid_val = (centroid != nullptr) ? centroid[j] : 0.0f;
        const float or_minus_c = x_val - centroid_val;
        const bool xb = (or_minus_c > 0.0f);

        if (xb) {
            rabitq_utils::set_bit_fastscan(fastscan_code, j);
        }
    }
}

bool IndexIVFRaBitQFastScan::lookup_table_is_3d() const {
    return true;
}

// Computes lookup table for residual vectors in RaBitQ FastScan format
void IndexIVFRaBitQFastScan::compute_residual_LUT(
        const float* residual,
        QueryFactorsData& query_factors,
        float* lut_out,
        const float* original_query) const {
    FAISS_THROW_IF_NOT(qb > 0 && qb <= 8);

    std::vector<float> rotated_q(d);
    std::vector<uint8_t> rotated_qq(d);

    // Use RaBitQUtils to compute query factors - eliminates code duplication
    query_factors = rabitq_utils::compute_query_factors(
            residual,
            d,
            nullptr,
            qb,
            centered,
            metric_type,
            rotated_q,
            rotated_qq);

    // Override query norm for inner product if original query is provided
    if (metric_type == MetricType::METRIC_INNER_PRODUCT &&
        original_query != nullptr) {
        query_factors.qr_norm_L2sqr = fvec_norm_L2sqr(original_query, d);
    }

    if (centered) {
        const float max_code_value = (1 << qb) - 1;

        for (size_t m = 0; m < M; m++) {
            const size_t dim_start = m * 4;

            for (int code_val = 0; code_val < 16; code_val++) {
                float xor_contribution = 0.0f;

                for (size_t dim_offset = 0; dim_offset < 4; dim_offset++) {
                    const size_t dim_idx = dim_start + dim_offset;

                    if (dim_idx < d) {
                        const bool db_bit = (code_val >> dim_offset) & 1;
                        const float query_value = rotated_qq[dim_idx];

                        xor_contribution += db_bit
                                ? (max_code_value - query_value)
                                : query_value;
                    }
                }

                lut_out[m * 16 + code_val] = xor_contribution;
            }
        }
    } else {
        for (size_t m = 0; m < M; m++) {
            const size_t dim_start = m * 4;

            for (int code_val = 0; code_val < 16; code_val++) {
                float inner_product = 0.0f;
                int popcount = 0;

                for (size_t dim_offset = 0; dim_offset < 4; dim_offset++) {
                    const size_t dim_idx = dim_start + dim_offset;

                    if (dim_idx < d && ((code_val >> dim_offset) & 1)) {
                        inner_product += rotated_qq[dim_idx];
                        popcount++;
                    }
                }
                lut_out[m * 16 + code_val] = query_factors.c1 * inner_product +
                        query_factors.c2 * popcount;
            }
        }
    }
}

void IndexIVFRaBitQFastScan::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(is_trained);

    query_factors_storage.clear();
    query_factors_storage.resize(n * nlist);

    // RaBitQ's specialized handlers require careful initialization that
    // conflicts with the multi-threaded approach in implementation 14.
    // Temporarily override implem to prevent implementation 14 from being used.
    if (implem == 14) {
        // Cast away const to temporarily modify implem for this search
        const_cast<IndexIVFRaBitQFastScan*>(this)->implem = 12;
    }

    IndexIVFFastScan::search(n, x, k, distances, labels, params);
}

void IndexIVFRaBitQFastScan::compute_LUT(
        size_t n,
        const float* x,
        const CoarseQuantized& cq,
        AlignedTable<float>& dis_tables,
        AlignedTable<float>& biases,
        idx_t query_offset) const {
    FAISS_THROW_IF_NOT(is_trained);
    FAISS_THROW_IF_NOT(by_residual);

    size_t nprobe = this->nprobe;

    size_t dim12 = 16 * M;

    dis_tables.resize(n * nprobe * dim12);
    biases.resize(n * nprobe);

    if (n * nprobe > 0) {
        memset(biases.get(), 0, sizeof(float) * n * nprobe);
    }
    std::unique_ptr<float[]> xrel(new float[n * nprobe * d]);

#pragma omp parallel for if (n * nprobe > 1000)
    for (idx_t ij = 0; ij < n * nprobe; ij++) {
        idx_t i = ij / nprobe;
        idx_t global_i = query_offset + i;
        float* xij = &xrel[ij * d];
        idx_t cij = cq.ids[ij];

        if (cij >= 0) {
            quantizer->compute_residual(x + i * d, xij, cij);

            idx_t storage_idx = global_i * nlist + cij;

            QueryFactorsData& query_factors =
                    query_factors_storage[storage_idx];

            compute_residual_LUT(
                    xij,
                    query_factors,
                    dis_tables.get() + ij * dim12,
                    x + i * d);

        } else {
            memset(xij, -1, sizeof(float) * d);
            memset(dis_tables.get() + ij * dim12, -1, sizeof(float) * dim12);
        }
    }
}

void IndexIVFRaBitQFastScan::sa_decode(idx_t n, const uint8_t* bytes, float* x)
        const {
    FAISS_THROW_IF_NOT(is_trained);
    FAISS_THROW_IF_NOT(n > 0);
    FAISS_THROW_IF_NOT(bytes != nullptr);
    FAISS_THROW_IF_NOT(x != nullptr);

    size_t coarse_size = coarse_code_size();
    size_t total_code_size = code_size + coarse_size;

#pragma omp parallel for if (n > 1000)
    for (idx_t i = 0; i < n; i++) {
        const uint8_t* code_i = bytes + i * total_code_size;
        float* x_i = x + i * d;

        idx_t list_no = decode_listno(code_i);

        if (list_no >= 0 && list_no < nlist) {
            std::vector<float> centroid(d);
            quantizer->reconstruct(list_no, centroid.data());

            const uint8_t* fastscan_code = code_i + coarse_size;

            std::vector<float> residual(d);
            decode_fastscan_to_residual(fastscan_code, residual.data(), i);

            for (size_t j = 0; j < d; j++) {
                x_i[j] = centroid[j] + residual[j];
            }
        } else {
            memset(x_i, 0, sizeof(float) * d);
        }
    }
}

void IndexIVFRaBitQFastScan::decode_fastscan_to_residual(
        const uint8_t* fastscan_code,
        float* residual,
        idx_t vector_idx) const {
    memset(residual, 0, sizeof(float) * d);

    const float inv_d_sqrt = (d == 0) ? 1.0f : (1.0f / std::sqrt((float)d));

    FAISS_THROW_IF_NOT_FMT(
            vector_idx < factors_storage.size(),
            "Vector index %ld out of bounds for factors storage (size: %zu)",
            vector_idx,
            factors_storage.size());

    const FactorsData& fac = factors_storage[vector_idx];

    for (size_t j = 0; j < d; j++) {
        // Use RaBitQUtils for consistent bit extraction
        bool bit_value = rabitq_utils::extract_bit_fastscan(fastscan_code, j);

        float bit_as_float = bit_value ? 1.0f : 0.0f;
        residual[j] =
                (bit_as_float - 0.5f) * fac.dp_multiplier * 2 * inv_d_sqrt;
    }
}

/*********************************************************
 * IndexIVFRaBitQFastScan::IVFRaBitQHeapHandler implementation
 *********************************************************/

template <class C>
IndexIVFRaBitQFastScan::IVFRaBitQHeapHandler<C>::IVFRaBitQHeapHandler(
        const IndexIVFRaBitQFastScan* idx,
        size_t nq_val,
        size_t k_val,
        float* distances,
        int64_t* labels,
        size_t global_offset)
        : simd_result_handlers::ResultHandlerCompare<C, true>(
                  nq_val,
                  0,
                  nullptr),
          index(idx),
          heap_distances(distances),
          heap_labels(labels),
          nq(nq_val),
          k(k_val),
          global_query_offset(global_offset) {
    current_list_no = 0;
    probe_indices.clear();
}

template <class C>
void IndexIVFRaBitQFastScan::IVFRaBitQHeapHandler<C>::handle(
        size_t q,
        size_t b,
        simd16uint16 d0,
        simd16uint16 d1) {
    // Store the original local query index before adjust_with_origin changes it
    size_t local_q = q;
    this->adjust_with_origin(q, d0, d1);

    ALIGNED(32) uint16_t d32tab[32];
    d0.store(d32tab);
    d1.store(d32tab + 16);

    float* const heap_dis = heap_distances + q * k;
    int64_t* const heap_ids = heap_labels + q * k;

    FAISS_THROW_IF_NOT_FMT(
            !probe_indices.empty() && local_q < probe_indices.size(),
            "set_list_context() must be called before handle() - probe_indices size: %zu, local_q: %zu, global_q: %zu",
            probe_indices.size(),
            local_q,
            q);

    size_t global_query_idx = global_query_offset + q;
    size_t storage_idx = global_query_idx * index->nlist + current_list_no;

    FAISS_THROW_IF_NOT_FMT(
            storage_idx < index->query_factors_storage.size(),
            "Query factors storage index %zu out of bounds (size: %zu, nlist: %zu, query: %zu, list: %zu)",
            storage_idx,
            index->query_factors_storage.size(),
            index->nlist,
            global_query_idx,
            current_list_no);

    const auto& query_factors = index->query_factors_storage[storage_idx];

    const float one_a =
            this->normalizers ? (1.0f / this->normalizers[2 * q]) : 1.0f;
    const float bias = this->normalizers ? this->normalizers[2 * q + 1] : 0.0f;

    uint64_t idx_base = this->j0 + b * 32;
    if (idx_base >= this->ntotal) {
        return;
    }

    size_t max_positions =
            std::min(static_cast<size_t>(32), this->ntotal - idx_base);

    // Process each candidate vector in the SIMD batch
    for (int j = 0; j < static_cast<int>(max_positions); j++) {
        const int64_t result_id = this->adjust_id(b, j);

        if (result_id < 0) {
            continue;
        }

        const float normalized_distance = d32tab[j] * one_a + bias;

        // Get database factors using global index (factors are stored by global
        // index)
        const auto& db_factors = index->factors_storage[result_id];
        float adjusted_distance;

        // Distance computation depends on quantization mode
        if (index->centered) {
            int64_t int_dot = ((1 << index->qb) - 1) * index->d;
            int_dot -= 2 * static_cast<int64_t>(normalized_distance);

            adjusted_distance = query_factors.qr_to_c_L2sqr +
                    db_factors.or_minus_c_l2sqr -
                    2 * db_factors.dp_multiplier * int_dot *
                            query_factors.int_dot_scale;

        } else {
            float final_dot = normalized_distance - query_factors.c34;
            adjusted_distance = db_factors.or_minus_c_l2sqr +
                    query_factors.qr_to_c_L2sqr -
                    2 * db_factors.dp_multiplier * final_dot;
        }

        // Convert L2 to inner product if needed
        if (query_factors.qr_norm_L2sqr != 0.0f) {
            adjusted_distance =
                    -0.5f * (adjusted_distance - query_factors.qr_norm_L2sqr);
        }

        if (Cfloat::cmp(heap_dis[0], adjusted_distance)) {
            heap_replace_top<Cfloat>(
                    k, heap_dis, heap_ids, adjusted_distance, result_id);
        }
    }
}

template <class C>
void IndexIVFRaBitQFastScan::IVFRaBitQHeapHandler<C>::set_list_context(
        size_t list_no,
        const std::vector<int>& probe_map) {
    current_list_no = list_no;
    probe_indices = probe_map;
}

template <class C>
void IndexIVFRaBitQFastScan::IVFRaBitQHeapHandler<C>::begin(
        const float* norms) {
    this->normalizers = norms;

#pragma omp parallel for
    for (int64_t q = 0; q < static_cast<int64_t>(nq); q++) {
        float* heap_dis = heap_distances + q * k;
        int64_t* heap_ids = heap_labels + q * k;

        heap_heapify<Cfloat>(k, heap_dis, heap_ids);
    }
}

template <class C>
void IndexIVFRaBitQFastScan::IVFRaBitQHeapHandler<C>::end() {
#pragma omp parallel for
    for (int64_t q = 0; q < static_cast<int64_t>(nq); q++) {
        float* heap_dis = heap_distances + q * k;
        int64_t* heap_ids = heap_labels + q * k;
        heap_reorder<Cfloat>(k, heap_dis, heap_ids);
    }
}

// Implementation of virtual make_knn_handler method
SIMDResultHandlerToFloat* IndexIVFRaBitQFastScan::make_knn_handler(
        bool is_max,
        int /* impl */,
        idx_t n,
        idx_t k,
        float* distances,
        idx_t* labels,
        const IDSelector* /* sel */,
        idx_t query_offset) const {
    if (is_max) {
        return new IVFRaBitQHeapHandler<CMax<uint16_t, int64_t>>(
                this, n, k, distances, labels, query_offset);
    } else {
        return new IVFRaBitQHeapHandler<CMin<uint16_t, int64_t>>(
                this, n, k, distances, labels, query_offset);
    }
}

// Explicit template instantiations
template struct IndexIVFRaBitQFastScan::IVFRaBitQHeapHandler<
        CMin<uint16_t, int64_t>>;
template struct IndexIVFRaBitQFastScan::IVFRaBitQHeapHandler<
        CMax<uint16_t, int64_t>>;

} // namespace faiss
