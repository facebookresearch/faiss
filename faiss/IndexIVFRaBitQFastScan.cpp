/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/IndexIVFRaBitQFastScan.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <memory>

#include <faiss/impl/CodePackerRaBitQ.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/RaBitQUtils.h>
#include <faiss/impl/RaBitQuantizerMultiBit.h>
#include <faiss/impl/fast_scan/FastScanDistancePostProcessing.h>
#include <faiss/impl/fast_scan/fast_scan.h>
#include <faiss/invlists/BlockInvertedLists.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/utils.h>

namespace faiss {

// Import shared utilities from RaBitQUtils
using rabitq_utils::ExtraBitsFactors;
using rabitq_utils::QueryFactorsData;
using rabitq_utils::SignBitFactors;
using rabitq_utils::SignBitFactorsWithError;

inline size_t roundup(size_t a, size_t b) {
    return (a + b - 1) / b * b;
}

/*********************************************************
 * IndexIVFRaBitQFastScan implementation
 *********************************************************/

IndexIVFRaBitQFastScan::IndexIVFRaBitQFastScan() = default;

IndexIVFRaBitQFastScan::IndexIVFRaBitQFastScan(
        Index* quantizer_in,
        size_t d_in,
        size_t nlist_in,
        MetricType metric,
        int bbs_in,
        bool own_invlists_in,
        uint8_t nb_bits)
        : IndexIVFFastScan(
                  quantizer_in,
                  d_in,
                  nlist_in,
                  0,
                  metric,
                  own_invlists_in),
          rabitq(d_in, metric, nb_bits) {
    FAISS_THROW_IF_NOT_MSG(d_in > 0, "Dimension must be positive");
    FAISS_THROW_IF_NOT_MSG(
            metric == METRIC_L2 || metric == METRIC_INNER_PRODUCT,
            "RaBitQ only supports L2 and Inner Product metrics");
    FAISS_THROW_IF_NOT_MSG(
            bbs_in % 32 == 0, "Batch size must be multiple of 32");
    FAISS_THROW_IF_NOT_MSG(quantizer_in != nullptr, "Quantizer cannot be null");

    by_residual = true;
    qb = 8; // RaBitQ quantization bits
    centered = false;

    // FastScan-specific parameters: 4 bits per sub-quantizer
    const size_t M_fastscan = (d_in + 3) / 4;
    constexpr size_t nbits_fastscan = 4;

    this->bbs = bbs_in;
    this->fine_quantizer = &rabitq;
    this->M = M_fastscan;
    this->nbits = nbits_fastscan;
    this->ksub = (1 << nbits_fastscan);
    this->M2 = roundup(M_fastscan, 2);

    // Compute code_size: bit_pattern + per-vector storage (factors/ex-codes)
    const size_t bit_pattern_size = (d + 7) / 8;
    this->code_size = bit_pattern_size + compute_per_vector_storage_size();

    is_trained = false;

    if (own_invlists) {
        replace_invlists(new BlockInvertedLists(nlist, get_CodePacker()), true);
    }
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

size_t IndexIVFRaBitQFastScan::compute_per_vector_storage_size() const {
    return rabitq_utils::compute_per_vector_storage_size(rabitq.nb_bits, d);
}

size_t IndexIVFRaBitQFastScan::fast_scan_code_size() const {
    return (d + 7) / 8;
}

size_t IndexIVFRaBitQFastScan::code_packing_stride() const {
    // Use code_size as stride to skip embedded factor data during packing
    return code_size;
}

CodePacker* IndexIVFRaBitQFastScan::get_CodePacker() const {
    return new CodePackerRaBitQ(M2, bbs, compute_per_vector_storage_size());
}

/*********************************************************
 * postprocess_packed_codes: write auxiliary data into blocks
 *********************************************************/

void IndexIVFRaBitQFastScan::postprocess_packed_codes(
        idx_t list_no,
        size_t list_offset,
        size_t n_added,
        const uint8_t* flat_codes) {
    auto* bil = dynamic_cast<BlockInvertedLists*>(invlists);
    FAISS_THROW_IF_NOT(bil);

    uint8_t* block_data = bil->codes[list_no].data();
    const size_t storage_size = compute_per_vector_storage_size();
    const size_t bit_pattern_size = (d + 7) / 8;
    const size_t packed_block_size = ((M2 + 1) / 2) * bbs;
    const size_t full_block_size = get_block_stride();

    for (size_t i = 0; i < n_added; i++) {
        const uint8_t* src = flat_codes + i * code_size + bit_pattern_size;
        uint8_t* dst = rabitq_utils::get_block_aux_ptr(
                block_data,
                list_offset + i,
                bbs,
                packed_block_size,
                full_block_size,
                storage_size);
        memcpy(dst, src, storage_size);
    }
}

/*********************************************************
 * train_encoder
 *********************************************************/

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

    const size_t ex_bits = rabitq.nb_bits - 1;

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

                const size_t bit_pattern_size = (d + 7) / 8;

                // Pack sign bits directly into FastScan format (inline)
                for (size_t j = 0; j < static_cast<size_t>(d); j++) {
                    const float or_minus_c = xi[j] - centroid[j];
                    if (or_minus_c > 0.0f) {
                        rabitq_utils::set_bit_fastscan(fastscan_code, j);
                    }
                }

                // Compute factors (with or without f_error depending on mode)
                SignBitFactorsWithError factors =
                        rabitq_utils::compute_vector_factors(
                                xi,
                                d,
                                centroid.data(),
                                rabitq.metric_type,
                                ex_bits > 0);

                if (ex_bits == 0) {
                    // 1-bit: store only SignBitFactors (8 bytes)
                    memcpy(fastscan_code + bit_pattern_size,
                           &factors,
                           sizeof(SignBitFactors));
                } else {
                    // Multi-bit: store full SignBitFactorsWithError (12 bytes)
                    memcpy(fastscan_code + bit_pattern_size,
                           &factors,
                           sizeof(SignBitFactorsWithError));

                    // Compute residual (needed for quantize_ex_bits)
                    std::vector<float> residual(d);
                    for (size_t j = 0; j < static_cast<size_t>(d); j++) {
                        residual[j] = xi[j] - centroid[j];
                    }

                    // Quantize ex-bits
                    const size_t ex_code_size = (d * ex_bits + 7) / 8;
                    uint8_t* ex_code = fastscan_code + bit_pattern_size +
                            sizeof(SignBitFactorsWithError);
                    ExtraBitsFactors ex_factors_temp;

                    rabitq_multibit::quantize_ex_bits(
                            residual.data(),
                            d,
                            rabitq.nb_bits,
                            ex_code,
                            ex_factors_temp,
                            rabitq.metric_type,
                            centroid.data());

                    memcpy(ex_code + ex_code_size,
                           &ex_factors_temp,
                           sizeof(ExtraBitsFactors));
                }

                // Include coarse codes if requested
                if (include_listnos) {
                    encode_listno(list_no, code_out);
                }
            }
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

    if (metric_type == MetricType::METRIC_INNER_PRODUCT &&
        original_query != nullptr) {
        query_factors.qr_norm_L2sqr = fvec_norm_L2sqr(original_query, d);
        query_factors.q_dot_c = query_factors.qr_norm_L2sqr -
                fvec_inner_product(original_query, residual, d);
    }

    const size_t ex_bits = rabitq.nb_bits - 1;
    if (ex_bits > 0) {
        query_factors.rotated_q = rotated_q;
    }

    if (centered) {
        const float max_code_value = (1 << qb) - 1;

        for (size_t m = 0; m < M; m++) {
            const size_t dim_start = m * 4;

            for (int code_val = 0; code_val < 16; code_val++) {
                float xor_contribution = 0.0f;

                for (size_t dim_offset = 0; dim_offset < 4; dim_offset++) {
                    const size_t dim_idx = dim_start + dim_offset;

                    if (dim_idx < static_cast<size_t>(d)) {
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

                    if (dim_idx < static_cast<size_t>(d) &&
                        ((code_val >> dim_offset) & 1)) {
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

void IndexIVFRaBitQFastScan::search_preassigned(
        idx_t n,
        const float* x,
        idx_t k,
        const idx_t* assign,
        const float* centroid_dis,
        float* distances,
        idx_t* labels,
        bool store_pairs,
        const IVFSearchParameters* params,
        IndexIVFStats* stats) const {
    FAISS_THROW_IF_NOT(is_trained);
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT_MSG(
            !store_pairs, "store_pairs not supported for RaBitQFastScan");
    FAISS_THROW_IF_NOT_MSG(!stats, "stats not supported for this index");

    size_t cur_nprobe = this->nprobe;
    if (params) {
        FAISS_THROW_IF_NOT(params->max_codes == 0);
        cur_nprobe = params->nprobe;
    }

    std::vector<QueryFactorsData> query_factors_storage(n * cur_nprobe);
    FastScanDistancePostProcessing context;
    context.query_factors = query_factors_storage.data();
    context.nprobe = cur_nprobe;

    const CoarseQuantized cq = {cur_nprobe, centroid_dis, assign};
    search_dispatch_implem(n, x, k, distances, labels, cq, context, params);
}

void IndexIVFRaBitQFastScan::compute_LUT(
        size_t n,
        const float* x,
        const CoarseQuantized& cq,
        AlignedTable<float>& dis_tables,
        AlignedTable<float>& biases,
        const FastScanDistancePostProcessing& context) const {
    FAISS_THROW_IF_NOT(is_trained);
    FAISS_THROW_IF_NOT(by_residual);

    size_t cq_nprobe = cq.nprobe;

    size_t dim12 = 16 * M;

    dis_tables.resize(n * cq_nprobe * dim12);
    biases.resize(n * cq_nprobe);

    if (n * cq_nprobe > 0) {
        memset(biases.get(), 0, sizeof(float) * n * cq_nprobe);
    }
    std::unique_ptr<float[]> xrel(new float[n * cq_nprobe * d]);

#pragma omp parallel for if (n * cq_nprobe > 1000)
    for (idx_t ij = 0; ij < static_cast<idx_t>(n * cq_nprobe); ij++) {
        idx_t i = ij / cq_nprobe;
        float* xij = &xrel[ij * d];
        idx_t cij = cq.ids[ij];

        if (cij >= 0) {
            quantizer->compute_residual(x + i * d, xij, cij);

            // Create QueryFactorsData for this query-list combination
            QueryFactorsData query_factors_data;

            compute_residual_LUT(
                    xij,
                    query_factors_data,
                    dis_tables.get() + ij * dim12,
                    x + i * d);

            // Store query factors using compact indexing (ij directly)
            if (context.query_factors != nullptr) {
                context.query_factors[ij] = query_factors_data;
            }

        } else {
            memset(xij, -1, sizeof(float) * d);
            memset(dis_tables.get() + ij * dim12, -1, sizeof(float) * dim12);
        }
    }
}

void IndexIVFRaBitQFastScan::reconstruct_from_offset(
        int64_t list_no,
        int64_t offset,
        float* recons) const {
    // Get centroid for this list
    std::vector<float> centroid(d);
    quantizer->reconstruct(list_no, centroid.data());

    // Unpack bit pattern from packed format
    const size_t bit_pattern_size = (d + 7) / 8;
    std::vector<uint8_t> fastscan_code(bit_pattern_size, 0);

    InvertedLists::ScopedCodes list_codes(invlists, list_no);
    for (size_t m = 0; m < M; m++) {
        uint8_t c =
                pq4_get_packed_element(list_codes.get(), bbs, M2, offset, m);

        size_t byte_idx = m / 2;
        if (m % 2 == 0) {
            fastscan_code[byte_idx] =
                    (fastscan_code[byte_idx] & 0xF0) | (c & 0x0F);
        } else {
            fastscan_code[byte_idx] =
                    (fastscan_code[byte_idx] & 0x0F) | ((c & 0x0F) << 4);
        }
    }

    const size_t storage_size = compute_per_vector_storage_size();
    const size_t packed_block_size = ((M2 + 1) / 2) * bbs;
    const size_t full_block_size = get_block_stride();

    InvertedLists::ScopedCodes list_block_codes(invlists, list_no);
    const uint8_t* aux_ptr = rabitq_utils::get_block_aux_ptr(
            list_block_codes.get(),
            offset,
            bbs,
            packed_block_size,
            full_block_size,
            storage_size);

    const auto& base_factors =
            *reinterpret_cast<const SignBitFactors*>(aux_ptr);
    const float dp_multiplier = base_factors.dp_multiplier;

    // Decode residual directly using dp_multiplier
    std::vector<float> residual(d);
    decode_fastscan_to_residual(
            fastscan_code.data(), residual.data(), dp_multiplier);

    // Reconstruct: x = centroid + residual
    for (size_t j = 0; j < static_cast<size_t>(d); j++) {
        recons[j] = centroid[j] + residual[j];
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
    std::vector<float> centroid(d);
    std::vector<float> residual(d);
    const size_t bit_pattern_size = (d + 7) / 8;

#pragma omp parallel for if (n > 1000)
    for (idx_t i = 0; i < n; i++) {
        const uint8_t* code_i = bytes + i * total_code_size;
        float* x_i = x + i * d;

        idx_t list_no = decode_listno(code_i);

        if (list_no >= 0 && list_no < static_cast<idx_t>(nlist)) {
            quantizer->reconstruct(list_no, centroid.data());

            const uint8_t* fastscan_code = code_i + coarse_size;

            const uint8_t* factors_ptr = fastscan_code + bit_pattern_size;
            const auto& base_factors =
                    *reinterpret_cast<const SignBitFactors*>(factors_ptr);

            decode_fastscan_to_residual(
                    fastscan_code, residual.data(), base_factors.dp_multiplier);

            for (size_t j = 0; j < static_cast<size_t>(d); j++) {
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
        float dp_multiplier) const {
    memset(residual, 0, sizeof(float) * d);

    const float inv_d_sqrt = (d == 0) ? 1.0f : (1.0f / std::sqrt((float)d));

    for (size_t j = 0; j < static_cast<size_t>(d); j++) {
        bool bit_value = rabitq_utils::extract_bit_fastscan(fastscan_code, j);

        float bit_as_float = bit_value ? 1.0f : 0.0f;
        residual[j] = (bit_as_float - 0.5f) * dp_multiplier * 2 * inv_d_sqrt;
    }
}

std::unique_ptr<FastScanCodeScanner> IndexIVFRaBitQFastScan::make_knn_scanner(
        bool is_max,
        idx_t n,
        idx_t k,
        float* distances,
        idx_t* labels,
        const IDSelector* sel,
        int /*impl*/,
        const FastScanDistancePostProcessing& context) const {
    const bool is_multibit = (rabitq.nb_bits - 1) > 0;
    return rabitq_ivf_make_knn_scanner(
            is_max, this, n, k, distances, labels, sel, &context, is_multibit);
}

/*********************************************************
 * IVFRaBitQFastScanScanner implementation
 *********************************************************/

namespace {

/// Provides IVF scanner interface using FastScan's SIMD batch processing.
struct IVFRaBitQFastScanScanner : InvertedListScanner {
    using InvertedListScanner::scan_codes;
    [[maybe_unused]] static constexpr int impl = 10;
    static constexpr size_t nq = 1;

    const IndexIVFRaBitQFastScan& index;

    AlignedTable<uint8_t> dis_tables;
    AlignedTable<uint16_t> biases;
    /// [scale, offset] for converting uint16 to float
    std::array<float, 2> normalizers{};

    const float* xi = nullptr;

    QueryFactorsData query_factors;
    FastScanDistancePostProcessing context;

    std::unique_ptr<FlatCodesDistanceComputer> dc;
    std::vector<float> centroid;

    IVFRaBitQFastScanScanner(
            const IndexIVFRaBitQFastScan& index_in,
            bool store_pairs_in,
            const IDSelector* sel_in)
            : InvertedListScanner(store_pairs_in, sel_in), index(index_in) {
        this->keep_max = is_similarity_metric(index_in.metric_type);
    }

    void set_query(const float* query) override {
        this->xi = query;
    }

    void set_list(idx_t list_no_in, float coarse_dis_in) override {
        this->list_no = list_no_in;

        IndexIVFFastScan::CoarseQuantized cq{
                .nprobe = 1,
                .dis = &coarse_dis_in,
                .ids = &list_no_in,
        };

        // Set up context for use in scan_codes
        context = FastScanDistancePostProcessing{};
        context.query_factors = &query_factors;
        context.nprobe = 1;

        index.compute_LUT_uint8(
                1, xi, cq, dis_tables, biases, &normalizers[0], context);

        // Set up distance computer for distance_to_code
        centroid.resize(index.d);
        index.quantizer->reconstruct(list_no, centroid.data());
        dc.reset(index.rabitq.get_distance_computer(
                index.qb, centroid.data(), index.centered));
        dc->set_query(xi);
    }

    float distance_to_code(const uint8_t* code) const override {
        FAISS_THROW_IF_NOT_MSG(
                dc,
                "set_query and set_list must be called before distance_to_code");
        return dc->distance_to_code(code);
    }

   public:
    size_t scan_codes(
            size_t ntotal,
            const uint8_t* codes,
            const idx_t* ids,
            float* distances,
            idx_t* labels,
            size_t k) const override {
        // initialize the current iteration heap to the worst possible value of
        // the prior loop
        std::vector<float> curr_dists(k, distances[0]);
        std::vector<idx_t> curr_labels(k, labels[0]);

        auto scanner = index.make_knn_scanner(
                !keep_max,
                nq,
                k,
                curr_dists.data(),
                curr_labels.data(),
                sel,
                0,
                context);
        auto* handler = scanner->handler();

        int qmap1[1] = {0};
        handler->q_map = qmap1;
        handler->begin(&normalizers[0]);

        const uint8_t* LUT = dis_tables.get();
        handler->dbias = biases.get();
        handler->ntotal = ntotal;
        handler->id_map = ids;

        // RaBitQ needs list context for factor lookup.
        // If invlists is unavailable (e.g., own_invlists=false), fall back
        // to the codes pointer which already contains the block data.
        std::vector<int> probe_map = {0};
        handler->set_list_context(list_no, probe_map);
        if (!handler->list_codes_ptr) {
            handler->list_codes_ptr = codes;
        }

        scanner->accumulate_loop(
                1,
                roundup(ntotal, index.bbs),
                index.bbs,
                static_cast<int>(index.M2),
                codes,
                LUT,
                0,
                index.get_block_stride());

        // Combine results across iterations
        handler->end();
        if (keep_max) {
            minheap_addn(
                    k,
                    distances,
                    labels,
                    curr_dists.data(),
                    curr_labels.data(),
                    k);
        } else {
            maxheap_addn(
                    k,
                    distances,
                    labels,
                    curr_dists.data(),
                    curr_labels.data(),
                    k);
        }
        return handler->num_updates();
    }
};

} // anonymous namespace

InvertedListScanner* IndexIVFRaBitQFastScan::get_InvertedListScanner(
        bool store_pairs,
        const IDSelector* sel,
        const IVFSearchParameters*) const {
    return new IVFRaBitQFastScanScanner(*this, store_pairs, sel);
}

} // namespace faiss
