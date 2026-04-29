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

// out[code] = base + sum of v_i for each set bit in code.
inline void write_subset_sum_lut(
        float* out,
        float base,
        float v0,
        float v1,
        float v2,
        float v3) {
    out[0] = base;
    out[1] = base + v0;
    out[2] = base + v1;
    out[3] = base + v0 + v1;
    out[4] = base + v2;
    out[5] = base + v0 + v2;
    out[6] = base + v1 + v2;
    out[7] = base + v0 + v1 + v2;
    out[8] = base + v3;
    out[9] = base + v0 + v3;
    out[10] = base + v1 + v3;
    out[11] = base + v0 + v1 + v3;
    out[12] = base + v2 + v3;
    out[13] = base + v0 + v2 + v3;
    out[14] = base + v1 + v2 + v3;
    out[15] = base + v0 + v1 + v2 + v3;
}

// Computes lookup table for residual vectors in RaBitQ FastScan format
void IndexIVFRaBitQFastScan::compute_residual_LUT(
        const float* query,
        idx_t centroid_id,
        QueryFactorsData& query_factors,
        float* lut_out,
        uint8_t qb_param,
        bool centered_param,
        std::vector<float>& rotated_q,
        std::vector<float>& centroid_buf) const {
    const size_t d_val = static_cast<size_t>(d);
    FAISS_THROW_IF_NOT(d_val > 0);
    rotated_q.resize(d_val);
    centroid_buf.resize(d_val);
    std::vector<uint8_t> rotated_qq(d_val);

    // Compute residual
    quantizer->reconstruct(centroid_id, centroid_buf.data());
    for (size_t i = 0; i < d_val; i++) {
        rotated_q[i] = query[i] - centroid_buf[i];
    }

    // Compute query factors using shared utility
    query_factors = rabitq_utils::compute_query_factors(
            rotated_q.data(),
            d_val,
            nullptr,
            qb_param,
            centered_param,
            metric_type,
            rotated_q,
            rotated_qq);

    if (metric_type == MetricType::METRIC_INNER_PRODUCT) {
        query_factors.qr_norm_L2sqr = fvec_norm_L2sqr(query, d_val);
        query_factors.q_dot_c =
                fvec_inner_product(query, centroid_buf.data(), d_val);
    }

    if (rabitq.nb_bits > 1) {
        query_factors.rotated_q = rotated_q;
    }

    // Build LUT using branchless subset-sum construction
    const size_t d_sz = d_val;

    if (centered_param) {
        const float mcv = static_cast<float>((1 << qb_param) - 1);

        for (size_t m = 0; m < M; m++) {
            const size_t ds = m * 4;
            float* out = lut_out + m * 16;

            float base = 0.0f;
            float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
            if (ds + 0 < d_sz) {
                float q = rotated_qq[ds + 0];
                base += q;
                v0 = mcv - 2.0f * q;
            }
            if (ds + 1 < d_sz) {
                float q = rotated_qq[ds + 1];
                base += q;
                v1 = mcv - 2.0f * q;
            }
            if (ds + 2 < d_sz) {
                float q = rotated_qq[ds + 2];
                base += q;
                v2 = mcv - 2.0f * q;
            }
            if (ds + 3 < d_sz) {
                float q = rotated_qq[ds + 3];
                base += q;
                v3 = mcv - 2.0f * q;
            }

            write_subset_sum_lut(out, base, v0, v1, v2, v3);
        }
    } else {
        const float c1 = query_factors.c1;
        const float c2 = query_factors.c2;

        for (size_t m = 0; m < M; m++) {
            const size_t ds = m * 4;
            float* out = lut_out + m * 16;

            float v0 = 0.0f, v1 = 0.0f, v2 = 0.0f, v3 = 0.0f;
            if (ds + 0 < d_sz) {
                v0 = c1 * rotated_qq[ds + 0] + c2;
            }
            if (ds + 1 < d_sz) {
                v1 = c1 * rotated_qq[ds + 1] + c2;
            }
            if (ds + 2 < d_sz) {
                v2 = c1 * rotated_qq[ds + 2] + c2;
            }
            if (ds + 3 < d_sz) {
                v3 = c1 * rotated_qq[ds + 3] + c2;
            }

            write_subset_sum_lut(out, 0.0f, v0, v1, v2, v3);
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
    uint8_t used_qb = qb;
    bool used_centered = centered;
    if (params) {
        FAISS_THROW_IF_NOT(params->max_codes == 0);
        cur_nprobe = params->nprobe;
        if (auto rparams =
                    dynamic_cast<const IVFRaBitQSearchParameters*>(params)) {
            used_qb = rparams->qb;
            used_centered = rparams->centered;
        }
    }

    std::vector<QueryFactorsData> query_factors_storage(n * cur_nprobe);
    FastScanDistancePostProcessing context;
    context.query_factors = query_factors_storage.data();
    context.nprobe = cur_nprobe;
    context.qb = used_qb;
    context.centered = used_centered;

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

    // Use overridden qb/centered from context if provided, else index defaults
    const uint8_t used_qb = context.qb > 0 ? context.qb : qb;
    const bool used_centered = context.qb > 0 ? context.centered : centered;

    size_t cq_nprobe = cq.nprobe;

    size_t dim12 = 16 * M;

    dis_tables.resize(n * cq_nprobe * dim12);
    biases.resize(n * cq_nprobe);

    if (n * cq_nprobe > 0) {
        memset(biases.get(), 0, sizeof(float) * n * cq_nprobe);
    }
    // Use per-thread buffers instead of one O(n * nprobe * d) allocation.
    // rotated_q / centroid_buf keep their capacity across iterations so the
    // allocator is only hit once per thread.
#pragma omp parallel if (n * cq_nprobe > 1000)
    {
        std::vector<float> rotated_q(d);
        std::vector<float> centroid_buf(d);

#pragma omp for
        for (idx_t ij = 0; ij < static_cast<idx_t>(n * cq_nprobe); ij++) {
            idx_t i = ij / cq_nprobe;
            idx_t cij = cq.ids[ij];

            if (cij >= 0) {
                QueryFactorsData query_factors_data;

                compute_residual_LUT(
                        x + i * d,
                        cij,
                        query_factors_data,
                        dis_tables.get() + ij * dim12,
                        used_qb,
                        used_centered,
                        rotated_q,
                        centroid_buf);

                if (context.query_factors != nullptr) {
                    context.query_factors[ij] = std::move(query_factors_data);
                }

            } else {
                memset(dis_tables.get() + ij * dim12, 0, sizeof(float) * dim12);
            }
        }
    }
}

void IndexIVFRaBitQFastScan::compute_LUT_uint8(
        size_t n,
        const float* x,
        const CoarseQuantized& cq,
        AlignedTable<uint8_t>& dis_tables,
        AlignedTable<uint16_t>& biases,
        float* normalizers,
        const FastScanDistancePostProcessing& context) const {
    FAISS_THROW_IF_NOT(is_trained);
    FAISS_THROW_IF_NOT(by_residual);

    const uint8_t used_qb = context.qb > 0 ? context.qb : qb;
    const bool used_centered = context.qb > 0 ? context.centered : centered;
    const size_t cur_nprobe = cq.nprobe;
    const size_t dim12 = 16 * M;
    const size_t dim12_2 = 16 * M2;

    // Allocate only the uint8 output table (no full float table)
    dis_tables.resize(n * cur_nprobe * dim12_2);
    biases.resize(n * cur_nprobe);

#pragma omp parallel if (n > 1)
    {
        // Per-thread buffers reused across queries
        AlignedTable<float> lut_float(cur_nprobe * dim12);
        std::vector<float> rotated_q(d);
        std::vector<float> centroid_buf(d);
        std::vector<float> all_mins(cur_nprobe * M);
        std::vector<float> probe_b(cur_nprobe);

#pragma omp for schedule(dynamic)
        for (int64_t i = 0; i < static_cast<int64_t>(n); i++) {
            const float* xi = x + i * d;

            // Compute float LUT for all probes using fused path
            for (size_t j = 0; j < cur_nprobe; j++) {
                const size_t ij = i * cur_nprobe + j;
                idx_t cij = cq.ids[ij];

                if (cij >= 0) {
                    QueryFactorsData qf;
                    compute_residual_LUT(
                            xi,
                            cij,
                            qf,
                            lut_float.get() + j * dim12,
                            used_qb,
                            used_centered,
                            rotated_q,
                            centroid_buf);

                    if (context.query_factors != nullptr) {
                        context.query_factors[ij] = qf;
                    }
                } else {
                    memset(lut_float.get() + j * dim12,
                           0,
                           sizeof(float) * dim12);
                }
            }

            // Quantize float LUT to uint8 inline.
            // Mirrors quantize_LUT_and_bias 3D path with zero biases.
            // Single pass: find per-sub-q mins, max span, and per-probe b.
            float glob_max_span = -HUGE_VAL;
            float glob_max_dis = -HUGE_VAL;
            float glob_b = HUGE_VAL;
            for (size_t j2 = 0; j2 < cur_nprobe; j2++) {
                float b_j = 0;
                float span_j = 0;
                for (size_t m = 0; m < M; m++) {
                    const float* tab = lut_float.get() + j2 * dim12 + m * ksub;
                    float mn = tab[0], mx = tab[0];
                    for (size_t s = 1; s < ksub; s++) {
                        mn = std::min(mn, tab[s]);
                        mx = std::max(mx, tab[s]);
                    }
                    all_mins[j2 * M + m] = mn;
                    float span = mx - mn;
                    glob_max_span = std::max(glob_max_span, span);
                    b_j += mn;
                    span_j += span;
                }
                probe_b[j2] = b_j;
                glob_max_dis = std::max(glob_max_dis, span_j);
                glob_b = std::min(glob_b, b_j);
            }
            float a = std::min(255.0f / glob_max_span, 65535.0f / glob_max_dis);

            // Second pass: quantize LUT and compute biasq
            uint8_t* out_base = dis_tables.get() + i * cur_nprobe * dim12_2;
            uint16_t* bq = biases.get() + i * cur_nprobe;
            for (size_t j2 = 0; j2 < cur_nprobe; j2++) {
                for (size_t m = 0; m < M; m++) {
                    const float* tab = lut_float.get() + j2 * dim12 + m * ksub;
                    float mn = all_mins[j2 * M + m];
                    uint8_t* out = out_base + j2 * dim12_2 + m * ksub;
                    for (size_t s = 0; s < ksub; s++) {
                        out[s] = static_cast<uint8_t>(
                                std::roundf(a * (tab[s] - mn)));
                    }
                }
                memset(out_base + j2 * dim12_2 + M * ksub, 0, (M2 - M) * ksub);
                bq[j2] = static_cast<uint16_t>(
                        std::roundf(a * (probe_b[j2] - glob_b)));
            }
            normalizers[2 * i] = a;
            normalizers[2 * i + 1] = glob_b;
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
/// Buffers are allocated once and reused across set_list + scan_codes calls.
struct IVFRaBitQFastScanScanner : InvertedListScanner {
    using InvertedListScanner::scan_codes;
    static constexpr size_t nq = 1;

    const IndexIVFRaBitQFastScan& index;
    const uint8_t qb;
    const bool centered;

    const float* xi = nullptr;

    // Reusable buffers (allocated once in constructor)
    AlignedTable<uint8_t> dis_tables;
    AlignedTable<uint16_t> biases;
    std::array<float, 2> normalizers{};
    AlignedTable<float> lut_float;
    std::vector<float> rotated_q;
    std::vector<float> centroid_buf;
    QueryFactorsData query_factors;
    FastScanDistancePostProcessing context;
    std::vector<int> probe_map;
    std::vector<float> mins_buf;

    // Distance computer for distance_to_code (created in set_list)
    std::unique_ptr<FlatCodesDistanceComputer> dc;

    IVFRaBitQFastScanScanner(
            const IndexIVFRaBitQFastScan& index_in,
            bool store_pairs_in,
            const IDSelector* sel_in,
            uint8_t qb_in,
            bool centered_in)
            : InvertedListScanner(store_pairs_in, sel_in),
              index(index_in),
              qb(qb_in),
              centered(centered_in),
              lut_float(16 * index_in.M),
              rotated_q(index_in.d),
              centroid_buf(index_in.d),
              probe_map({0}),
              mins_buf(index_in.M) {
        this->keep_max = is_similarity_metric(index_in.metric_type);
        this->code_size = index_in.code_size;

        // Pre-allocate output tables for single probe
        dis_tables.resize(16 * index_in.M2);
        biases.resize(1);

        // Set up context once
        context.query_factors = &query_factors;
        context.nprobe = 1;
        context.qb = qb;
        context.centered = centered;
    }

    void set_query(const float* query) override {
        this->xi = query;
    }

    void set_list(idx_t list_no_in, float /*coarse_dis_in*/) override {
        this->list_no = list_no_in;

        index.compute_residual_LUT(
                xi,
                list_no_in,
                query_factors,
                lut_float.get(),
                qb,
                centered,
                rotated_q,
                centroid_buf);

        // Single-probe quantization (simplified inline, no OMP, no 3D)
        const size_t M = index.M;
        const size_t M2 = index.M2;
        const size_t ksub = index.ksub;

        float max_span = -HUGE_VAL;
        float max_dis = 0;
        float b = 0;
        float* mins = mins_buf.data();

        for (size_t m = 0; m < M; m++) {
            const float* tab = lut_float.get() + m * ksub;
            float mn = tab[0], mx = tab[0];
            for (size_t s = 1; s < ksub; s++) {
                mn = std::min(mn, tab[s]);
                mx = std::max(mx, tab[s]);
            }
            mins[m] = mn;
            float span = mx - mn;
            max_span = std::max(max_span, span);
            max_dis += span;
            b += mn;
        }

        float a = std::min(255.0f / max_span, 65535.0f / max_dis);
        uint8_t* out = dis_tables.get();
        for (size_t m = 0; m < M; m++) {
            const float* tab = lut_float.get() + m * ksub;
            for (size_t s = 0; s < ksub; s++) {
                out[m * ksub + s] = static_cast<uint8_t>(
                        std::roundf(a * (tab[s] - mins[m])));
            }
        }
        memset(out + M * ksub, 0, (M2 - M) * ksub);
        biases[0] = 0;
        normalizers[0] = a;
        normalizers[1] = b;

        // Create distance computer (reuses centroid_buf from
        // compute_residual_LUT)
        dc.reset(index.rabitq.get_distance_computer(
                qb, centroid_buf.data(), centered));
        dc->set_query(xi);
    }

    float distance_to_code(const uint8_t* code) const override {
        return dc->distance_to_code(code);
    }

    size_t scan_codes(
            size_t ntotal,
            const uint8_t* codes,
            const idx_t* ids,
            float* distances,
            idx_t* labels,
            size_t k) const override {
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
        handler->dbias = biases.get();
        handler->ntotal = ntotal;
        handler->id_map = ids;

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
                dis_tables.get(),
                0,
                index.get_block_stride());

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
        const IVFSearchParameters* search_params_in) const {
    uint8_t used_qb = qb;
    bool used_centered = centered;
    if (auto params = dynamic_cast<const IVFRaBitQSearchParameters*>(
                search_params_in)) {
        used_qb = params->qb;
        used_centered = params->centered;
    }
    return new IVFRaBitQFastScanScanner(
            *this, store_pairs, sel, used_qb, used_centered);
}

} // namespace faiss
