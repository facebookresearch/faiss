/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/*
 * Per-ISA implementation of Hamming distance computation for
 * IndexIVFSpectralHash. Included once per SIMD TU with THE_SIMD_LEVEL
 * set to the desired SIMDLevel.
 */

#pragma once

#ifndef THE_SIMD_LEVEL
#error "THE_SIMD_LEVEL must be defined before including this file"
#endif

#include <faiss/utils/hamming_distance/hamming_computer.h>

#include <cmath>
#include <cstring>

#include <faiss/IndexIVFSpectralHash.h>
#include <faiss/VectorTransform.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/impl/binary_hamming/dispatch.h>
#include <faiss/utils/hamming.h>

namespace faiss {

namespace {

void binarize_with_freq_impl(
        size_t nbit,
        float freq,
        const float* x,
        const float* c,
        uint8_t* codes) {
    memset(codes, 0, (nbit + 7) / 8);
    for (size_t i = 0; i < nbit; i++) {
        float xf = (x[i] - c[i]);
        int64_t xi = int64_t(std::floor(xf * freq));
        int64_t bit = xi & 1;
        codes[i >> 3] |= bit << (i & 7);
    }
}

template <class HammingComputer>
struct IVFScanner : InvertedListScanner {
    using InvertedListScanner::scan_codes;
    const IndexIVFSpectralHash* index;
    size_t nbit;

    float period, freq;
    std::vector<float> q;
    std::vector<float> zero;
    std::vector<uint8_t> qcode;
    HammingComputer hc;

    IVFScanner(const IndexIVFSpectralHash* index_in, bool store_pairs_in)
            : index(index_in),
              nbit(index_in->nbit),
              period(index_in->period),
              freq(2.0 / index_in->period),
              q(nbit),
              zero(nbit),
              qcode(index_in->code_size),
              hc(qcode.data(), static_cast<int>(index_in->code_size)) {
        this->store_pairs = store_pairs_in;
        this->code_size = index->code_size;
        this->keep_max = is_similarity_metric(index->metric_type);
    }

    void set_query(const float* query) override {
        FAISS_THROW_IF_NOT(query);
        FAISS_THROW_IF_NOT(q.size() == static_cast<size_t>(nbit));
        index->vt->apply_noalloc(1, query, q.data());

        if (index->threshold_type == IndexIVFSpectralHash::Thresh_global) {
            binarize_with_freq_impl(
                    nbit, freq, q.data(), zero.data(), qcode.data());
            hc.set(qcode.data(), code_size);
        }
    }

    void set_list(idx_t list_no_in, float /*coarse_dis*/) override {
        this->list_no = list_no_in;
        if (index->threshold_type != IndexIVFSpectralHash::Thresh_global) {
            const float* c = index->trained.data() + list_no_in * nbit;
            binarize_with_freq_impl(nbit, freq, q.data(), c, qcode.data());
            hc.set(qcode.data(), code_size);
        }
    }

    float distance_to_code(const uint8_t* code) const final {
        return hc.hamming(code);
    }

    size_t scan_codes(
            size_t list_size,
            const uint8_t* codes,
            const idx_t* ids,
            ResultHandler& handler) const override {
        size_t nup = 0;
        // get_InvertedListScanner asserts no IDSelector, so every code
        // in the list has its distance computed.
        for (size_t j = 0; j < list_size; j++) {
            float dis = hc.hamming(codes);
            handler.stats.scan_cnt++;

            if (dis < handler.threshold) {
                int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                if (handler.add_result(dis, id)) {
                    handler.stats.nheap_updates++;
                    nup++;
                }
            }
            codes += code_size;
        }
        return nup;
    }

    void scan_codes_range(
            size_t list_size,
            const uint8_t* codes,
            const idx_t* ids,
            float radius,
            RangeQueryResult& result) const override {
        for (size_t j = 0; j < list_size; j++) {
            float dis = hc.hamming(codes);
            result.stats.scan_cnt++;
            if (dis < radius) {
                int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
                result.add(dis, id);
                result.stats.nheap_updates++;
            }
            codes += code_size;
        }
    }
};

} // anonymous namespace

template <>
InvertedListScanner* make_spectral_hash_scanner_fixSL<THE_SIMD_LEVEL>(
        int code_size,
        const IndexIVFSpectralHash* index,
        bool store_pairs) {
    return with_HammingComputer<THE_SIMD_LEVEL>(
            code_size, [&]<class HammingComputer>() -> InvertedListScanner* {
                return new IVFScanner<HammingComputer>(index, store_pairs);
            });
}

} // namespace faiss
