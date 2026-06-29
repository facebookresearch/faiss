/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#ifndef THE_SIMD_LEVEL
#error "THE_SIMD_LEVEL must be defined before including IVFPQScanner_impl.h"
#endif

#include <faiss/IndexIVFPQ.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/impl/ResultHandler.h>
#include <faiss/impl/pq_code_distance/IVFPQ_QueryTables.h>
#include <faiss/impl/pq_code_distance/pq_code_distance-inl.h>
#include <faiss/impl/simd_dispatch.h>
#include <faiss/invlists/DirectMap.h>
#include <faiss/utils/distances_dispatch.h>
#include <faiss/utils/hamming.h>

namespace faiss {
namespace pq_code_distance {

template <class C, bool use_sel>
struct WrappedSearchResult {
    ResultHandler& res;
    size_t nup = 0;
    idx_t list_no;

    const idx_t* ids;
    const IDSelector* sel;

    WrappedSearchResult(
            idx_t list_no_in,
            const idx_t* ids_in,
            const IDSelector* sel_in,
            ResultHandler& res_in)
            : res(res_in), list_no(list_no_in), ids(ids_in), sel(sel_in) {}

    inline bool skip_entry(idx_t j) {
        return use_sel && !sel->is_member(ids[j]);
    }

    inline void add(idx_t j, float dis) {
        // Reached only for codes that passed skip_entry — i.e. distance
        // was actually computed for this code (post-filter).
        res.stats.scan_cnt++;
        if (C::cmp(res.threshold, dis)) {
            idx_t id = ids ? ids[j] : lo_build(this->list_no, j);
            if (res.add_result(dis, id)) {
                res.stats.nheap_updates++;
                nup++;
            }
        }
    }
};

/*****************************************************
 * Scaning the codes.
 * The scanning functions call their favorite precompute_*
 * function to precompute the tables they need.
 *****************************************************/
template <typename IDType, MetricType METRIC_TYPE, class PQCodeDist>
struct IVFPQScannerT : QueryTables {
    using PQDecoder = typename PQCodeDist::PQDecoder;
    const uint8_t* list_codes = nullptr;
    const IDType* list_ids;
    size_t list_size = 0;

    IVFPQScannerT(
            const IndexIVFPQ& ivfpq_in,
            const IVFSearchParameters* params_in)
            : QueryTables(ivfpq_in, params_in) {
        FAISS_THROW_IF_NOT(METRIC_TYPE == metric_type);
    }

    /*****************************************************
     * Scaning the codes: simple PQ scan.
     *****************************************************/

    // This is the baseline version of scan_list_with_tables().
    // It demonstrates what this function actually does.
    //
    // /// version of the scan where we use precomputed tables.
    // template <class SearchResultType>
    // void scan_list_with_table(
    //         size_t ncode,
    //         const uint8_t* codes,
    //         SearchResultType& res) const {
    //
    //     for (size_t j = 0; j < ncode; j++, codes += pq.code_size) {
    //         if (res.skip_entry(j)) {
    //             continue;
    //         }
    //         float dis = dis0 + PQCodeDist::distance_single_code(
    //             pq, sim_table, codes);
    //         res.add(j, dis);
    //     }
    // }

    // This is the modified version of scan_list_with_tables().
    // It was observed that doing manual unrolling of the loop that
    //    utilizes distance_single_code() speeds up the computations.

    /// version of the scan where we use precomputed tables.
    template <class SearchResultType>
    void scan_list_with_table(
            size_t ncode,
            const uint8_t* codes,
            SearchResultType& res) const {
        int counter = 0;

        size_t saved_j[4] = {0, 0, 0, 0};
        for (size_t j = 0; j < ncode; j++) {
            if (res.skip_entry(j)) {
                continue;
            }

            saved_j[0] = (counter == 0) ? j : saved_j[0];
            saved_j[1] = (counter == 1) ? j : saved_j[1];
            saved_j[2] = (counter == 2) ? j : saved_j[2];
            saved_j[3] = (counter == 3) ? j : saved_j[3];

            counter += 1;
            if (counter == 4) {
                float distance_0 = 0;
                float distance_1 = 0;
                float distance_2 = 0;
                float distance_3 = 0;
                PQCodeDist::distance_four_codes(
                        pq.M,
                        pq.nbits,
                        sim_table,
                        codes + saved_j[0] * pq.code_size,
                        codes + saved_j[1] * pq.code_size,
                        codes + saved_j[2] * pq.code_size,
                        codes + saved_j[3] * pq.code_size,
                        distance_0,
                        distance_1,
                        distance_2,
                        distance_3);

                res.add(saved_j[0], dis0 + distance_0);
                res.add(saved_j[1], dis0 + distance_1);
                res.add(saved_j[2], dis0 + distance_2);
                res.add(saved_j[3], dis0 + distance_3);
                counter = 0;
            }
        }

        if (counter >= 1) {
            float dis = dis0 +
                    PQCodeDist::distance_single_code(
                                pq.M,
                                pq.nbits,
                                sim_table,
                                codes + saved_j[0] * pq.code_size);
            res.add(saved_j[0], dis);
        }
        if (counter >= 2) {
            float dis = dis0 +
                    PQCodeDist::distance_single_code(
                                pq.M,
                                pq.nbits,
                                sim_table,
                                codes + saved_j[1] * pq.code_size);
            res.add(saved_j[1], dis);
        }
        if (counter >= 3) {
            float dis = dis0 +
                    PQCodeDist::distance_single_code(
                                pq.M,
                                pq.nbits,
                                sim_table,
                                codes + saved_j[2] * pq.code_size);
            res.add(saved_j[2], dis);
        }
    }

    /// tables are not precomputed, but pointers are provided to the
    /// relevant X_c|x_r tables
    template <class SearchResultType>
    void scan_list_with_pointer(
            size_t ncode,
            const uint8_t* codes,
            SearchResultType& res) const {
        for (size_t j = 0; j < ncode; j++, codes += pq.code_size) {
            if (res.skip_entry(j)) {
                continue;
            }
            PQDecoder decoder(codes, pq.nbits);
            float dis = dis0;
            const float* tab = sim_table_2;

            for (size_t m = 0; m < pq.M; m++) {
                int ci = decoder.decode();
                dis += sim_table_ptrs[m][ci] - 2 * tab[ci];
                tab += pq.ksub;
            }
            res.add(j, dis);
        }
    }

    /// nothing is precomputed: access residuals on-the-fly
    template <class SearchResultType>
    void scan_on_the_fly_dist(
            size_t ncode,
            const uint8_t* codes,
            SearchResultType& res) const {
        const float* dvec;
        float local_dis0 = 0;
        if (by_residual) {
            if (METRIC_TYPE == METRIC_INNER_PRODUCT) {
                ivfpq.quantizer->reconstruct(key, residual_vec);
                local_dis0 = fvec_inner_product_dispatch(residual_vec, qi, d);
            } else {
                ivfpq.quantizer->compute_residual(qi, residual_vec, key);
            }
            dvec = residual_vec;
        } else {
            dvec = qi;
            local_dis0 = 0;
        }

        for (size_t j = 0; j < ncode; j++, codes += pq.code_size) {
            if (res.skip_entry(j)) {
                continue;
            }
            pq.decode(codes, decoded_vec);

            float dis;
            if (METRIC_TYPE == METRIC_INNER_PRODUCT) {
                dis = local_dis0 +
                        fvec_inner_product_dispatch(decoded_vec, qi, d);
            } else {
                dis = fvec_L2sqr_dispatch(decoded_vec, dvec, d);
            }
            res.add(j, dis);
        }
    }

    /*****************************************************
     * Scanning codes with polysemous filtering
     *****************************************************/

    // This is the baseline version of scan_list_polysemous_hc().
    // It demonstrates what this function actually does.

    //     template <class HammingComputer, class SearchResultType>
    //     void scan_list_polysemous_hc(
    //             size_t ncode,
    //             const uint8_t* codes,
    //             SearchResultType& res) const {
    //         int ht = ivfpq.polysemous_ht;
    //         size_t n_hamming_pass = 0, nup = 0;
    //
    //         int code_size = pq.code_size;
    //
    //         HammingComputer hc(q_code.data(), code_size);
    //
    //         for (size_t j = 0; j < ncode; j++, codes += code_size) {
    //             if (res.skip_entry(j)) {
    //                 continue;
    //             }
    //             const uint8_t* b_code = codes;
    //             int hd = hc.hamming(b_code);
    //             if (hd < ht) {
    //                 n_hamming_pass++;
    //
    //                 float dis =
    //                         dis0 +
    //                         PQCodeDist::distance_single_code(
    //                             pq, sim_table, codes);
    //
    //                 res.add(j, dis);
    //             }
    //         }
    // #pragma omp critical
    //         { indexIVFPQ_stats.n_hamming_pass += n_hamming_pass; }
    //     }

    // This is the modified version of scan_list_with_tables().
    // It was observed that doing manual unrolling of the loop that
    //    utilizes distance_single_code() speeds up the computations.

    template <class HammingComputer, class SearchResultType>
    void scan_list_polysemous_hc(
            size_t ncode,
            const uint8_t* codes,
            SearchResultType& res) const {
        int ht = ivfpq.polysemous_ht;
        size_t n_hamming_pass = 0;

        int code_size = static_cast<int>(pq.code_size);

        size_t saved_j[8];
        int counter = 0;

        HammingComputer hc(q_code.data(), code_size);

        for (size_t j = 0; j < (ncode / 4) * 4; j += 4) {
            const uint8_t* b_code = codes + j * code_size;

            // Unrolling is a key. Basically, doing multiple popcount
            // operations one after another speeds things up.

            // 9999999 is just an arbitrary large number
            int hd0 = (res.skip_entry(j + 0))
                    ? 99999999
                    : hc.hamming(b_code + 0 * code_size);
            int hd1 = (res.skip_entry(j + 1))
                    ? 99999999
                    : hc.hamming(b_code + 1 * code_size);
            int hd2 = (res.skip_entry(j + 2))
                    ? 99999999
                    : hc.hamming(b_code + 2 * code_size);
            int hd3 = (res.skip_entry(j + 3))
                    ? 99999999
                    : hc.hamming(b_code + 3 * code_size);

            saved_j[counter] = j + 0;
            counter = (hd0 < ht) ? (counter + 1) : counter;
            saved_j[counter] = j + 1;
            counter = (hd1 < ht) ? (counter + 1) : counter;
            saved_j[counter] = j + 2;
            counter = (hd2 < ht) ? (counter + 1) : counter;
            saved_j[counter] = j + 3;
            counter = (hd3 < ht) ? (counter + 1) : counter;

            if (counter >= 4) {
                // process four codes at the same time
                n_hamming_pass += 4;

                float distance_0 = dis0;
                float distance_1 = dis0;
                float distance_2 = dis0;
                float distance_3 = dis0;
                PQCodeDist::distance_four_codes(
                        pq.M,
                        pq.nbits,
                        sim_table,
                        codes + saved_j[0] * pq.code_size,
                        codes + saved_j[1] * pq.code_size,
                        codes + saved_j[2] * pq.code_size,
                        codes + saved_j[3] * pq.code_size,
                        distance_0,
                        distance_1,
                        distance_2,
                        distance_3);

                res.add(saved_j[0], dis0 + distance_0);
                res.add(saved_j[1], dis0 + distance_1);
                res.add(saved_j[2], dis0 + distance_2);
                res.add(saved_j[3], dis0 + distance_3);

                //
                counter -= 4;
                saved_j[0] = saved_j[4];
                saved_j[1] = saved_j[5];
                saved_j[2] = saved_j[6];
                saved_j[3] = saved_j[7];
            }
        }

        for (int kk = 0; kk < counter; kk++) {
            n_hamming_pass++;

            float dis = dis0 +
                    PQCodeDist::distance_single_code(
                                pq.M,
                                pq.nbits,
                                sim_table,
                                codes + saved_j[kk] * pq.code_size);

            res.add(saved_j[kk], dis);
        }

        // process leftovers
        for (size_t j = (ncode / 4) * 4; j < ncode; j++) {
            if (res.skip_entry(j)) {
                continue;
            }
            const uint8_t* b_code = codes + j * code_size;
            int hd = hc.hamming(b_code);
            if (hd < ht) {
                n_hamming_pass++;

                float dis = dis0 +
                        PQCodeDist::distance_single_code(
                                    pq.M,
                                    pq.nbits,
                                    sim_table,
                                    codes + j * code_size);

                res.add(j, dis);
            }
        }

#pragma omp critical
        {
            indexIVFPQ_stats.n_hamming_pass += n_hamming_pass;
        }
    }

    template <class SearchResultType>
    void scan_list_polysemous(
            size_t ncode,
            const uint8_t* codes,
            SearchResultType& res) const {
        with_HammingComputer<PQCodeDist::simd_level>(
                pq.code_size, [&]<class HammingComputer>() {
                    this->scan_list_polysemous_hc<
                            HammingComputer,
                            SearchResultType>(ncode, codes, res);
                });
    }
};

/* We put as many parameters as possible in template. Hopefully the
 * gain in runtime is worth the code bloat.
 *
 * C is the comparator < or >, it is directly related to METRIC_TYPE.
 *
 * precompute_mode is how much we precompute (2 = precompute distance tables,
 * 1 = precompute pointers to distances, 0 = compute distances one by one).
 * Currently only 2 is supported
 *
 * use_sel: store or ignore the IDSelector
 */
template <MetricType METRIC_TYPE, class C, class PQCodeDist, bool use_sel>
struct IVFPQScanner : IVFPQScannerT<idx_t, METRIC_TYPE, PQCodeDist>,
                      InvertedListScanner {
    using InvertedListScanner::scan_codes;
    int precompute_mode;
    const IDSelector* sel;

    IVFPQScanner(
            const IndexIVFPQ& ivfpq_in,
            bool store_pairs_in,
            int precompute_mode_in,
            const IDSelector* sel_in)
            : IVFPQScannerT<idx_t, METRIC_TYPE, PQCodeDist>(ivfpq_in, nullptr),
              precompute_mode(precompute_mode_in),
              sel(sel_in) {
        this->store_pairs = store_pairs_in;
        this->keep_max = is_similarity_metric(METRIC_TYPE);
        this->code_size = this->pq.code_size;
    }

    void set_query(const float* query) override {
        this->init_query(query);
    }

    void set_list(idx_t list_no_in, float coarse_dis_in) override {
        this->list_no = list_no_in;
        this->init_list(list_no_in, coarse_dis_in, precompute_mode);
    }

    float distance_to_code(const uint8_t* code) const override {
        FAISS_THROW_IF_NOT(precompute_mode == 2);
        float dis = this->dis0 +
                PQCodeDist::distance_single_code(
                            this->pq.M, this->pq.nbits, this->sim_table, code);
        return dis;
    }

    size_t scan_codes(
            size_t ncode,
            const uint8_t* codes,
            const idx_t* ids,
            ResultHandler& handler) const override {
        WrappedSearchResult<C, use_sel> res(
                this->key,
                this->store_pairs ? nullptr : ids,
                this->sel,
                handler);

        if (this->polysemous_ht > 0) {
            FAISS_THROW_IF_NOT(precompute_mode == 2);
            this->scan_list_polysemous(ncode, codes, res);
        } else if (precompute_mode == 2) {
            this->scan_list_with_table(ncode, codes, res);
        } else if (precompute_mode == 1) {
            this->scan_list_with_pointer(ncode, codes, res);
        } else if (precompute_mode == 0) {
            this->scan_on_the_fly_dist(ncode, codes, res);
        } else {
            FAISS_THROW_MSG("bad precomp mode");
        }
        return res.nup;
    }
};

template <SIMDLevel SL>
InvertedListScanner* make_IVFPQInvertedListScanner(
        const IndexIVFPQ& ivfpq,
        bool store_pairs,
        const IDSelector* sel);

// NOLINTNEXTLINE(facebook-hte-MisplacedTemplateSpecialization)
template <>
InvertedListScanner* make_IVFPQInvertedListScanner<THE_SIMD_LEVEL>(
        const IndexIVFPQ& ivfpq,
        bool store_pairs,
        const IDSelector* sel) {
    auto make = [&]<class PQCodeDist, bool use_sel>() -> InvertedListScanner* {
        if (ivfpq.metric_type == METRIC_INNER_PRODUCT) {
            return new IVFPQScanner<
                    METRIC_INNER_PRODUCT,
                    CMin<float, idx_t>,
                    PQCodeDist,
                    use_sel>(ivfpq, store_pairs, 2, sel);
        } else if (ivfpq.metric_type == METRIC_L2) {
            return new IVFPQScanner<
                    METRIC_L2,
                    CMax<float, idx_t>,
                    PQCodeDist,
                    use_sel>(ivfpq, store_pairs, 2, sel);
        } else {
            FAISS_THROW_MSG("unsupported metric type");
        }
    };

    auto with_decoder = [&]<bool use_sel>() -> InvertedListScanner* {
        if (ivfpq.pq.nbits == 8) {
            return make.template
            operator()<PQCodeDistance<PQDecoder8, THE_SIMD_LEVEL>, use_sel>();
        } else if (ivfpq.pq.nbits == 16) {
            return make.template
            operator()<PQCodeDistance<PQDecoder16, THE_SIMD_LEVEL>, use_sel>();
        } else {
            return make.template operator()<
                    PQCodeDistance<PQDecoderGeneric, THE_SIMD_LEVEL>,
                    use_sel>();
        }
    };

    if (sel) {
        return with_decoder.template operator()<true>();
    } else {
        return with_decoder.template operator()<false>();
    }
}

} // namespace pq_code_distance
} // namespace faiss
