/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdint>
#include <vector>

#include <faiss/IndexIVFPQ.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/ProductQuantizer.h>
#include <faiss/utils/utils.h>

namespace faiss {
namespace pq_code_distance {

#define TIC t0 = get_cycles()
#define TOC get_cycles() - t0

/** QueryTables manages the various ways of searching an
 * IndexIVFPQ. The code contains a lot of branches, depending on:
 * - metric_type: are we computing L2 or Inner product similarity?
 * - by_residual: do we encode raw vectors or residuals?
 * - use_precomputed_table: are x_R|x_C tables precomputed?
 * - polysemous_ht: are we filtering with polysemous codes?
 */
struct QueryTables {
    /*****************************************************
     * General data from the IVFPQ
     *****************************************************/

    const IndexIVFPQ& ivfpq;
    const IVFSearchParameters* params;

    // copied from IndexIVFPQ for easier access
    int d;
    const ProductQuantizer& pq;
    MetricType metric_type;
    bool by_residual;
    int use_precomputed_table;
    int polysemous_ht;

    // pre-allocated data buffers
    float *sim_table, *sim_table_2;
    float *residual_vec, *decoded_vec;

    // single data buffer
    std::vector<float> mem;

    // for table pointers
    std::vector<const float*> sim_table_ptrs;

    explicit QueryTables(
            const IndexIVFPQ& ivfpq_in,
            const IVFSearchParameters* params_in);

    /*****************************************************
     * What we do when query is known
     *****************************************************/

    // field specific to query
    const float* qi = nullptr;

    // query-specific initialization
    void init_query(const float* qi_in);

    void init_query_IP();

    void init_query_L2();

    /*****************************************************
     * When inverted list is known: prepare computations
     *****************************************************/

    // fields specific to list
    idx_t key = 0;
    float coarse_dis = 0.0f;
    std::vector<uint8_t> q_code;

    uint64_t init_list_cycles;

    /// once we know the query and the centroid, we can prepare the
    /// sim_table that will be used for accumulation
    /// and dis0, the initial value
    float precompute_list_tables();

    float precompute_list_table_pointers();

    float dis0 = 0.0f;

    void init_list(idx_t list_no, float coarse_dis_in, int mode);

    float precompute_list_tables_IP();

    float precompute_list_tables_L2();

    float precompute_list_table_pointers_L2();
};

} // namespace pq_code_distance
} // namespace faiss
