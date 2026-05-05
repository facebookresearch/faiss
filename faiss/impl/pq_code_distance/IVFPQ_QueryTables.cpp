/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/pq_code_distance/IVFPQ_QueryTables.h>

#include <faiss/Clustering.h>
#include <faiss/utils/distances_dispatch.h>

namespace faiss {
namespace pq_code_distance {

QueryTables::QueryTables(
        const IndexIVFPQ& ivfpq_in,
        const IVFSearchParameters* params_in)
        : ivfpq(ivfpq_in),
          params(params_in),
          d(ivfpq_in.d),
          pq(ivfpq_in.pq),
          metric_type(ivfpq_in.metric_type),
          by_residual(ivfpq_in.by_residual),
          use_precomputed_table(ivfpq_in.use_precomputed_table) {
    mem.resize(pq.ksub * pq.M * 2 + d * 2);
    sim_table = mem.data();
    sim_table_2 = sim_table + pq.ksub * pq.M;
    residual_vec = sim_table_2 + pq.ksub * pq.M;
    decoded_vec = residual_vec + d;

    // for polysemous
    polysemous_ht = ivfpq_in.polysemous_ht;
    if (auto ivfpq_params =
                dynamic_cast<const IVFPQSearchParameters*>(params_in)) {
        polysemous_ht = ivfpq_params->polysemous_ht;
    }
    if (polysemous_ht != 0) {
        q_code.resize(pq.code_size);
    }
    init_list_cycles = 0;
    sim_table_ptrs.resize(pq.M);
}

void QueryTables::init_query(const float* qi_in) {
    this->qi = qi_in;
    if (metric_type == METRIC_INNER_PRODUCT) {
        init_query_IP();
    } else {
        init_query_L2();
    }
    if (!by_residual && polysemous_ht != 0) {
        pq.compute_code(qi_in, q_code.data());
    }
}

void QueryTables::init_query_IP() {
    // precompute some tables specific to the query qi
    pq.compute_inner_prod_table(qi, sim_table);
}

void QueryTables::init_query_L2() {
    if (!by_residual) {
        pq.compute_distance_table(qi, sim_table);
    } else if (use_precomputed_table) {
        pq.compute_inner_prod_table(qi, sim_table_2);
    }
}

float QueryTables::precompute_list_tables() {
    float dis0 = 0;
    uint64_t t0;
    TIC;
    if (by_residual) {
        if (metric_type == METRIC_INNER_PRODUCT) {
            dis0 = precompute_list_tables_IP();
        } else {
            dis0 = precompute_list_tables_L2();
        }
    }
    init_list_cycles += TOC;
    return dis0;
}

float QueryTables::precompute_list_table_pointers() {
    float dis0 = 0;
    uint64_t t0;
    TIC;
    if (by_residual) {
        if (metric_type == METRIC_INNER_PRODUCT) {
            FAISS_THROW_MSG("not implemented");
        } else {
            dis0 = precompute_list_table_pointers_L2();
        }
    }
    init_list_cycles += TOC;
    return dis0;
}

void QueryTables::init_list(idx_t list_no, float coarse_dis_in, int mode) {
    this->key = list_no;
    this->coarse_dis = coarse_dis_in;

    if (mode == 2) {
        dis0 = precompute_list_tables();
    } else if (mode == 1) {
        dis0 = precompute_list_table_pointers();
    }
}

float QueryTables::precompute_list_tables_IP() {
    // prepare the sim_table that will be used for accumulation
    // and dis0, the initial value
    ivfpq.quantizer->reconstruct(key, decoded_vec);
    // decoded_vec = centroid
    float dis0 = fvec_inner_product_dispatch(qi, decoded_vec, d);

    if (polysemous_ht) {
        for (int i = 0; i < d; i++) {
            residual_vec[i] = qi[i] - decoded_vec[i];
        }
        pq.compute_code(residual_vec, q_code.data());
    }
    return dis0;
}

float QueryTables::precompute_list_tables_L2() {
    float dis0 = 0;

    if (use_precomputed_table == 0 || use_precomputed_table == -1) {
        ivfpq.quantizer->compute_residual(qi, residual_vec, key);
        pq.compute_distance_table(residual_vec, sim_table);

        if (polysemous_ht != 0) {
            pq.compute_code(residual_vec, q_code.data());
        }

    } else if (use_precomputed_table == 1) {
        dis0 = coarse_dis;

        fvec_madd_dispatch(
                pq.M * pq.ksub,
                ivfpq.precomputed_table.data() + key * pq.ksub * pq.M,
                -2.0,
                sim_table_2,
                sim_table);

        if (polysemous_ht != 0) {
            ivfpq.quantizer->compute_residual(qi, residual_vec, key);
            pq.compute_code(residual_vec, q_code.data());
        }

    } else if (use_precomputed_table == 2) {
        dis0 = coarse_dis;

        const MultiIndexQuantizer* miq =
                dynamic_cast<const MultiIndexQuantizer*>(ivfpq.quantizer);
        FAISS_THROW_IF_NOT(miq);
        const ProductQuantizer& cpq = miq->pq;
        size_t Mf = pq.M / cpq.M;

        const float* qtab = sim_table_2; // query-specific table
        float* ltab = sim_table;         // (output) list-specific table

        long k = key;
        for (size_t cm = 0; cm < cpq.M; cm++) {
            // compute PQ index
            size_t ki = k & ((uint64_t(1) << cpq.nbits) - 1);
            k >>= cpq.nbits;

            // get corresponding table
            const float* pc = ivfpq.precomputed_table.data() +
                    (ki * pq.M + cm * Mf) * pq.ksub;

            if (polysemous_ht == 0) {
                // sum up with query-specific table
                fvec_madd_dispatch(Mf * pq.ksub, pc, -2.0, qtab, ltab);
                ltab += Mf * pq.ksub;
                qtab += Mf * pq.ksub;
            } else {
                for (size_t m = cm * Mf; m < (cm + 1) * Mf; m++) {
                    q_code[m] = fvec_madd_and_argmin_dispatch(
                            pq.ksub, pc, -2, qtab, ltab);
                    pc += pq.ksub;
                    ltab += pq.ksub;
                    qtab += pq.ksub;
                }
            }
        }
    }

    return dis0;
}

float QueryTables::precompute_list_table_pointers_L2() {
    float dis0 = 0;

    if (use_precomputed_table == 1) {
        dis0 = coarse_dis;

        const float* s = ivfpq.precomputed_table.data() + key * pq.ksub * pq.M;
        for (size_t m = 0; m < pq.M; m++) {
            sim_table_ptrs[m] = s;
            s += pq.ksub;
        }
    } else if (use_precomputed_table == 2) {
        dis0 = coarse_dis;

        const MultiIndexQuantizer* miq =
                dynamic_cast<const MultiIndexQuantizer*>(ivfpq.quantizer);
        FAISS_THROW_IF_NOT(miq);
        const ProductQuantizer& cpq = miq->pq;
        size_t Mf = pq.M / cpq.M;

        long k = key;
        size_t m0 = 0;
        for (size_t cm = 0; cm < cpq.M; cm++) {
            size_t ki = k & ((uint64_t(1) << cpq.nbits) - 1);
            k >>= cpq.nbits;

            const float* pc = ivfpq.precomputed_table.data() +
                    (ki * pq.M + cm * Mf) * pq.ksub;

            for (size_t m = m0; m < m0 + Mf; m++) {
                sim_table_ptrs[m] = pc;
                pc += pq.ksub;
            }
            m0 += Mf;
        }
    } else {
        FAISS_THROW_MSG("need precomputed tables");
    }

    if (polysemous_ht) {
        FAISS_THROW_MSG("not implemented");
        // Not clear that it makes sense to implemente this,
        // because it costs M * ksub, which is what we wanted to
        // avoid with the tables pointers.
    }

    return dis0;
}

} // namespace pq_code_distance
} // namespace faiss
