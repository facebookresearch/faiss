/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef FAISS_INDEX_PQ_H
#define FAISS_INDEX_PQ_H

#include <stdint.h>

#include <vector>

#include <faiss/IndexFlatCodes.h>
#include <faiss/impl/PolysemousTraining.h>
#include <faiss/impl/ProductQuantizer.h>
#include <faiss/impl/platform_macros.h>

namespace faiss {

/** Index based on a product quantizer. Stored vectors are
 * approximated by PQ codes. */
struct IndexPQ : IndexFlatCodes {
    /// The product quantizer used to encode the vectors
    ProductQuantizer pq;

    /** Constructor.
     *
     * @param d      dimensionality of the input vectors
     * @param M      number of subquantizers
     * @param nbits  number of bit per subvector index
     */
    IndexPQ(int d,        ///< dimensionality of the input vectors
            size_t M,     ///< number of subquantizers
            size_t nbits, ///< number of bit per subvector index
            MetricType metric = METRIC_L2);

    IndexPQ();

    void train(idx_t n, const float* x) override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels) const override;

    /* The standalone codec interface */
    void sa_encode(idx_t n, const float* x, uint8_t* bytes) const override;

    void sa_decode(idx_t n, const uint8_t* bytes, float* x) const override;

    FlatCodesDistanceComputer* get_FlatCodesDistanceComputer() const override;

    /******************************************************
     * Polysemous codes implementation
     ******************************************************/
    bool do_polysemous_training; ///< false = standard PQ

    /// parameters used for the polysemous training
    PolysemousTraining polysemous_training;

    /// how to perform the search in search_core
    enum Search_type_t {
        ST_PQ,                    ///< asymmetric product quantizer (default)
        ST_HE,                    ///< Hamming distance on codes
        ST_generalized_HE,        ///< nb of same codes
        ST_SDC,                   ///< symmetric product quantizer (SDC)
        ST_polysemous,            ///< HE filter (using ht) + PQ combination
        ST_polysemous_generalize, ///< Filter on generalized Hamming
    };

    Search_type_t search_type;

    // just encode the sign of the components, instead of using the PQ encoder
    // used only for the queries
    bool encode_signs;

    /// Hamming threshold used for polysemy
    int polysemous_ht;

    // actual polysemous search
    void search_core_polysemous(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels) const;

    /// prepare query for a polysemous search, but instead of
    /// computing the result, just get the histogram of Hamming
    /// distances. May be computed on a provided dataset if xb != NULL
    /// @param dist_histogram (M * nbits + 1)
    void hamming_distance_histogram(
            idx_t n,
            const float* x,
            idx_t nb,
            const float* xb,
            int64_t* dist_histogram);

    /** compute pairwise distances between queries and database
     *
     * @param n    nb of query vectors
     * @param x    query vector, size n * d
     * @param dis  output distances, size n * ntotal
     */
    void hamming_distance_table(idx_t n, const float* x, int32_t* dis) const;
};

/// statistics are robust to internal threading, but not if
/// IndexPQ::search is called by multiple threads
struct IndexPQStats {
    size_t nq;    // nb of queries run
    size_t ncode; // nb of codes visited

    size_t n_hamming_pass; // nb of passed Hamming distance tests (for polysemy)

    IndexPQStats() {
        reset();
    }
    void reset();
};

FAISS_API extern IndexPQStats indexPQ_stats;

/** Quantizer where centroids are virtual: they are the Cartesian
 *  product of sub-centroids. */
struct MultiIndexQuantizer : Index {
    ProductQuantizer pq;

    MultiIndexQuantizer(
            int d,         ///< dimension of the input vectors
            size_t M,      ///< number of subquantizers
            size_t nbits); ///< number of bit per subvector index

    void train(idx_t n, const float* x) override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels) const override;

    /// add and reset will crash at runtime
    void add(idx_t n, const float* x) override;
    void reset() override;

    MultiIndexQuantizer() {}

    void reconstruct(idx_t key, float* recons) const override;
};

/** MultiIndexQuantizer where the PQ assignmnet is performed by sub-indexes
 */
struct MultiIndexQuantizer2 : MultiIndexQuantizer {
    /// M Indexes on d / M dimensions
    std::vector<Index*> assign_indexes;
    bool own_fields;

    MultiIndexQuantizer2(int d, size_t M, size_t nbits, Index** indexes);

    MultiIndexQuantizer2(
            int d,
            size_t nbits,
            Index* assign_index_0,
            Index* assign_index_1);

    void train(idx_t n, const float* x) override;

    void search(
            idx_t n,
            const float* x,
            idx_t k,
            float* distances,
            idx_t* labels) const override;
};

} // namespace faiss

#endif
