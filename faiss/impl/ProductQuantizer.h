/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#ifndef FAISS_PRODUCT_QUANTIZER_H
#define FAISS_PRODUCT_QUANTIZER_H

#include <stdint.h>

#include <vector>

#include <faiss/Clustering.h>
#include <faiss/impl/Quantizer.h>
#include <faiss/impl/platform_macros.h>
#include <faiss/utils/Heap.h>

namespace faiss {

/** Product Quantizer.
 * PQ is trained using k-means, minimizing the L2 distance to centroids.
 * PQ supports L2 and Inner Product search, however the quantization error is
 * biased towards L2 distance.
 */
struct ProductQuantizer : Quantizer {
    size_t M;     ///< number of subquantizers
    size_t nbits; ///< number of bits per quantization index

    // values derived from the above
    size_t dsub;  ///< dimensionality of each subvector
    size_t ksub;  ///< number of centroids for each subquantizer
    bool verbose; ///< verbose during training?

    /// initialization
    enum train_type_t {
        Train_default,
        Train_hot_start,     ///< the centroids are already initialized
        Train_shared,        ///< share dictionary across PQ segments
        Train_hypercube,     ///< initialize centroids with nbits-D hypercube
        Train_hypercube_pca, ///< initialize centroids with nbits-D hypercube
    };
    train_type_t train_type;

    ClusteringParameters cp; ///< parameters used during clustering

    /// if non-NULL, use this index for assignment (should be of size
    /// d / M)
    Index* assign_index;

    /// Centroid table, size M * ksub * dsub.
    /// Layout: (M, ksub, dsub)
    std::vector<float> centroids;

    /// Transposed centroid table, size M * ksub * dsub.
    /// Layout: (dsub, M, ksub)
    std::vector<float> transposed_centroids;

    /// Squared lengths of centroids, size M * ksub
    /// Layout: (M, ksub)
    std::vector<float> centroids_sq_lengths;

#ifdef COMPILE_SIMD_ARM_NEON
    /// NEON optimized centroid storage for PQ distance table computation.
    /// Supports three precision modes selectable at initialization time.
    struct NeonTransposedCentroids {
        /// Precision mode:
        ///   FP32: block-transposed fp32, layout (num_blocks, dsub, blocksize) per subq
        ///   FP16: quantized fp16, layout (M, ksub, dsub) — no transpose
        ///   INT8: quantized int8, layout (M, ksub, dsub) — no transpose
        enum class Precision { FP32, FP16, INT8 };

        Precision precision = Precision::FP32;
        size_t blocksize = 64;  ///< 16, 32, or 64 (only used for FP32)
        size_t ceil_ksub = 0;   ///< ksub rounded up to blocksize (only used for FP32)

        std::vector<float> data;    ///< FP32 block-transposed: M * ceil_ksub * dsub floats
        std::vector<uint16_t> data_f16; ///< FP16 quantized: M * ksub * dsub uint16_t
        std::vector<uint8_t> data_u8;   ///< INT8 quantized: M * ksub * dsub uint8_t
    };
    NeonTransposedCentroids neon_transposed;

    /// Initialize NEON optimized centroid storage.
    /// precision: 0=fp32 (block-transpose, default), 1=int8, 2=fp16
    /// blocksize: 16, 32, or 64 (only relevant for fp32 precision)
    /// Call after centroids are set (e.g. after train or index load).
    void initialize_neon_transposed_centroids(
            size_t blocksize = 64,
            int precision = 0);
#endif // COMPILE_SIMD_ARM_NEON

    /// return the centroids associated with subvector m
    float* get_centroids(size_t m, size_t i) {
        return &centroids[(m * ksub + i) * dsub];
    }
    const float* get_centroids(size_t m, size_t i) const {
        return &centroids[(m * ksub + i) * dsub];
    }

    // Train the product quantizer on a set of points. A clustering
    // can be set on input to define non-default clustering parameters
    void train(size_t n, const float* x) override;

    ProductQuantizer(
            size_t d,      /* dimensionality of the input vectors */
            size_t M,      /* number of subquantizers */
            size_t nbits); /* number of bit per subvector index */

    ProductQuantizer();

    /// compute derived values when d, M and nbits have been set
    void set_derived_values();

    /// Define the centroids for subquantizer m
    void set_params(const float* centroids, int m);

    /// Quantize one vector with the product quantizer
    void compute_code(const float* x, uint8_t* code) const;

    /// same as compute_code for several vectors
    void compute_codes(const float* x, uint8_t* codes, size_t n) const override;

    /// speed up code assignment using assign_index
    /// (non-const because the index is changed)
    void compute_codes_with_assign_index(
            const float* x,
            uint8_t* codes,
            size_t n);

    /// decode a vector from a given code (or n vectors if third argument)
    void decode(const uint8_t* code, float* x) const;
    void decode(const uint8_t* code, float* x, size_t n) const override;

    /// If we happen to have the distance tables precomputed, this is
    /// more efficient to compute the codes.
    void compute_code_from_distance_table(const float* tab, uint8_t* code)
            const;

    /** Compute distance table for one vector.
     *
     * The distance table for x = [x_0 x_1 .. x_(M-1)] is a M * ksub
     * matrix that contains
     *
     *   dis_table (m, j) = || x_m - c_(m, j)||^2
     *   for m = 0..M-1 and j = 0 .. ksub - 1
     *
     * where c_(m, j) is the centroid no j of sub-quantizer m.
     *
     * @param x         input vector size d
     * @param dis_table output table, size M * ksub
     */
    void compute_distance_table(const float* x, float* dis_table) const;

    void compute_inner_prod_table(const float* x, float* dis_table) const;

    /** compute distance table for several vectors
     * @param nx        nb of input vectors
     * @param x         input vector size nx * d
     * @param dis_table output table, size nx * M * ksub
     */
    void compute_distance_tables(size_t nx, const float* x, float* dis_tables)
            const;

    void compute_inner_prod_tables(size_t nx, const float* x, float* dis_tables)
            const;

    /** perform a search (L2 distance)
     * @param x        query vectors, size nx * d
     * @param nx       nb of queries
     * @param codes    database codes, size ncodes * code_size
     * @param ncodes   nb of nb vectors
     * @param res      heap array to store results (nh == nx)
     * @param init_finalize_heap  initialize heap (input) and sort (output)?
     */
    void search(
            const float* x,
            size_t nx,
            const uint8_t* codes,
            const size_t ncodes,
            float_maxheap_array_t* res,
            bool init_finalize_heap = true) const;

    /** same as search, but with inner product similarity */
    void search_ip(
            const float* x,
            size_t nx,
            const uint8_t* codes,
            const size_t ncodes,
            float_minheap_array_t* res,
            bool init_finalize_heap = true) const;

    /// Symmetric Distance Table
    std::vector<float> sdc_table;

    // initialize the SDC table from the centroids
    void compute_sdc_table();

    void search_sdc(
            const uint8_t* qcodes,
            size_t nq,
            const uint8_t* bcodes,
            const size_t ncodes,
            float_maxheap_array_t* res,
            bool init_finalize_heap = true) const;

    /// Sync transposed centroids with regular centroids. This call
    /// is needed if centroids were edited directly.
    void sync_transposed_centroids();

    /// Clear transposed centroids table so ones are no longer used.
    void clear_transposed_centroids();
};

// block size used in ProductQuantizer::compute_codes
FAISS_API extern int product_quantizer_compute_codes_bs;

#ifdef COMPILE_SIMD_ARM_NEON
/// NEON-optimized inner product table computation using blocksize-aligned transposed centroids.
void pq_compute_inner_prod_table_neon(
        const ProductQuantizer& pq,
        const float* x,
        float* dis_table);

/// NEON-optimized L2sqr distance table computation using blocksize-aligned transposed centroids.
void pq_compute_distance_table_neon(
        const ProductQuantizer& pq,
        const float* x,
        float* dis_table);
#endif // COMPILE_SIMD_ARM_NEON

/*************************************************
 * Objects to encode / decode strings of bits
 *************************************************/

struct PQEncoderGeneric {
    uint8_t* code; ///< code for this vector
    uint8_t offset;
    const int nbits; ///< number of bits per subquantizer index

    uint8_t reg;

    PQEncoderGeneric(uint8_t* code, int nbits, uint8_t offset = 0);

    void encode(uint64_t x);

    ~PQEncoderGeneric();
};

struct PQEncoder8 {
    uint8_t* code;
    PQEncoder8(uint8_t* code, int nbits);
    void encode(uint64_t x);
};

struct PQEncoder16 {
    uint16_t* code;
    PQEncoder16(uint8_t* code, int nbits);
    void encode(uint64_t x);
};

struct PQDecoderGeneric {
    const uint8_t* code;
    uint8_t offset;
    const int nbits;
    const uint64_t mask;
    uint8_t reg;
    PQDecoderGeneric(const uint8_t* code, int nbits);
    uint64_t decode();
};

struct PQDecoder8 {
    static const int nbits = 8;
    const uint8_t* code;
    PQDecoder8(const uint8_t* code, int nbits);
    uint64_t decode();
};

struct PQDecoder16 {
    static const int nbits = 16;
    const uint16_t* code;
    PQDecoder16(const uint8_t* code, int nbits);
    uint64_t decode();
};

} // namespace faiss

#include <faiss/impl/ProductQuantizer-inl.h>

#endif
