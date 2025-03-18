/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef FAISS_IVFLIB_H
#define FAISS_IVFLIB_H

/** Since IVF (inverted file) indexes are of so much use for
 * large-scale use cases, we group a few functions related to them in
 * this small library. Most functions work both on IndexIVFs and
 * IndexIVFs embedded within an IndexPreTransform.
 */

#include <faiss/IndexBinaryIVF.h>
#include <faiss/IndexIVF.h>
#include <vector>

namespace faiss {

struct IndexIVFResidualQuantizer;
struct IndexResidualQuantizer;
struct ResidualQuantizer;

namespace ivflib {

/** check if two indexes have the same parameters and are trained in
 * the same way, otherwise throw. */
void check_compatible_for_merge(const Index* index1, const Index* index2);

/** get an IndexIVF from an index. The index may be an IndexIVF or
 * some wrapper class that encloses an IndexIVF
 *
 * throws an exception if this is not the case.
 */
const IndexIVF* extract_index_ivf(const Index* index);
IndexIVF* extract_index_ivf(Index* index);

/// same as above but returns nullptr instead of throwing on failure
const IndexIVF* try_extract_index_ivf(const Index* index);
IndexIVF* try_extract_index_ivf(Index* index);

/** Merge index1 into index0. Works on IndexIVF's and IndexIVF's
 *  embedded in a IndexPreTransform. On output, the index1 is empty.
 *
 * @param shift_ids: translate the ids from index1 to index0->prev_ntotal
 */
void merge_into(Index* index0, Index* index1, bool shift_ids);

/* Returns the cluster the embeddings belong to.
 *
 * @param index      Index, which should be an IVF index
 *                   (otherwise there are no clusters)
 * @param embeddings object descriptors for which the centroids should be found,
 *                   size num_objects * d
 * @param centroid_ids
 *                   cluster id each object belongs to, size num_objects
 */
void search_centroid(Index* index, const float* x, int n, idx_t* centroid_ids);

/* Returns the cluster the embeddings belong to.
 *
 * @param index      Index, which should be an IVF index
 *                   (otherwise there are no clusters)
 * @param query_centroid_ids
 *                   centroid ids corresponding to the query vectors (size n)
 * @param result_centroid_ids
 *                   centroid ids corresponding to the results (size n * k)
 * other arguments are the same as the standard search function
 */
void search_and_return_centroids(
        Index* index,
        size_t n,
        const float* xin,
        long k,
        float* distances,
        idx_t* labels,
        idx_t* query_centroid_ids,
        idx_t* result_centroid_ids);

/** A set of IndexIVFs concatenated together in a FIFO fashion.
 * at each "step", the oldest index slice is removed and a new index is added.
 */
struct SlidingIndexWindow {
    /// common index that contains the sliding window
    Index* index;

    /// InvertedLists of index
    ArrayInvertedLists* ils;

    /// number of slices currently in index
    int n_slice;

    /// same as index->nlist
    size_t nlist;

    /// cumulative list sizes at each slice
    std::vector<std::vector<size_t>> sizes;

    /// index should be initially empty and trained
    SlidingIndexWindow(Index* index);

    /** Add one index to the current index and remove the oldest one.
     *
     * @param sub_index        slice to swap in (can be NULL)
     * @param remove_oldest    if true, remove the oldest slices */
    void step(const Index* sub_index, bool remove_oldest);
};

/// Get a subset of inverted lists [i0, i1)
ArrayInvertedLists* get_invlist_range(const Index* index, long i0, long i1);

/// Set a subset of inverted lists
void set_invlist_range(Index* index, long i0, long i1, ArrayInvertedLists* src);

/** search an IndexIVF, possibly embedded in an IndexPreTransform with
 * given parameters. This is a way to set the nprobe and get
 * statdistics in a thread-safe way.
 *
 * Optionally returns (if non-nullptr):
 * - nb_dis: number of distances computed
 * - ms_per_stage: [0]: preprocessing time
 *                 [1]: coarse quantization,
 *                 [2]: list scanning
 */
void search_with_parameters(
        const Index* index,
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const IVFSearchParameters* params,
        size_t* nb_dis = nullptr,
        double* ms_per_stage = nullptr);

/** same as search_with_parameters but for range search */
void range_search_with_parameters(
        const Index* index,
        idx_t n,
        const float* x,
        float radius,
        RangeSearchResult* result,
        const IVFSearchParameters* params,
        size_t* nb_dis = nullptr,
        double* ms_per_stage = nullptr);

/** Build an IndexIVFResidualQuantizer from an ResidualQuantizer, using the
 * nlevel first components as coarse quantizer and the rest as codes in invlists
 */
IndexIVFResidualQuantizer* ivf_residual_from_quantizer(
        const ResidualQuantizer&,
        int nlevel);

/** add from codes. NB that the norm component is not used, so the code_size can
 * be provided.
 *
 * @param ivfrq      index to populate with the codes
 * @param codes      codes to add, size (ncode, code_size)
 * @param code_size  override the ivfrq's code_size, useful if the norm encoding
 *                   is different
 */
void ivf_residual_add_from_flat_codes(
        IndexIVFResidualQuantizer* ivfrq,
        size_t ncode,
        const uint8_t* codes,
        int64_t code_size = -1);

struct ShardingFunction {
    virtual int64_t operator()(int64_t i, int64_t shard_count) = 0;
    virtual ~ShardingFunction() = default;
    ShardingFunction() {}
    ShardingFunction(const ShardingFunction&) = default;
    ShardingFunction(ShardingFunction&&) = default;
    ShardingFunction& operator=(const ShardingFunction&) = default;
    ShardingFunction& operator=(ShardingFunction&&) = default;
};
struct DefaultShardingFunction : ShardingFunction {
    int64_t operator()(int64_t i, int64_t shard_count) override;
};

/**
 * Shards an IVF index centroids by the given sharding function, and writes
 * the index to the path given by filename_generator. The centroids must already
 * be added to the index quantizer.
 *
 * @param index             The IVF index containing centroids to shard.
 * @param shard_count       Number of shards.
 * @param filename_template Template for shard filenames.
 * @param sharding_function The function to shard by. The default is ith vector
 *                          mod shard_count.
 * @param generate_ids      Generates ids using IndexIDMap2. If true, ids will
 *                          match the default ids in the unsharded index.
 * @return                  The number of shards written.
 */
void shard_ivf_index_centroids(
        IndexIVF* index,
        int64_t shard_count = 20,
        const std::string& filename_template = "shard.%d.index",
        ShardingFunction* sharding_function = nullptr,
        bool generate_ids = false);

void shard_binary_ivf_index_centroids(
        faiss::IndexBinaryIVF* index,
        int64_t shard_count = 20,
        const std::string& filename_template = "shard.%d.index",
        ShardingFunction* sharding_function = nullptr,
        bool generate_ids = false);

} // namespace ivflib
} // namespace faiss

#endif
