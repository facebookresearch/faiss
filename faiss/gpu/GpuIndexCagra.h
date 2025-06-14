// @lint-ignore-every LICENSELINT
/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */
/*
 * Copyright (c) 2024-2025, NVIDIA CORPORATION.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <faiss/IndexIVF.h>
#include <faiss/gpu/GpuIndex.h>
#include <faiss/gpu/GpuIndexIVFPQ.h>

#include <variant>
#include "faiss/Index.h"

namespace faiss {
struct IndexHNSWCagra;
}

namespace faiss {
namespace gpu {

template <typename data_t>
class CuvsCagra;

enum class graph_build_algo {
    /// Use IVF-PQ to build all-neighbors knn graph
    IVF_PQ,
    /// Use NN-Descent to build all-neighbors knn graph
    NN_DESCENT
};

/// A type for specifying how PQ codebooks are created.
enum class codebook_gen { // NOLINT
    PER_SUBSPACE = 0,     // NOLINT
    PER_CLUSTER = 1,      // NOLINT
};

struct IVFPQBuildCagraConfig {
    ///
    /// The number of inverted lists (clusters)
    ///
    /// Hint: the number of vectors per cluster (`n_rows/n_lists`) should be
    /// approximately 1,000 to 10,000.

    uint32_t n_lists = 1024;
    /// The number of iterations searching for kmeans centers (index building).
    uint32_t kmeans_n_iters = 20;
    /// The fraction of data to use during iterative kmeans building.
    double kmeans_trainset_fraction = 0.5;
    ///
    /// The bit length of the vector element after compression by PQ.
    ///
    /// Possible values: [4, 5, 6, 7, 8].
    ///
    /// Hint: the smaller the 'pq_bits', the smaller the index size and the
    /// better the search performance, but the lower the recall.

    uint32_t pq_bits = 8;
    ///
    /// The dimensionality of the vector after compression by PQ. When zero, an
    /// optimal value is selected using a heuristic.
    ///
    /// NB: `pq_dim  /// pq_bits` must be a multiple of 8.
    ///
    /// Hint: a smaller 'pq_dim' results in a smaller index size and better
    /// search performance, but lower recall. If 'pq_bits' is 8, 'pq_dim' can be
    /// set to any number, but multiple of 8 are desirable for good performance.
    /// If 'pq_bits' is not 8, 'pq_dim' should be a multiple of 8. For good
    /// performance, it is desirable that 'pq_dim' is a multiple of 32. Ideally,
    /// 'pq_dim' should be also a divisor of the dataset dim.

    uint32_t pq_dim = 0;
    /// How PQ codebooks are created.
    codebook_gen codebook_kind = codebook_gen::PER_SUBSPACE;
    ///
    /// Apply a random rotation matrix on the input data and queries even if
    /// `dim % pq_dim == 0`.
    ///
    /// Note: if `dim` is not multiple of `pq_dim`, a random rotation is always
    /// applied to the input data and queries to transform the working space
    /// from `dim` to `rot_dim`, which may be slightly larger than the original
    /// space and and is a multiple of `pq_dim` (`rot_dim % pq_dim == 0`).
    /// However, this transform is not necessary when `dim` is multiple of
    /// `pq_dim`
    ///   (`dim == rot_dim`, hence no need in adding "extra" data columns /
    ///   features).
    ///
    /// By default, if `dim == rot_dim`, the rotation transform is initialized
    /// with the identity matrix. When `force_random_rotation == true`, a random
    /// orthogonal transform matrix is generated regardless of the values of
    /// `dim` and `pq_dim`.

    bool force_random_rotation = false;
    ///
    /// By default, the algorithm allocates more space than necessary for
    /// individual clusters
    /// (`list_data`). This allows to amortize the cost of memory allocation and
    /// reduce the number of data copies during repeated calls to `extend`
    /// (extending the database).
    ///
    /// The alternative is the conservative allocation behavior; when enabled,
    /// the algorithm always allocates the minimum amount of memory required to
    /// store the given number of records. Set this flag to `true` if you prefer
    /// to use as little GPU memory for the database as possible.
    bool conservative_memory_allocation = false;
};

struct IVFPQSearchCagraConfig {
    /// The number of clusters to search.
    uint32_t n_probes = 20;
    ///
    /// Data type of look up table to be created dynamically at search time.
    ///
    /// Possible values: [CUDA_R_32F, CUDA_R_16F, CUDA_R_8U]
    ///
    /// The use of low-precision types reduces the amount of shared memory
    /// required at search time, so fast shared memory kernels can be used even
    /// for datasets with large dimansionality. Note that the recall is slightly
    /// degraded when low-precision type is selected.

    cudaDataType_t lut_dtype = CUDA_R_32F;
    ///
    /// Storage data type for distance/similarity computed at search time.
    ///
    /// Possible values: [CUDA_R_16F, CUDA_R_32F]
    ///
    /// If the performance limiter at search time is device memory access,
    /// selecting FP16 will improve performance slightly.

    cudaDataType_t internal_distance_dtype = CUDA_R_32F;
    ///
    /// Preferred fraction of SM's unified memory / L1 cache to be used as
    /// shared memory.
    ///
    /// Possible values: [0.0 - 1.0] as a fraction of the
    /// `sharedMemPerMultiprocessor`.
    ///
    /// One wants to increase the carveout to make sure a good GPU occupancy for
    /// the main search kernel, but not to keep it too high to leave some memory
    /// to be used as L1 cache. Note, this value is interpreted only as a hint.
    /// Moreover, a GPU usually allows only a fixed set of cache configurations,
    /// so the provided value is rounded up to the nearest configuration. Refer
    /// to the NVIDIA tuning guide for the target GPU architecture.
    ///
    /// Note, this is a low-level tuning parameter that can have drastic
    /// negative effects on the search performance if tweaked incorrectly.

    double preferred_shmem_carveout = 1.0;
};

struct GpuIndexCagraConfig : public GpuIndexConfig {
    /// Degree of input graph for pruning.
    size_t intermediate_graph_degree = 128;
    /// Degree of output graph.
    size_t graph_degree = 64;
    /// ANN algorithm to build knn graph.
    graph_build_algo build_algo = graph_build_algo::IVF_PQ;
    /// Number of Iterations to run if building with NN_DESCENT
    size_t nn_descent_niter = 20;

    std::shared_ptr<IVFPQBuildCagraConfig> ivf_pq_params{nullptr};
    std::shared_ptr<IVFPQSearchCagraConfig> ivf_pq_search_params{nullptr};
    float refine_rate = 2.0f;
    bool store_dataset = true;

    /// Whether to use MST optimization to guarantee graph connectivity.
    bool guarantee_connectivity = false;
};

enum class search_algo {
    /// For large batch sizes.
    SINGLE_CTA,
    /// For small batch sizes.
    MULTI_CTA,
    MULTI_KERNEL,
    AUTO
};

enum class hash_mode { HASH, SMALL, AUTO };

struct SearchParametersCagra : SearchParameters {
    /// Maximum number of queries to search at the same time (batch size). Auto
    /// select when 0.
    size_t max_queries = 0;

    /// Number of intermediate search results retained during the search.
    ///
    ///  This is the main knob to adjust trade off between accuracy and search
    /// speed. Higher values improve the search accuracy.

    size_t itopk_size = 64;

    /// Upper limit of search iterations. Auto select when 0.
    size_t max_iterations = 0;

    // In the following we list additional search parameters for fine tuning.
    // Reasonable default values are automatically chosen.

    /// Which search implementation to use.
    search_algo algo = search_algo::AUTO;

    /// Number of threads used to calculate a single distance. 4, 8, 16, or 32.

    size_t team_size = 0;

    /// Number of graph nodes to select as the starting point for the search in
    /// each iteration. aka search width?
    size_t search_width = 1;
    /// Lower limit of search iterations.
    size_t min_iterations = 0;

    /// Thread block size. 0, 64, 128, 256, 512, 1024. Auto selection when 0.
    size_t thread_block_size = 0;
    /// Hashmap type. Auto selection when AUTO.
    hash_mode hashmap_mode = hash_mode::AUTO;
    /// Lower limit of hashmap bit length. More than 8.
    size_t hashmap_min_bitlen = 0;
    /// Upper limit of hashmap fill rate. More than 0.1, less than 0.9.
    float hashmap_max_fill_rate = 0.5;

    /// Number of iterations of initial random seed node selection. 1 or more.

    uint32_t num_random_samplings = 1;
    /// Bit mask used for initial random seed node selection.
    uint64_t seed = 0x128394;
};

struct GpuIndexCagra : public GpuIndex {
   public:
    GpuIndexCagra(
            GpuResourcesProvider* provider,
            int dims,
            faiss::MetricType metric = faiss::METRIC_L2,
            GpuIndexCagraConfig config = GpuIndexCagraConfig());

    /// Trains CAGRA based on the given vector data and add them along with ids.
    /// NB: The use of the add function here is to build the CAGRA graph on
    /// the base dataset. Use this function when you want to add vectors with
    /// ids. Ref: https://github.com/facebookresearch/faiss/issues/4107
    void add(idx_t n, const float* x) override;
    void add(idx_t n, const void* x, NumericType numeric_type) override;

    /// Trains CAGRA based on the given vector data.
    /// NB: The use of the train function here is to build the CAGRA graph on
    /// the base dataset and is currently the only function to add the full set
    /// of vectors (without IDs) to the index. There is no external quantizer to
    /// be trained here.
    void train(idx_t n, const float* x) override;
    void train(idx_t n, const void* x, NumericType numeric_type) override;

    /// Initialize ourselves from the given CPU index; will overwrite
    /// all data in ourselves
    void copyFrom(const faiss::IndexHNSWCagra* index);
    void copyFrom(const faiss::IndexHNSWCagra* index, NumericType numeric_type);

    /// Copy ourselves to the given CPU index; will overwrite all data
    /// in the index instance
    void copyTo(faiss::IndexHNSWCagra* index) const;

    void reset() override;

    std::vector<idx_t> get_knngraph() const;

    faiss::NumericType get_numeric_type() const;

   protected:
    bool addImplRequiresIDs_() const override;

    void addImpl_(idx_t n, const float* x, const idx_t* ids) override;

    /// Called from GpuIndex for search
    void searchImpl_(
            idx_t n,
            const float* x,
            int k,
            float* distances,
            idx_t* labels,
            const SearchParameters* search_params) const override;
    void searchImpl_(
            idx_t n,
            const void* x,
            NumericType numeric_type,
            int k,
            float* distances,
            idx_t* labels,
            const SearchParameters* search_params) const override;

    /// Our configuration options
    const GpuIndexCagraConfig cagraConfig_;

    faiss::NumericType numeric_type_;

    /// Instance that we own; contains the inverted lists
    std::variant<
            std::monostate,
            std::shared_ptr<CuvsCagra<float>>,
            std::shared_ptr<CuvsCagra<half>>>
            index_;
};

} // namespace gpu
} // namespace faiss
