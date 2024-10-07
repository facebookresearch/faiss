/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

/** Implementation of k-means clustering with many variants. */

#ifndef FAISS_CLUSTERING_H
#define FAISS_CLUSTERING_H
#include <faiss/Index.h>

#include <vector>

namespace faiss {

/** Class for the clustering parameters. Can be passed to the
 * constructor of the Clustering object.
 */
struct ClusteringParameters {
    /// number of clustering iterations
    int niter = 25;
    /// redo clustering this many times and keep the clusters with the best
    /// objective
    int nredo = 1;

    bool verbose = false;
    /// whether to normalize centroids after each iteration (useful for inner
    /// product clustering)
    bool spherical = false;
    /// round centroids coordinates to integer after each iteration?
    bool int_centroids = false;
    /// re-train index after each iteration?
    bool update_index = false;

    /// Use the subset of centroids provided as input and do not change them
    /// during iterations
    bool frozen_centroids = false;
    /// If fewer than this number of training vectors per centroid are provided,
    /// writes a warning. Note that fewer than 1 point per centroid raises an
    /// exception.
    int min_points_per_centroid = 39;
    /// to limit size of dataset, otherwise the training set is subsampled
    int max_points_per_centroid = 256;
    /// seed for the random number generator.
    /// negative values lead to seeding an internal rng with
    /// std::high_resolution_clock.
    int seed = 1234;

    /// when the training set is encoded, batch size of the codec decoder
    size_t decode_block_size = 32768;

    /// whether to check for NaNs in an input data
    bool check_input_data_for_NaNs = true;

    /// Whether to use splitmix64-based random number generator for subsampling,
    /// which is faster, but may pick duplicate points.
    bool use_faster_subsampling = false;
};

struct ClusteringIterationStats {
    float obj;   ///< objective values (sum of distances reported by index)
    double time; ///< seconds for iteration
    double time_search;      ///< seconds for just search
    double imbalance_factor; ///< imbalance factor of iteration
    int nsplit;              ///< number of cluster splits
};

/** K-means clustering based on assignment - centroid update iterations
 *
 * The clustering is based on an Index object that assigns training
 * points to the centroids. Therefore, at each iteration the centroids
 * are added to the index.
 *
 * On output, the centoids table is set to the latest version
 * of the centroids and they are also added to the index. If the
 * centroids table it is not empty on input, it is also used for
 * initialization.
 *
 */
struct Clustering : ClusteringParameters {
    size_t d; ///< dimension of the vectors
    size_t k; ///< nb of centroids

    /** centroids (k * d)
     * if centroids are set on input to train, they will be used as
     * initialization
     */
    std::vector<float> centroids;

    /// stats at every iteration of clustering
    std::vector<ClusteringIterationStats> iteration_stats;

    Clustering(int d, int k);
    Clustering(int d, int k, const ClusteringParameters& cp);

    /** run k-means training
     *
     * @param x          training vectors, size n * d
     * @param index      index used for assignment
     * @param x_weights  weight associated to each vector: NULL or size n
     */
    virtual void train(
            idx_t n,
            const float* x,
            faiss::Index& index,
            const float* x_weights = nullptr);

    /** run with encoded vectors
     *
     * win addition to train()'s parameters takes a codec as parameter
     * to decode the input vectors.
     *
     * @param codec      codec used to decode the vectors (nullptr =
     *                   vectors are in fact floats)
     */
    void train_encoded(
            idx_t nx,
            const uint8_t* x_in,
            const Index* codec,
            Index& index,
            const float* weights = nullptr);

    /// Post-process the centroids after each centroid update.
    /// includes optional L2 normalization and nearest integer rounding
    void post_process_centroids();

    virtual ~Clustering() {}
};

/** Exact 1D clustering algorithm
 *
 * Since it does not use an index, it does not overload the train() function
 */
struct Clustering1D : Clustering {
    explicit Clustering1D(int k);

    Clustering1D(int k, const ClusteringParameters& cp);

    void train_exact(idx_t n, const float* x);

    virtual ~Clustering1D() {}
};

struct ProgressiveDimClusteringParameters : ClusteringParameters {
    int progressive_dim_steps; ///< number of incremental steps
    bool apply_pca;            ///< apply PCA on input

    ProgressiveDimClusteringParameters();
};

/** generates an index suitable for clustering when called */
struct ProgressiveDimIndexFactory {
    /// ownership transferred to caller
    virtual Index* operator()(int dim);

    virtual ~ProgressiveDimIndexFactory() {}
};

/** K-means clustering with progressive dimensions used
 *
 * The clustering first happens in dim 1, then with exponentially increasing
 * dimension until d (I steps). This is typically applied after a PCA
 * transformation (optional). Reference:
 *
 * "Improved Residual Vector Quantization for High-dimensional Approximate
 * Nearest Neighbor Search"
 *
 * Shicong Liu, Hongtao Lu, Junru Shao, AAAI'15
 *
 * https://arxiv.org/abs/1509.05195
 */
struct ProgressiveDimClustering : ProgressiveDimClusteringParameters {
    size_t d; ///< dimension of the vectors
    size_t k; ///< nb of centroids

    /** centroids (k * d) */
    std::vector<float> centroids;

    /// stats at every iteration of clustering
    std::vector<ClusteringIterationStats> iteration_stats;

    ProgressiveDimClustering(int d, int k);
    ProgressiveDimClustering(
            int d,
            int k,
            const ProgressiveDimClusteringParameters& cp);

    void train(idx_t n, const float* x, ProgressiveDimIndexFactory& factory);

    virtual ~ProgressiveDimClustering() {}
};

/** simplified interface
 *
 * @param d dimension of the data
 * @param n nb of training vectors
 * @param k nb of output centroids
 * @param x training set (size n * d)
 * @param centroids output centroids (size k * d)
 * @return final quantization error
 */
float kmeans_clustering(
        size_t d,
        size_t n,
        size_t k,
        const float* x,
        float* centroids);

} // namespace faiss

#endif
