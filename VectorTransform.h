/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#ifndef FAISS_VECTOR_TRANSFORM_H
#define FAISS_VECTOR_TRANSFORM_H

/** Defines a few objects that apply transformations to a set of
 * vectors Often these are pre-processing steps.
 */

#include <vector>

#include "Index.h"


namespace faiss {


/** Any transformation applied on a set of vectors */
struct VectorTransform {

    typedef Index::idx_t idx_t;

    int d_in;      ///! input dimension
    int d_out;     ///! output dimension

    explicit VectorTransform (int d_in = 0, int d_out = 0):
    d_in(d_in), d_out(d_out), is_trained(true)
    {}


    /// set if the VectorTransform does not require training, or if
    /// training is done already
    bool is_trained;


    /** Perform training on a representative set of vectors. Does
     * nothing by default.
     *
     * @param n      nb of training vectors
     * @param x      training vecors, size n * d
     */
    virtual void train (idx_t n, const float *x);

    /** apply the random roation, return new allocated matrix
     * @param     x size n * d_in
     * @return    size n * d_out
     */
    float *apply (idx_t n, const float * x) const;

    /// same as apply, but result is pre-allocated
    virtual void apply_noalloc (idx_t n, const float * x,
                                float *xt) const = 0;

    /// reverse transformation. May not be implemented or may return
    /// approximate result
    virtual void reverse_transform (idx_t n, const float * xt,
                                    float *x) const;

    virtual ~VectorTransform () {}

};



/** Generic linear transformation, with bias term applied on output
 * y = A * x + b
 */
struct LinearTransform: VectorTransform {

    bool have_bias; ///! whether to use the bias term

    /// check if matrix A is orthonormal (enables reverse_transform)
    bool is_orthonormal;

    /// Transformation matrix, size d_out * d_in
    std::vector<float> A;

     /// bias vector, size d_out
    std::vector<float> b;

    /// both d_in > d_out and d_out < d_in are supported
    explicit LinearTransform (int d_in = 0, int d_out = 0,
                              bool have_bias = false);

    /// same as apply, but result is pre-allocated
    void apply_noalloc(idx_t n, const float* x, float* xt) const override;

    /// compute x = A^T * (x - b)
    /// is reverse transform if A has orthonormal lines
    void transform_transpose (idx_t n, const float * y,
                              float *x) const;

    /// works only if is_orthonormal
    void reverse_transform (idx_t n, const float * xt,
                            float *x) const override;

    /// compute A^T * A to set the is_orthonormal flag
    void set_is_orthonormal ();

    bool verbose;

    ~LinearTransform() override {}
};



/// Randomly rotate a set of vectors
struct RandomRotationMatrix: LinearTransform {

     /// both d_in > d_out and d_out < d_in are supported
     RandomRotationMatrix (int d_in, int d_out):
         LinearTransform(d_in, d_out, false) {}

     /// must be called before the transform is used
     void init(int seed);

     // intializes with an arbitrary seed
     void train(Index::idx_t n, const float* x) override;

     RandomRotationMatrix () {}
};


/** Applies a principal component analysis on a set of vectors,
 *  with optionally whitening and random rotation. */
struct PCAMatrix: LinearTransform {

    /** after transformation the components are multiplied by
     * eigenvalues^eigen_power
     *
     * =0: no whitening
     * =-0.5: full whitening
     */
    float eigen_power;

    /// random rotation after PCA
    bool random_rotation;

    /// ratio between # training vectors and dimension
    size_t max_points_per_d;

    /// try to distribute output eigenvectors in this many bins
    int balanced_bins;

    /// Mean, size d_in
    std::vector<float> mean;

    /// eigenvalues of covariance matrix (= squared singular values)
    std::vector<float> eigenvalues;

    /// PCA matrix, size d_in * d_in
    std::vector<float> PCAMat;

    // the final matrix is computed after random rotation and/or whitening
    explicit PCAMatrix (int d_in = 0, int d_out = 0,
                        float eigen_power = 0, bool random_rotation = false);

    /// train on n vectors. If n < d_in then the eigenvector matrix
    /// will be completed with 0s
    void train(Index::idx_t n, const float* x) override;

    /// copy pre-trained PCA matrix
    void copy_from (const PCAMatrix & other);

    /// called after mean, PCAMat and eigenvalues are computed
    void prepare_Ab();

};


struct ProductQuantizer;

/** Applies a rotation to align the dimensions with a PQ to minimize
 *  the reconstruction error. Can be used before an IndexPQ or an
 *  IndexIVFPQ. The method is the non-parametric version described in:
 *
 * "Optimized Product Quantization for Approximate Nearest Neighbor Search"
 * Tiezheng Ge, Kaiming He, Qifa Ke, Jian Sun, CVPR'13
 *
 */
struct OPQMatrix: LinearTransform {

    int M;          ///< nb of subquantizers
    int niter;      ///< Number of outer training iterations
    int niter_pq;   ///< Number of training iterations for the PQ
    int niter_pq_0; ///< same, for the first outer iteration

    /// if there are too many training points, resample
    size_t max_train_points;
    bool verbose;

    /// if non-NULL, use this product quantizer for training
    /// should be constructed with (d_out, M, _)
    ProductQuantizer * pq;

    /// if d2 != -1, output vectors of this dimension
    explicit OPQMatrix (int d = 0, int M = 1, int d2 = -1);

    void train(Index::idx_t n, const float* x) override;
};


/** remap dimensions for intput vectors, possibly inserting 0s
 * strictly speaking this is also a linear transform but we don't want
 * to compute it with matrix multiplies */
struct RemapDimensionsTransform: VectorTransform {

    /// map from output dimension to input, size d_out
    /// -1 -> set output to 0
    std::vector<int> map;

    RemapDimensionsTransform (int d_in, int d_out, const int *map);

    /// remap input to output, skipping or inserting dimensions as needed
    /// if uniform: distribute dimensions uniformly
    /// otherwise just take the d_out first ones.
    RemapDimensionsTransform (int d_in, int d_out, bool uniform = true);

    void apply_noalloc(idx_t n, const float* x, float* xt) const override;

    /// reverse transform correct only when the mapping is a permuation
    void reverse_transform(idx_t n, const float* xt, float* x) const override;

    RemapDimensionsTransform () {}
};


/** per-vector normalization */
struct NormalizationTransform: VectorTransform {
    float norm;

    explicit NormalizationTransform (int d, float norm = 2.0);
    NormalizationTransform ();

    void apply_noalloc(idx_t n, const float* x, float* xt) const override;

    /// Identity transform since norm is not revertible
    void reverse_transform(idx_t n, const float* xt, float* x) const override;
};

/** Subtract the mean of each component from the vectors. */
struct CenteringTransform: VectorTransform {

    /// Mean, size d_in = d_out
    std::vector<float> mean;

    explicit CenteringTransform (int d = 0);

    /// train on n vectors.
    void train(Index::idx_t n, const float* x) override;

    /// subtract the mean
    void apply_noalloc(idx_t n, const float* x, float* xt) const override;

    /// add the mean
    void reverse_transform (idx_t n, const float * xt,
                            float *x) const override;

};


/** Index that applies a LinearTransform transform on vectors before
 *  handing them over to a sub-index */
struct IndexPreTransform: Index {

    std::vector<VectorTransform *> chain;  ///! chain of tranforms
    Index * index;            ///! the sub-index

    bool own_fields;          ///! whether pointers are deleted in destructor

    explicit IndexPreTransform (Index *index);

    IndexPreTransform ();

    /// ltrans is the last transform before the index
    IndexPreTransform (VectorTransform * ltrans, Index * index);

    void prepend_transform (VectorTransform * ltrans);

    void train(idx_t n, const float* x) override;

    void add(idx_t n, const float* x) override;

    void add_with_ids(idx_t n, const float* x, const long* xids) override;

    void reset() override;

    /** removes IDs from the index. Not supported by all indexes.
     */
    long remove_ids(const IDSelector& sel) override;

    void search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels) const override;


    /* range search, no attempt is done to change the radius */
    void range_search (idx_t n, const float* x, float radius,
                       RangeSearchResult* result) const override;


    void reconstruct (idx_t key, float * recons) const override;

    void reconstruct_n (idx_t i0, idx_t ni, float *recons)
        const override;

    void search_and_reconstruct (idx_t n, const float *x, idx_t k,
                                 float *distances, idx_t *labels,
                                 float *recons) const override;

    /// apply the transforms in the chain. The returned float * may be
    /// equal to x, otherwise it should be deallocated.
    const float * apply_chain (idx_t n, const float *x) const;

    /// Reverse the transforms in the chain. May not be implemented for
    /// all transforms in the chain or may return approximate results.
    void reverse_chain (idx_t n, const float* xt, float* x) const;

    ~IndexPreTransform() override;
};


} // namespace faiss


#endif
