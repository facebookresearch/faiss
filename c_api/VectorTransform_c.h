/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.
// -*- c -*-

#ifndef FAISS_VECTOR_TRANSFORM_C_H
#define FAISS_VECTOR_TRANSFORM_C_H

/** Defines a few objects that apply transformations to a set of
 * vectors Often these are pre-processing steps.
 */

#include "faiss_c.h"

#ifdef __cplusplus
extern "C" {
#endif

/// Opaque type for referencing to a VectorTransform object
FAISS_DECLARE_CLASS(VectorTransform)
FAISS_DECLARE_DESTRUCTOR(VectorTransform)

/// Getter for is_trained
FAISS_DECLARE_GETTER(VectorTransform, int, is_trained)

/// Getter for input dimension
FAISS_DECLARE_GETTER(VectorTransform, int, d_in)

/// Getter for output dimension
FAISS_DECLARE_GETTER(VectorTransform, int, d_out)

/** Perform training on a representative set of vectors
 *
 * @param vt     opaque pointer to VectorTransform object
 * @param n      nb of training vectors
 * @param x      training vectors, size n * d
 */
int faiss_VectorTransform_train(
        FaissVectorTransform* vt,
        idx_t n,
        const float* x);

/** apply the random rotation, return new allocated matrix
 * @param     x size n * d_in
 * @return    size n * d_out
 */
float* faiss_VectorTransform_apply(
        const FaissVectorTransform* vt,
        idx_t n,
        const float* x);

/** apply transformation and result is pre-allocated
 * @param     x size n * d_in
 * @param     xt size n * d_out
 */
void faiss_VectorTransform_apply_noalloc(
        const FaissVectorTransform* vt,
        idx_t n,
        const float* x,
        float* xt);

/// reverse transformation. May not be implemented or may return
/// approximate result
void faiss_VectorTransform_reverse_transform(
        const FaissVectorTransform* vt,
        idx_t n,
        const float* xt,
        float* x);

/// Opaque type for referencing to a LinearTransform object
FAISS_DECLARE_CLASS_INHERITED(LinearTransform, VectorTransform)
FAISS_DECLARE_DESTRUCTOR(LinearTransform)

/// compute x = A^T * (x - b)
/// is reverse transform if A has orthonormal lines
void faiss_LinearTransform_transform_transpose(
        const FaissLinearTransform* vt,
        idx_t n,
        const float* y,
        float* x);

/// compute A^T * A to set the is_orthonormal flag
void faiss_LinearTransform_set_is_orthonormal(FaissLinearTransform* vt);

/// Getter for have_bias
FAISS_DECLARE_GETTER(LinearTransform, int, have_bias)

/// Getter for is_orthonormal
FAISS_DECLARE_GETTER(LinearTransform, int, is_orthonormal)

FAISS_DECLARE_CLASS_INHERITED(RandomRotationMatrix, VectorTransform)
FAISS_DECLARE_DESTRUCTOR(RandomRotationMatrix)

int faiss_RandomRotationMatrix_new_with(
        FaissRandomRotationMatrix** p_vt,
        int d_in,
        int d_out);

FAISS_DECLARE_CLASS_INHERITED(PCAMatrix, VectorTransform)
FAISS_DECLARE_DESTRUCTOR(PCAMatrix)

int faiss_PCAMatrix_new_with(
        FaissPCAMatrix** p_vt,
        int d_in,
        int d_out,
        float eigen_power,
        int random_rotation);

/// Getter for eigen_power
FAISS_DECLARE_GETTER(PCAMatrix, float, eigen_power)

/// Getter for random_rotation
FAISS_DECLARE_GETTER(PCAMatrix, int, random_rotation)

FAISS_DECLARE_CLASS_INHERITED(ITQMatrix, VectorTransform)
FAISS_DECLARE_DESTRUCTOR(ITQMatrix)

int faiss_ITQMatrix_new_with(FaissITQMatrix** p_vt, int d);

FAISS_DECLARE_CLASS_INHERITED(ITQTransform, VectorTransform)
FAISS_DECLARE_DESTRUCTOR(ITQTransform)

int faiss_ITQTransform_new_with(
        FaissITQTransform** p_vt,
        int d_in,
        int d_out,
        int do_pca);

/// Getter for do_pca
FAISS_DECLARE_GETTER(ITQTransform, int, do_pca)

FAISS_DECLARE_CLASS_INHERITED(OPQMatrix, VectorTransform)
FAISS_DECLARE_DESTRUCTOR(OPQMatrix)

int faiss_OPQMatrix_new_with(FaissOPQMatrix** p_vt, int d, int M, int d2);

FAISS_DECLARE_GETTER_SETTER(OPQMatrix, int, verbose)
FAISS_DECLARE_GETTER_SETTER(OPQMatrix, int, niter)
FAISS_DECLARE_GETTER_SETTER(OPQMatrix, int, niter_pq)

FAISS_DECLARE_CLASS_INHERITED(RemapDimensionsTransform, VectorTransform)
FAISS_DECLARE_DESTRUCTOR(RemapDimensionsTransform)

int faiss_RemapDimensionsTransform_new_with(
        FaissRemapDimensionsTransform** p_vt,
        int d_in,
        int d_out,
        int uniform);

FAISS_DECLARE_CLASS_INHERITED(NormalizationTransform, VectorTransform)
FAISS_DECLARE_DESTRUCTOR(NormalizationTransform)

int faiss_NormalizationTransform_new_with(
        FaissNormalizationTransform** p_vt,
        int d,
        float norm);

FAISS_DECLARE_GETTER(NormalizationTransform, float, norm)

FAISS_DECLARE_CLASS_INHERITED(CenteringTransform, VectorTransform)
FAISS_DECLARE_DESTRUCTOR(CenteringTransform)

int faiss_CenteringTransform_new_with(FaissCenteringTransform** p_vt, int d);

#ifdef __cplusplus
}
#endif

#endif