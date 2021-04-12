/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.
// -*- c++ -*-

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

/// Opaque type for referencing to a LinearTransform object
FAISS_DECLARE_CLASS_INHERITED(LinearTransform, VectorTransform)
FAISS_DECLARE_DESTRUCTOR(LinearTransform)

/// Getter for have_bias
FAISS_DECLARE_GETTER(LinearTransform, int, have_bias)

/// Getter for is_orthonormal
FAISS_DECLARE_GETTER(LinearTransform, int, is_orthonormal)

FAISS_DECLARE_CLASS_INHERITED(RandomRotationMatrix, LinearTransform)
FAISS_DECLARE_DESTRUCTOR(RandomRotationMatrix)

int faiss_RandomRotationMatrix_new_with(
        FaissRandomRotationMatrix** p_vt,
        int d_in,
        int d_out);

FAISS_DECLARE_CLASS_INHERITED(PCAMatrix, LinearTransform)
FAISS_DECLARE_DESTRUCTOR(PCAMatrix)

int faiss_PCAMatrix_new_with(
        FaissPCAMatrix** p_vt,
        int d_in,
        int d_out,
        float eigen_power,
        int random_rotation);

/// Getter for input dimension
FAISS_DECLARE_GETTER(PCAMatrix, int, d_in)

/// Getter for output dimension
FAISS_DECLARE_GETTER(PCAMatrix, int, d_out)

/// Getter for eigen_power
FAISS_DECLARE_GETTER(PCAMatrix, float, eigen_power)

/// Getter for random_rotation
FAISS_DECLARE_GETTER(PCAMatrix, int, random_rotation)

#ifdef __cplusplus
}
#endif

#endif