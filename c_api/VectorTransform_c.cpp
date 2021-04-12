/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.
// -*- c++ -*-

#include "VectorTransform_c.h"
#include <faiss/VectorTransform.h>
#include "macros_impl.h"

extern "C" {

DEFINE_DESTRUCTOR(VectorTransform)

DEFINE_GETTER(VectorTransform, int, is_trained)

DEFINE_GETTER(VectorTransform, int, d_in)

DEFINE_GETTER(VectorTransform, int, d_out)

int faiss_VectorTransform_train(FaissVectorTransform* vt, idx_t n, const float* x) {
    try {
        reinterpret_cast<faiss::VectorTransform*>(vt)->train(n, x);
    }
    CATCH_AND_HANDLE
}

/*********************************************
 * LinearTransform
 *********************************************/

DEFINE_DESTRUCTOR(LinearTransform)

DEFINE_GETTER(LinearTransform, int, have_bias)

DEFINE_GETTER(LinearTransform, int, is_orthonormal)

/*********************************************
 * RandomRotationMatrix
 *********************************************/

DEFINE_DESTRUCTOR(RandomRotationMatrix)

int faiss_RandomRotationMatrix_new_with(FaissRandomRotationMatrix** p_vt,
        int d_in,
        int d_out) {
    try {
        *p_vt = reinterpret_cast<FaissRandomRotationMatrix*>(
                new faiss::RandomRotationMatrix(d_in, d_out));
    }
    CATCH_AND_HANDLE
}

/*********************************************
 * PCAMatrix
 *********************************************/

DEFINE_DESTRUCTOR(PCAMatrix)

int faiss_PCAMatrix_new_with(FaissPCAMatrix** p_vt,
        int d_in,
        int d_out,
        float eigen_power,
        int random_rotation) {
    try {
        *p_vt = reinterpret_cast<FaissPCAMatrix*>(
                new faiss::PCAMatrix(d_in, d_out, eigen_power, random_rotation));
    }
    CATCH_AND_HANDLE
}

DEFINE_GETTER(PCAMatrix, int, d_in)

DEFINE_GETTER(PCAMatrix, int, d_out)

DEFINE_GETTER(PCAMatrix, float, eigen_power)

DEFINE_GETTER(PCAMatrix, int, random_rotation)

}
