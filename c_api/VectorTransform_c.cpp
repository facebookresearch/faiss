/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include "VectorTransform_c.h"
#include <faiss/VectorTransform.h>
#include "macros_impl.h"

extern "C" {

DEFINE_DESTRUCTOR(VectorTransform)

DEFINE_GETTER(VectorTransform, int, is_trained)
DEFINE_SETTER_STATIC(VectorTransform, bool, int, is_trained)

DEFINE_GETTER(VectorTransform, int, d_in)

DEFINE_GETTER(VectorTransform, int, d_out)

int faiss_VectorTransform_train(
        FaissVectorTransform* vt,
        idx_t n,
        const float* x) {
    try {
        reinterpret_cast<faiss::VectorTransform*>(vt)->train(n, x);
    }
    CATCH_AND_HANDLE
}

float* faiss_VectorTransform_apply(
        const FaissVectorTransform* vt,
        idx_t n,
        const float* x) {
    return reinterpret_cast<const faiss::VectorTransform*>(vt)->apply(n, x);
}

void faiss_VectorTransform_apply_noalloc(
        const FaissVectorTransform* vt,
        idx_t n,
        const float* x,
        float* xt) {
    return reinterpret_cast<const faiss::VectorTransform*>(vt)->apply_noalloc(
            n, x, xt);
}

void faiss_VectorTransform_reverse_transform(
        const FaissVectorTransform* vt,
        idx_t n,
        const float* xt,
        float* x) {
    return reinterpret_cast<const faiss::VectorTransform*>(vt)
            ->reverse_transform(n, xt, x);
}

/*********************************************
 * LinearTransform
 *********************************************/

DEFINE_DESTRUCTOR(LinearTransform)

int faiss_LinearTransform_new_with(
        FaissLinearTransform** p_vt,
        int d_in,
        int d_out,
        int have_bias) {
    try {
        *p_vt = reinterpret_cast<FaissLinearTransform*>(
                new faiss::LinearTransform(
                        d_in, d_out, static_cast<bool>(have_bias)));
    }
    CATCH_AND_HANDLE
}

DEFINE_GETTER(LinearTransform, int, have_bias)
DEFINE_SETTER_STATIC(LinearTransform, bool, int, have_bias)

DEFINE_GETTER(LinearTransform, int, is_orthonormal)

void faiss_LinearTransform_set_is_orthonormal_flag(
        FaissLinearTransform* vt,
        int is_orthonormal) {
    reinterpret_cast<faiss::LinearTransform*>(vt)->is_orthonormal =
            static_cast<bool>(is_orthonormal);
}

DEFINE_GETTER(LinearTransform, int, verbose)
DEFINE_SETTER_STATIC(LinearTransform, bool, int, verbose)

void faiss_LinearTransform_A(
        const FaissLinearTransform* vt,
        const float** A,
        size_t* size) {
    const std::vector<float>& v =
            reinterpret_cast<const faiss::LinearTransform*>(vt)->A;
    if (A) {
        *A = v.data();
    }
    if (size) {
        *size = v.size();
    }
}

int faiss_LinearTransform_set_A(FaissLinearTransform* vt, const float* A) {
    try {
        auto lt = reinterpret_cast<faiss::LinearTransform*>(vt);
        lt->A.assign(A, A + static_cast<size_t>(lt->d_out) * lt->d_in);
    }
    CATCH_AND_HANDLE
}

void faiss_LinearTransform_b(
        const FaissLinearTransform* vt,
        const float** b,
        size_t* size) {
    const std::vector<float>& v =
            reinterpret_cast<const faiss::LinearTransform*>(vt)->b;
    if (b) {
        *b = v.data();
    }
    if (size) {
        *size = v.size();
    }
}

int faiss_LinearTransform_set_b(FaissLinearTransform* vt, const float* b) {
    try {
        auto lt = reinterpret_cast<faiss::LinearTransform*>(vt);
        lt->b.assign(b, b + lt->d_out);
    }
    CATCH_AND_HANDLE
}

void faiss_LinearTransform_transform_transpose(
        const FaissLinearTransform* vt,
        idx_t n,
        const float* y,
        float* x) {
    return reinterpret_cast<const faiss::LinearTransform*>(vt)
            ->transform_transpose(n, y, x);
}

void faiss_LinearTransform_set_is_orthonormal(FaissLinearTransform* vt) {
    return reinterpret_cast<faiss::LinearTransform*>(vt)->set_is_orthonormal();
}

/*********************************************
 * RandomRotationMatrix
 *********************************************/

DEFINE_DESTRUCTOR(RandomRotationMatrix)

int faiss_RandomRotationMatrix_new_with(
        FaissRandomRotationMatrix** p_vt,
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

int faiss_PCAMatrix_new_with(
        FaissPCAMatrix** p_vt,
        int d_in,
        int d_out,
        float eigen_power,
        int random_rotation) {
    try {
        bool random_rotation_ = static_cast<bool>(random_rotation);
        *p_vt = reinterpret_cast<FaissPCAMatrix*>(new faiss::PCAMatrix(
                d_in, d_out, eigen_power, random_rotation_));
    }
    CATCH_AND_HANDLE
}

DEFINE_GETTER(PCAMatrix, float, eigen_power)

DEFINE_GETTER(PCAMatrix, int, random_rotation)

DEFINE_GETTER(PCAMatrix, int, balanced_bins)

DEFINE_SETTER(PCAMatrix, int, balanced_bins)

/*********************************************
 * ITQMatrix
 *********************************************/

DEFINE_DESTRUCTOR(ITQMatrix)

int faiss_ITQMatrix_new_with(FaissITQMatrix** p_vt, int d) {
    try {
        *p_vt = reinterpret_cast<FaissITQMatrix*>(new faiss::ITQMatrix(d));
    }
    CATCH_AND_HANDLE
}

DEFINE_DESTRUCTOR(ITQTransform)

int faiss_ITQTransform_new_with(
        FaissITQTransform** p_vt,
        int d_in,
        int d_out,
        int do_pca) {
    try {
        bool do_pca_ = static_cast<bool>(do_pca);
        *p_vt = reinterpret_cast<FaissITQTransform*>(
                new faiss::ITQTransform(d_in, d_out, do_pca_));
    }
    CATCH_AND_HANDLE
}

DEFINE_GETTER(ITQTransform, int, do_pca)

/*********************************************
 * OPQMatrix
 *********************************************/

DEFINE_DESTRUCTOR(OPQMatrix)

int faiss_OPQMatrix_new_with(FaissOPQMatrix** p_vt, int d, int M, int d2) {
    try {
        *p_vt = reinterpret_cast<FaissOPQMatrix*>(
                new faiss::OPQMatrix(d, M, d2));
    }
    CATCH_AND_HANDLE
}

DEFINE_GETTER(OPQMatrix, int, verbose)
DEFINE_SETTER(OPQMatrix, int, verbose)

DEFINE_GETTER(OPQMatrix, int, niter)
DEFINE_SETTER(OPQMatrix, int, niter)

DEFINE_GETTER(OPQMatrix, int, niter_pq)
DEFINE_SETTER(OPQMatrix, int, niter_pq)

/*********************************************
 * RemapDimensionsTransform
 *********************************************/

DEFINE_DESTRUCTOR(RemapDimensionsTransform)

int faiss_RemapDimensionsTransform_new_with(
        FaissRemapDimensionsTransform** p_vt,
        int d_in,
        int d_out,
        int uniform) {
    try {
        bool uniform_ = static_cast<bool>(uniform);
        *p_vt = reinterpret_cast<FaissRemapDimensionsTransform*>(
                new faiss::RemapDimensionsTransform(d_in, d_out, uniform_));
    }
    CATCH_AND_HANDLE
}

/*********************************************
 * NormalizationTransform
 *********************************************/

DEFINE_DESTRUCTOR(NormalizationTransform)

int faiss_NormalizationTransform_new_with(
        FaissNormalizationTransform** p_vt,
        int d,
        float norm) {
    try {
        *p_vt = reinterpret_cast<FaissNormalizationTransform*>(
                new faiss::NormalizationTransform(d, norm));
    }
    CATCH_AND_HANDLE
}

DEFINE_GETTER(NormalizationTransform, float, norm)

/*********************************************
 * CenteringTransform
 *********************************************/

DEFINE_DESTRUCTOR(CenteringTransform)

int faiss_CenteringTransform_new_with(FaissCenteringTransform** p_vt, int d) {
    try {
        *p_vt = reinterpret_cast<FaissCenteringTransform*>(
                new faiss::CenteringTransform(d));
    }
    CATCH_AND_HANDLE
}
}
