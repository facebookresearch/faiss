
/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the CC-by-NC license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

// Copyright 2004-present Facebook. All Rights Reserved.
// -*- c++ -*-

#ifndef BINARYCODE_TRANSFORM_H
#define BINARYCODE_TRANSFORM_H

/** Defines a few objects that apply transformations to a set of
 * vectors Often these are pre-processing steps.
 */
#include <stdint.h>
#include <vector>

#include "VectorTransform.h"



namespace faiss {


/// ITQ or ckmeans optimization implemented as a matlab/octave sctipt called as a
/// sub-process. The ITQ transform should be applied after a PCA that
/// reduces the # dimensions to the nb of bits of the required code
struct ExternalTransform: LinearTransform {
    std::string octave_wd; ///< directory to call octave from
    std::string model_type;
    int random_seed;

    explicit ExternalTransform (int d);

    virtual void train (faiss::Index::idx_t n, const float *x);


    virtual void reverse_transform (idx_t n, const float * xt,
                                    float *x) const;
    virtual ~ExternalTransform () {}
};


// binary code = VectorTransform then threshold on dimensions
struct BinaryCode {
    size_t d;
    int nbits;
    int code_size;
    faiss::PCAMatrix pca; // 1st transform: a PCA (only if nbits < d)

    bool train_thresholds;
    bool train_means;
    faiss::VectorTransform & vt;

    // the random rotation can also increase the # dimensions
    bool is_trained;

    std::vector <float> thresholds; ///< thresholds to compare with
    std::vector <float> mean_0, mean_1; ///< mean value above or below threshold

    BinaryCode (size_t d, int nbits,
                faiss::VectorTransform & vt,
                bool train_thresholds = true,
                bool train_means = true);

    void encode(size_t n, const float *x, uint8_t *codes) const;
    void decode(size_t n, const uint8_t *codes, float *decoded) const;

    void train (long n, const float *x);

};




}




#endif
