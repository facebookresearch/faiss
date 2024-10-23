/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

// I/O code for indexes

#pragma once

namespace faiss {

struct Index;
struct IndexIVF;
struct VectorTransform;
struct Quantizer;
struct IndexBinary;

/* cloning functions */
Index* clone_index(const Index*);

/** Cloner class, useful to override classes with other cloning
 * functions. The cloning function above just calls
 * Cloner::clone_Index. */
struct Cloner {
    virtual VectorTransform* clone_VectorTransform(const VectorTransform*);
    virtual Index* clone_Index(const Index*);
    virtual IndexIVF* clone_IndexIVF(const IndexIVF*);
    virtual ~Cloner() {}
};

Quantizer* clone_Quantizer(const Quantizer* quant);

IndexBinary* clone_binary_index(const IndexBinary* index);

} // namespace faiss
