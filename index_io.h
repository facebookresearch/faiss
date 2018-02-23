/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

//  Copyright 2004-present Facebook. All Rights Reserved
// -*- c++ -*-
// I/O code for indexes

#ifndef FAISS_INDEX_IO_H
#define FAISS_INDEX_IO_H

#include <cstdio>

namespace faiss {

struct Index;
struct VectorTransform;
struct IndexIVF;
struct ProductQuantizer;

void write_index (const Index *idx, FILE *f);
void write_index (const Index *idx, const char *fname);


const int IO_FLAG_MMAP = 1;
const int IO_FLAG_READ_ONLY = 2;

Index *read_index (FILE * f, int io_flags = 0);
Index *read_index (const char *fname, int io_flags = 0);



void write_VectorTransform (const VectorTransform *vt, const char *fname);
VectorTransform *read_VectorTransform (const char *fname);

ProductQuantizer * read_ProductQuantizer (const char*fname);
void write_ProductQuantizer (const ProductQuantizer*pq, const char *fname);



/* cloning functions */
Index *clone_index (const Index *);

/** Cloner class, useful to override classes with other cloning
 * functions. The cloning function above just calls
 * Cloner::clone_Index. */
struct Cloner {
    virtual VectorTransform *clone_VectorTransform (const VectorTransform *);
    virtual Index *clone_Index (const Index *);
    virtual IndexIVF *clone_IndexIVF (const IndexIVF *);
    virtual ~Cloner() {}
};

}

#endif
