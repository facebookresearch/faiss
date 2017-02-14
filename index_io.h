
/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the CC-by-NC license found in the
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

/**
 * mmap'ing currently works only for IndexIVFPQCompact, the
 * IndexIVFPQCompact destructor will unmap the file.
 */
Index *read_index (FILE * f, bool try_mmap = false);
Index *read_index (const char *fname, bool try_mmap = false);



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
