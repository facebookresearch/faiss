/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

// I/O code for indexes

#ifndef FAISS_INDEX_IO_H
#define FAISS_INDEX_IO_H


#include <cstdio>

namespace faiss {


struct Index;
struct IndexBinary;
struct VectorTransform;
struct IndexIVF;
struct ProductQuantizer;
struct IOReader;
struct IOWriter;
struct InvertedLists;

void write_index (const Index *idx, const char *fname);
void write_index (const Index *idx, FILE *f);
void write_index (const Index *idx, IOWriter *writer);

void write_index_binary (const IndexBinary *idx, const char *fname);
void write_index_binary (const IndexBinary *idx, FILE *f);
void write_index_binary (const IndexBinary *idx, IOWriter *writer);

const int IO_FLAG_MMAP = 1; // try to memmap if possible
const int IO_FLAG_READ_ONLY = 2;

Index *read_index (const char *fname, int io_flags = 0);
Index *read_index (FILE * f, int io_flags = 0);
Index *read_index (IOReader *reader, int io_flags = 0);

IndexBinary *read_index_binary (const char *fname, int io_flags = 0);
IndexBinary *read_index_binary (FILE * f, int io_flags = 0);
IndexBinary *read_index_binary (IOReader *reader, int io_flags = 0);

void write_VectorTransform (const VectorTransform *vt, const char *fname);
VectorTransform *read_VectorTransform (const char *fname);

ProductQuantizer * read_ProductQuantizer (const char*fname);
void write_ProductQuantizer (const ProductQuantizer*pq, const char *fname);

void write_InvertedLists (const InvertedLists *ils, IOWriter *f);
InvertedLists *read_InvertedLists (IOReader *reader, int io_flags = 0);

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



} // namespace faiss


#endif
