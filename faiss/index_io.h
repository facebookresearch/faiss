/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

// I/O code for indexes

#ifndef FAISS_INDEX_IO_H
#define FAISS_INDEX_IO_H


#include <cstdio>
#include <typeinfo>
#include <string>
#include <vector>

/** I/O functions can read/write to a filename, a file handle or to an
 * object that abstracts the medium.
 *
 * The read functions return objects that should be deallocated with
 * delete. All references within these objectes are owned by the
 * object.
 */

namespace faiss {

struct Index;
struct IndexBinary;
struct VectorTransform;
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

// The read_index flags are implemented only for a subset of index types.
const int IO_FLAG_READ_ONLY = 2;
// strip directory component from ondisk filename, and assume it's in
// the same directory as the index file
const int IO_FLAG_ONDISK_SAME_DIR = 4;
// don't load IVF data to RAM, only list sizes
const int IO_FLAG_SKIP_IVF_DATA = 8;
// try to memmap data (useful for OnDiskInvertedLists)
const int IO_FLAG_MMAP = IO_FLAG_SKIP_IVF_DATA | 0x646f0000;


Index *read_index (const char *fname, int io_flags = 0);
Index *read_index (FILE * f, int io_flags = 0);
Index *read_index (IOReader *reader, int io_flags = 0);

IndexBinary *read_index_binary (const char *fname, int io_flags = 0);
IndexBinary *read_index_binary (FILE * f, int io_flags = 0);
IndexBinary *read_index_binary (IOReader *reader, int io_flags = 0);

void write_VectorTransform (const VectorTransform *vt, const char *fname);
VectorTransform *read_VectorTransform (const char *fname);

ProductQuantizer * read_ProductQuantizer (const char*fname);
ProductQuantizer * read_ProductQuantizer (IOReader *reader);

void write_ProductQuantizer (const ProductQuantizer*pq, const char *fname);
void write_ProductQuantizer (const ProductQuantizer*pq, IOWriter *f);

void write_InvertedLists (const InvertedLists *ils, IOWriter *f);
InvertedLists *read_InvertedLists (IOReader *reader, int io_flags = 0);


#ifndef _MSC_VER
/** Callbacks to handle other types of InvertedList objects.
 *
 * The callbacks should be registered with add_callback before calling
 * read_index or read_InvertedLists. The callbacks for
 * OnDiskInvertedLists are registrered by default. The invlist type is
 * identified by:
 *
 * - the key (a fourcc) at read time
 * - the class name (as given by typeid.name) at write time
 */
struct InvertedListsIOHook {
    const std::string key; ///< string version of the fourcc
    const std::string classname; ///< typeid.name

    InvertedListsIOHook(const std::string & key, const std::string & classname);

    /// write the index to the IOWriter (including the fourcc)
    virtual void write(const InvertedLists *ils, IOWriter *f) const = 0;

    /// called when the fourcc matches this class's fourcc
    virtual InvertedLists * read(IOReader *f, int io_flags) const = 0;

    /** read from a ArrayInvertedLists into this invertedlist type.
     * For this to work, the callback has to be enabled and the io_flag has to be set to
     * IO_FLAG_SKIP_IVF_DATA | (16 upper bits of the fourcc)
     */
    virtual InvertedLists * read_ArrayInvertedLists(
            IOReader *f, int io_flags,
            size_t nlist, size_t code_size,
            const std::vector<size_t> &sizes) const = 0;

    virtual ~InvertedListsIOHook() {}

    /**************************** Manage the set of callbacks ******/

    // transfers ownership
    static void add_callback(InvertedListsIOHook *);
    static void print_callbacks();
    static InvertedListsIOHook* lookup(int h);
    static InvertedListsIOHook* lookup_classname(const std::string & classname);

};

#endif // !_MSC_VER


} // namespace faiss


#endif
