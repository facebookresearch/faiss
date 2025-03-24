/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

/***********************************************************
 * Abstract I/O objects
 *
 * I/O is always sequential, seek does not need to be supported
 * (indexes could be read or written to a pipe).
 ***********************************************************/

#pragma once

#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

namespace faiss {

struct IOReader {
    // name that can be used in error messages
    std::string name;

    // fread. Returns number of items read or 0 in case of EOF.
    virtual size_t operator()(void* ptr, size_t size, size_t nitems) = 0;

    // return a file number that can be memory-mapped
    virtual int filedescriptor();

    virtual ~IOReader() {}
};

struct IOWriter {
    // name that can be used in error messages
    std::string name;

    // fwrite. Return number of items written
    virtual size_t operator()(const void* ptr, size_t size, size_t nitems) = 0;

    // return a file number that can be memory-mapped
    virtual int filedescriptor();

    virtual ~IOWriter() noexcept(false) {}
};

struct VectorIOReader : IOReader {
    std::vector<uint8_t> data;
    size_t rp = 0;
    size_t operator()(void* ptr, size_t size, size_t nitems) override;
};

struct VectorIOWriter : IOWriter {
    std::vector<uint8_t> data;
    size_t operator()(const void* ptr, size_t size, size_t nitems) override;
};

struct FileIOReader : IOReader {
    FILE* f = nullptr;
    bool need_close = false;

    FileIOReader(FILE* rf);

    FileIOReader(const char* fname);

    ~FileIOReader() override;

    size_t operator()(void* ptr, size_t size, size_t nitems) override;

    int filedescriptor() override;
};

struct FileIOWriter : IOWriter {
    FILE* f = nullptr;
    bool need_close = false;

    FileIOWriter(FILE* wf);

    FileIOWriter(const char* fname);

    ~FileIOWriter() override;

    size_t operator()(const void* ptr, size_t size, size_t nitems) override;

    int filedescriptor() override;
};

/*******************************************************
 * Buffered reader + writer
 *
 * They attempt to read and write only buffers of size bsz to the
 * underlying reader or writer. This is done by splitting or merging
 * the read/write functions.
 *******************************************************/

/** wraps an ioreader to make buffered reads to avoid too small reads */
struct BufferedIOReader : IOReader {
    IOReader* reader;
    size_t bsz;
    size_t ofs;    ///< offset in input stream
    size_t ofs2;   ///< number of bytes returned to caller
    size_t b0, b1; ///< range of available bytes in the buffer
    std::vector<char> buffer;

    /**
     * @param bsz    buffer size (bytes). Reads will be done by batched of
     *               this size
     */
    explicit BufferedIOReader(IOReader* reader, size_t bsz = 1024 * 1024);

    size_t operator()(void* ptr, size_t size, size_t nitems) override;
};

struct BufferedIOWriter : IOWriter {
    IOWriter* writer;
    size_t bsz;
    size_t ofs;
    size_t ofs2; ///< number of bytes received from caller
    size_t b0;   ///< amount of data in buffer
    std::vector<char> buffer;

    explicit BufferedIOWriter(IOWriter* writer, size_t bsz = 1024 * 1024);

    size_t operator()(const void* ptr, size_t size, size_t nitems) override;

    // flushes
    ~BufferedIOWriter() override;
};

/// cast a 4-character string to a uint32_t that can be written and read easily
uint32_t fourcc(const char sx[4]);
uint32_t fourcc(const std::string& sx);

// decoding of fourcc (int32 -> string)
void fourcc_inv(uint32_t x, char str[5]);
std::string fourcc_inv(uint32_t x);
std::string fourcc_inv_printable(uint32_t x);

} // namespace faiss
