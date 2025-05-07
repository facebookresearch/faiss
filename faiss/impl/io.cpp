/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <algorithm>
#include <cassert>
#include <cstring>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/io.h>

namespace faiss {

/***********************************************************************
 * IO functions
 ***********************************************************************/

int IOReader::filedescriptor() {
    FAISS_THROW_MSG("IOReader does not support memory mapping");
}

int IOWriter::filedescriptor() {
    FAISS_THROW_MSG("IOWriter does not support memory mapping");
}

/***********************************************************************
 * IO Vector
 ***********************************************************************/

size_t VectorIOWriter::operator()(const void* ptr, size_t size, size_t nitems) {
    size_t bytes = size * nitems;
    if (bytes > 0) {
        size_t o = data.size();
        data.resize(o + bytes);
        memcpy(&data[o], ptr, size * nitems);
    }
    return nitems;
}

size_t VectorIOReader::operator()(void* ptr, size_t size, size_t nitems) {
    if (rp >= data.size())
        return 0;
    size_t nremain = (data.size() - rp) / size;
    if (nremain < nitems)
        nitems = nremain;
    if (size * nitems > 0) {
        memcpy(ptr, &data[rp], size * nitems);
        rp += size * nitems;
    }
    return nitems;
}

/***********************************************************************
 * IO File
 ***********************************************************************/

FileIOReader::FileIOReader(FILE* rf) : f(rf) {}

FileIOReader::FileIOReader(const char* fname) {
    name = fname;
    f = fopen(fname, "rb");
    FAISS_THROW_IF_NOT_FMT(
            f, "could not open %s for reading: %s", fname, strerror(errno));
    need_close = true;
}

FileIOReader::~FileIOReader() {
    if (need_close) {
        int ret = fclose(f);
        if (ret != 0) { // we cannot raise and exception in the destructor
            fprintf(stderr,
                    "file %s close error: %s",
                    name.c_str(),
                    strerror(errno));
        }
    }
}

size_t FileIOReader::operator()(void* ptr, size_t size, size_t nitems) {
    return fread(ptr, size, nitems, f);
}

int FileIOReader::filedescriptor() {
#ifdef _AIX
    return fileno(f);
#else
    return ::fileno(f);
#endif
}

FileIOWriter::FileIOWriter(FILE* wf) : f(wf) {}

FileIOWriter::FileIOWriter(const char* fname) {
    name = fname;
    f = fopen(fname, "wb");
    FAISS_THROW_IF_NOT_FMT(
            f, "could not open %s for writing: %s", fname, strerror(errno));
    need_close = true;
}

FileIOWriter::~FileIOWriter() {
    if (need_close) {
        int ret = fclose(f);
        if (ret != 0) {
            // we cannot raise and exception in the destructor
            fprintf(stderr,
                    "file %s close error: %s",
                    name.c_str(),
                    strerror(errno));
        }
    }
}

size_t FileIOWriter::operator()(const void* ptr, size_t size, size_t nitems) {
    return fwrite(ptr, size, nitems, f);
}

int FileIOWriter::filedescriptor() {
#ifdef _AIX
    return fileno(f);
#else
    return ::fileno(f);
#endif
}

/***********************************************************************
 * IO buffer
 ***********************************************************************/

BufferedIOReader::BufferedIOReader(IOReader* reader, size_t bsz)
        : reader(reader),
          bsz(bsz),
          ofs(0),
          ofs2(0),
          b0(0),
          b1(0),
          buffer(bsz) {}

size_t BufferedIOReader::operator()(void* ptr, size_t unitsize, size_t nitems) {
    size_t size = unitsize * nitems;
    if (size == 0)
        return 0;
    char* dst = (char*)ptr;
    size_t nb;

    { // first copy available bytes
        nb = std::min(b1 - b0, size);
        memcpy(dst, buffer.data() + b0, nb);
        b0 += nb;
        dst += nb;
        size -= nb;
    }

    // while we would like to have more data
    while (size > 0) {
        assert(b0 == b1); // buffer empty on input
        // try to read from main reader
        b0 = 0;
        b1 = (*reader)(buffer.data(), 1, bsz);

        if (b1 == 0) {
            // no more bytes available
            break;
        }
        ofs += b1;

        // copy remaining bytes
        size_t nb2 = std::min(b1, size);
        memcpy(dst, buffer.data(), nb2);
        b0 = nb2;
        nb += nb2;
        dst += nb2;
        size -= nb2;
    }
    ofs2 += nb;
    return nb / unitsize;
}

BufferedIOWriter::BufferedIOWriter(IOWriter* writer, size_t bsz)
        : writer(writer), bsz(bsz), ofs2(0), b0(0), buffer(bsz) {}

size_t BufferedIOWriter::operator()(
        const void* ptr,
        size_t unitsize,
        size_t nitems) {
    size_t size = unitsize * nitems;
    if (size == 0)
        return 0;
    const char* src = (const char*)ptr;
    size_t nb;

    { // copy as many bytes as possible to buffer
        nb = std::min(bsz - b0, size);
        memcpy(buffer.data() + b0, src, nb);
        b0 += nb;
        src += nb;
        size -= nb;
    }
    while (size > 0) {
        assert(b0 == bsz);
        // now we need to flush to add more bytes
        size_t ofs_2 = 0;
        do {
            assert(ofs_2 < 10000000);
            size_t written = (*writer)(buffer.data() + ofs_2, 1, bsz - ofs_2);
            FAISS_THROW_IF_NOT(written > 0);
            ofs_2 += written;
        } while (ofs_2 != bsz);

        // copy src to buffer
        size_t nb1 = std::min(bsz, size);
        memcpy(buffer.data(), src, nb1);
        b0 = nb1;
        nb += nb1;
        src += nb1;
        size -= nb1;
    }
    ofs2 += nb;
    return nb / unitsize;
}

BufferedIOWriter::~BufferedIOWriter() {
    size_t ofs_2 = 0;
    while (ofs_2 != b0) {
        // printf("Destructor write %zd \n", b0 - ofs_2);
        size_t written = (*writer)(buffer.data() + ofs_2, 1, b0 - ofs_2);
        FAISS_THROW_IF_NOT(written > 0);
        ofs_2 += written;
    }
}

uint32_t fourcc(const char sx[4]) {
    FAISS_THROW_IF_NOT(4 == strlen(sx));
    const unsigned char* x = (unsigned char*)sx;
    return x[0] | x[1] << 8 | x[2] << 16 | x[3] << 24;
}

uint32_t fourcc(const std::string& sx) {
    FAISS_THROW_IF_NOT(sx.length() == 4);
    const unsigned char* x = (unsigned char*)sx.c_str();
    return x[0] | x[1] << 8 | x[2] << 16 | x[3] << 24;
}

void fourcc_inv(uint32_t x, char str[5]) {
    *(uint32_t*)str = x;
    str[4] = 0;
}

std::string fourcc_inv(uint32_t x) {
    char str[5];
    fourcc_inv(x, str);
    return std::string(str);
}

std::string fourcc_inv_printable(uint32_t x) {
    char cstr[5];
    fourcc_inv(x, cstr);
    std::string str;
    for (int i = 0; i < 4; i++) {
        uint8_t c = cstr[i];
        if (32 <= c && c < 127) {
            str += c;
        } else {
            char buf[10];
            snprintf(buf, sizeof(buf), "\\x%02x", c);
            str += buf;
        }
    }
    return str;
}

} // namespace faiss
