/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <stdio.h>
#include <string.h>

#ifdef __linux__

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#elif defined(_WIN32)

#include <Windows.h> // @manual
#include <io.h>      // @manual

#endif

#include <cstring>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/mapped_io.h>

namespace faiss {

#ifdef __linux__

struct MmappedFileMappingOwner::PImpl {
    void* ptr = nullptr;
    size_t ptr_size = 0;

    PImpl(const std::string& filename) {
        auto f = std::unique_ptr<FILE, decltype(&fclose)>(
                fopen(filename.c_str(), "r"), &fclose);
        FAISS_THROW_IF_NOT_FMT(
                f.get(),
                "could not open %s for reading: %s",
                filename.c_str(),
                strerror(errno));

        // get the size
        struct stat s;
        int status = fstat(fileno(f.get()), &s);
        FAISS_THROW_IF_NOT_FMT(
                status >= 0, "fstat() failed: %s", strerror(errno));

        const size_t filesize = s.st_size;

        void* address = mmap(
                nullptr, filesize, PROT_READ, MAP_SHARED, fileno(f.get()), 0);
        FAISS_THROW_IF_NOT_FMT(
                address != nullptr, "could not mmap(): %s", strerror(errno));

        // btw, fd can be closed here

        madvise(address, filesize, MADV_RANDOM);

        // save it
        ptr = address;
        ptr_size = filesize;
    }

    PImpl(FILE* f) {
        // get the size
        struct stat s;
        int status = fstat(fileno(f), &s);
        FAISS_THROW_IF_NOT_FMT(
                status >= 0, "fstat() failed: %s", strerror(errno));

        const size_t filesize = s.st_size;

        void* address =
                mmap(nullptr, filesize, PROT_READ, MAP_SHARED, fileno(f), 0);
        FAISS_THROW_IF_NOT_FMT(
                address != nullptr, "could not mmap(): %s", strerror(errno));

        // btw, fd can be closed here

        madvise(address, filesize, MADV_RANDOM);

        // save it
        ptr = address;
        ptr_size = filesize;
    }

    ~PImpl() {
        // todo: check for an error
        munmap(ptr, ptr_size);
    }
};

#elif defined(_WIN32)

struct MmappedFileMappingOwner::PImpl {
    void* ptr = nullptr;
    size_t ptr_size = 0;
    HANDLE mapping_handle = INVALID_HANDLE_VALUE;

    PImpl(const std::string& filename) {
        HANDLE file_handle = CreateFile(
                filename.c_str(),
                GENERIC_READ,
                FILE_SHARE_READ,
                nullptr,
                OPEN_EXISTING,
                0,
                nullptr);
        if (file_handle == INVALID_HANDLE_VALUE) {
            const auto error = GetLastError();
            FAISS_THROW_FMT(
                    "could not open the file, %s (error %d)",
                    filename.c_str(),
                    error);
        }

        // get the size of the file
        LARGE_INTEGER len_li;
        if (GetFileSizeEx(file_handle, &len_li) == 0) {
            const auto error = GetLastError();

            CloseHandle(file_handle);

            FAISS_THROW_FMT(
                    "could not get the file size, %s (error %d)",
                    filename.c_str(),
                    error);
        }

        // create a mapping
        mapping_handle = CreateFileMapping(
                file_handle, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (mapping_handle == 0) {
            const auto error = GetLastError();

            CloseHandle(file_handle);

            FAISS_THROW_FMT(
                    "could not create a file mapping, %s (error %d)",
                    filename.c_str(),
                    error);
        }
        CloseHandle(file_handle);

        char* data =
                (char*)MapViewOfFile(mapping_handle, FILE_MAP_READ, 0, 0, 0);
        if (data == nullptr) {
            const auto error = GetLastError();

            CloseHandle(mapping_handle);
            mapping_handle = INVALID_HANDLE_VALUE;

            FAISS_THROW_FMT(
                    "could not get map the file, %s (error %d)",
                    filename.c_str(),
                    error);
        }

        ptr = data;
        ptr_size = len_li.QuadPart;
    }

    PImpl(FILE* f) {
        // obtain a HANDLE from a FILE
        const int fd = _fileno(f);
        if (fd == -1) {
            // no good
            FAISS_THROW_FMT("could not get a HANDLE");
        }

        HANDLE file_handle = (HANDLE)_get_osfhandle(fd);
        if (file_handle == INVALID_HANDLE_VALUE) {
            FAISS_THROW_FMT("could not get an OS HANDLE");
        }

        // get the size of the file
        LARGE_INTEGER len_li;
        if (GetFileSizeEx(file_handle, &len_li) == 0) {
            const auto error = GetLastError();
            FAISS_THROW_FMT("could not get the file size (error %d)", error);
        }

        // create a mapping
        mapping_handle = CreateFileMapping(
                file_handle, nullptr, PAGE_READONLY, 0, 0, nullptr);
        if (mapping_handle == 0) {
            const auto error = GetLastError();
            FAISS_THROW_FMT(
                    "could not create a file mapping, (error %d)", error);
        }

        // the handle is provided externally, so this is not our business
        //   to close file_handle.

        char* data =
                (char*)MapViewOfFile(mapping_handle, FILE_MAP_READ, 0, 0, 0);
        if (data == nullptr) {
            const auto error = GetLastError();

            CloseHandle(mapping_handle);
            mapping_handle = INVALID_HANDLE_VALUE;

            FAISS_THROW_FMT("could not get map the file, (error %d)", error);
        }

        ptr = data;
        ptr_size = len_li.QuadPart;
    }

    ~PImpl() {
        if (mapping_handle != INVALID_HANDLE_VALUE) {
            UnmapViewOfFile(ptr);
            CloseHandle(mapping_handle);

            mapping_handle = INVALID_HANDLE_VALUE;
            ptr = nullptr;
        }
    }
};

#else

struct MmappedFileMappingOwner::PImpl {
    void* ptr = nullptr;
    size_t ptr_size = 0;

    PImpl(const std::string& filename) {
        FAISS_THROW_MSG("Not implemented");
    }

    PImpl(FILE* f) {
        FAISS_THROW_MSG("Not implemented");
    }
};

#endif

MmappedFileMappingOwner::MmappedFileMappingOwner(const std::string& filename) {
    p_impl = std::make_unique<MmappedFileMappingOwner::PImpl>(filename);
}

MmappedFileMappingOwner::MmappedFileMappingOwner(FILE* f) {
    p_impl = std::make_unique<MmappedFileMappingOwner::PImpl>(f);
}

MmappedFileMappingOwner::~MmappedFileMappingOwner() = default;

//
void* MmappedFileMappingOwner::data() const {
    return p_impl->ptr;
}

size_t MmappedFileMappingOwner::size() const {
    return p_impl->ptr_size;
}

MappedFileIOReader::MappedFileIOReader(
        const std::shared_ptr<MmappedFileMappingOwner>& owner)
        : mmap_owner(owner) {}

// this operation performs a copy
size_t MappedFileIOReader::operator()(void* ptr, size_t size, size_t nitems) {
    if (size * nitems == 0) {
        return 0;
    }

    char* ptr_c = nullptr;

    const size_t actual_nitems = this->mmap((void**)&ptr_c, size, nitems);
    if (actual_nitems > 0) {
        memcpy(ptr, ptr_c, size * actual_nitems);
    }

    return actual_nitems;
}

// this operation returns a mmapped address, owned by mmap_owner
size_t MappedFileIOReader::mmap(void** ptr, size_t size, size_t nitems) {
    if (size == 0) {
        return nitems;
    }

    size_t actual_size = size * nitems;
    if (pos + size * nitems > mmap_owner->size()) {
        actual_size = mmap_owner->size() - pos;
    }

    size_t actual_nitems = (actual_size + size - 1) / size;
    if (actual_nitems == 0) {
        return 0;
    }

    // get an address
    *ptr = (void*)(reinterpret_cast<const char*>(mmap_owner->data()) + pos);

    // alter pos
    pos += size * actual_nitems;

    return actual_nitems;
}

int MappedFileIOReader::filedescriptor() {
    // todo
    return -1;
}

} // namespace faiss
