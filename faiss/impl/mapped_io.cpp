#include <stdio.h>
#include <string.h>

#ifdef __linux__

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#elif defined(_WIN32)

#include <Windows.h>

#endif

#include <cstring>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/mapped_io.h>

namespace faiss {

#ifdef __linux__

struct MmappedFileMappingOwner::PImpl {
    void* ptr = nullptr;
    size_t ptr_size = 0;

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

        // set 'random' access pattern
        // todo: check the error
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
    PImpl(FILE* f) {
        // todo: use CreateFileMapping and MapViewOfFile
        FAISS_THROW_FMT("Not implemented");
    }

    ~PImpl() {
        // todo: use UnmapViewOfFile
        FAISS_THROW_FMT("Not implemented");
    }
};

#else

struct MmappedFileMappingOwner::PImpl {
    PImpl(FILE* f) {
        FAISS_THROW_FMT("Not implemented");
    }

    ~PImpl() {
        FAISS_THROW_FMT("Not implemented");
    }
};

#endif

MmappedFileMappingOwner::MmappedFileMappingOwner(const std::string& filename) {
    auto fd = std::unique_ptr<FILE, decltype(&fclose)>(
            fopen(filename.c_str(), "r"), &fclose);
    FAISS_THROW_IF_NOT_FMT(
            fd.get(),
            "could not open %s for reading: %s",
            filename.c_str(),
            strerror(errno));

    p_impl = std::make_unique<MmappedFileMappingOwner::PImpl>(fd.get());
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