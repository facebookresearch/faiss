/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/invlists/RaBitQSSDInvertedLists.h>

#include <fcntl.h>
#include <sys/types.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cinttypes>
#include <cstring>
#include <memory>
#include <random>
#include <typeinfo>

#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/fast_scan/fast_scan.h>
#include <faiss/impl/io.h>
#include <faiss/impl/io_macros.h>
#include <faiss/index_io.h>
#include <faiss/invlists/DiskIOBackend.h>

namespace faiss {

namespace {

constexpr char kFileMagic[8] = {'F', 'A', 'I', 'S', 'S', 'D', 'R', 'Q'};
constexpr uint32_t kFileVersion = 1;

/// stored at the start of the file
struct FileHeader {
    char magic[8];
    uint32_t version;
    uint32_t reserved;
    uint64_t d;
    uint64_t mem_dim;
    uint64_t ex_bits;
    uint64_t record_size;
    uint64_t page_size;
    uint64_t file_id;
};

static_assert(sizeof(FileHeader) <= RaBitQSSDInvertedLists::kSectorSize);

/// I/O buffers are aligned for O_DIRECT
constexpr int kBufferAlignment = 4096;
static_assert(kBufferAlignment == RaBitQSSDInvertedLists::kSectorSize);
using IOBuffer = AlignedTableTightAlloc<uint8_t, kBufferAlignment>;

/// largest single read or write issued when processing a whole list
constexpr size_t kMaxIOBytes = size_t(1) << 23;

/// Open a file with O_DIRECT, so that reads bypass the page cache.
int open_direct(const std::string& fname, int flags) {
    return open(fname.c_str(), flags | O_DIRECT, 0644);
}

void pread_all(
        int fd,
        uint8_t* buf,
        size_t len,
        size_t offset,
        const std::string& fname) {
    while (len > 0) {
        ssize_t ret = pread(fd, buf, len, static_cast<off_t>(offset));
        if (ret < 0 && errno == EINTR) {
            continue;
        }
        FAISS_THROW_IF_NOT_FMT(
                ret >= 0,
                "read of %s at offset %zu failed: %s",
                fname.c_str(),
                offset,
                strerror(errno));
        FAISS_THROW_IF_NOT_FMT(
                ret > 0,
                "read of %s at offset %zu: unexpected end of file",
                fname.c_str(),
                offset);
        buf += ret;
        len -= ret;
        offset += ret;
    }
}

void pwrite_all(
        int fd,
        const uint8_t* buf,
        size_t len,
        size_t offset,
        const std::string& fname) {
    while (len > 0) {
        ssize_t ret = pwrite(fd, buf, len, static_cast<off_t>(offset));
        if (ret < 0 && errno == EINTR) {
            continue;
        }
        FAISS_THROW_IF_NOT_FMT(
                ret > 0,
                "write of %s at offset %zu failed: %s",
                fname.c_str(),
                offset,
                strerror(errno));
        buf += ret;
        len -= ret;
        offset += ret;
    }
}

uint64_t random_file_id() {
    std::random_device rd;
    uint64_t id = (uint64_t(rd()) << 32) ^ rd();
    return id ^
            uint64_t(
                    std::chrono::steady_clock::now()
                            .time_since_epoch()
                            .count());
}

} // namespace

/*********************************************************
 * RaBitQSSDInvertedLists
 *********************************************************/

RaBitQSSDInvertedLists::RaBitQSSDInvertedLists(
        size_t nlist_in,
        size_t d_in,
        size_t mem_dim_in,
        size_t ex_bits_in,
        const char* filename_in)
        : InvertedLists(nlist_in, INVALID_CODE_SIZE),
          d(d_in),
          mem_dim(mem_dim_in),
          ex_bits(ex_bits_in) {
    FAISS_THROW_IF_NOT_MSG(
            filename_in && filename_in[0],
            "RaBitQSSDInvertedLists needs a file name");
    filename = filename_in;
    set_derived_values();
    sizes.resize(nlist, 0);
    mem_data.resize(nlist);
    extents.resize(nlist);
    file_id = random_file_id();

    fd = open_direct(filename, O_RDWR | O_CREAT | O_TRUNC);
    FAISS_THROW_IF_NOT_FMT(
            fd >= 0,
            "could not create %s with O_DIRECT: %s",
            filename.c_str(),
            strerror(errno));
    try {
        write_header();
    } catch (...) {
        close(fd);
        fd = -1;
        throw;
    }
}

RaBitQSSDInvertedLists::RaBitQSSDInvertedLists()
        : InvertedLists(0, INVALID_CODE_SIZE) {}

RaBitQSSDInvertedLists::~RaBitQSSDInvertedLists() {
    // destroying a libaio context waits for an RCU grace period, tens of
    // milliseconds: destroy them in parallel. The contexts of a parent
    // process, before a fork(), are only dropped, without a parallel region:
    // OpenMP can hang in a forked child.
    const auto own_end = std::partition(
            io_pool.begin(),
            io_pool.end(),
            [](const std::unique_ptr<DiskIOBackend>& io) {
                return io->owned_by_this_process();
            });
    const int n_io = int(own_end - io_pool.begin());
    if (n_io > 1) {
#pragma omp parallel for num_threads(n_io)
        for (int i = 0; i < n_io; i++) {
            io_pool[i].reset();
        }
    }
    io_pool.clear();
    if (fd >= 0) {
        close(fd);
    }
}

void RaBitQSSDInvertedLists::set_derived_values() {
    FAISS_THROW_IF_NOT_FMT(
            d > 0 && d % 64 == 0 && d < (size_t(1) << 31),
            "d=%zu must be a positive multiple of 64",
            d);
    FAISS_THROW_IF_NOT_FMT(
            mem_dim >= 64 && mem_dim <= d && mem_dim % 64 == 0,
            "mem_dim=%zu must be a multiple of 64 in [64, d=%zu]",
            mem_dim,
            d);
    FAISS_THROW_IF_NOT_FMT(
            ex_bits >= 1 && ex_bits <= 8,
            "ex_bits=%zu must be in [1, 8]",
            ex_bits);
    batch_bytes =
            kBatchSize * mem_dim / 8 + kBatchSize * kMemFactors * sizeof(float);
    record_size = (d - mem_dim) / 8 + d * ex_bits / 8 + 2 * sizeof(float) +
            sizeof(idx_t);
    page_size = (record_size + kSectorSize - 1) / kSectorSize * kSectorSize;
    records_per_page = page_size / record_size;
}

size_t RaBitQSSDInvertedLists::list_batches(size_t list_no) const {
    return (sizes[list_no] + kBatchSize - 1) / kBatchSize;
}

size_t RaBitQSSDInvertedLists::list_pages(size_t list_no) const {
    return (sizes[list_no] + records_per_page - 1) / records_per_page;
}

int RaBitQSSDInvertedLists::file_descriptor() const {
    FAISS_THROW_IF_NOT_MSG(
            fd >= 0, "RaBitQSSDInvertedLists: the file is not open");
    return fd;
}

std::unique_ptr<DiskIOBackend> RaBitQSSDInvertedLists::acquire_io(
        int max_events) const {
    {
        std::lock_guard<std::mutex> lock(io_pool_mutex);
        for (size_t i = io_pool.size(); i > 0; i--) {
            DiskIOBackend& io = *io_pool[i - 1];
            if (!io.owned_by_this_process()) {
                // kept by a parent process before a fork()
                io_pool.erase(io_pool.begin() + (i - 1));
            } else if (io.max_events() >= max_events) {
                std::unique_ptr<DiskIOBackend> taken =
                        std::move(io_pool[i - 1]);
                io_pool.erase(io_pool.begin() + (i - 1));
                return taken;
            }
        }
    }
    return std::make_unique<DiskIOBackend>(max_events);
}

void RaBitQSSDInvertedLists::release_io(
        std::unique_ptr<DiskIOBackend> io) const noexcept {
    try {
        std::lock_guard<std::mutex> lock(io_pool_mutex);
        io_pool.push_back(std::move(io));
    } catch (...) {
        // the context is destroyed instead of kept
    }
}

size_t RaBitQSSDInvertedLists::list_size(size_t list_no) const {
    return sizes[list_no];
}

const uint8_t* RaBitQSSDInvertedLists::get_codes(size_t) const {
    FAISS_THROW_MSG(
            "RaBitQSSDInvertedLists: the codes are split between memory "
            "and disk and cannot be accessed as flat codes");
}

const idx_t* RaBitQSSDInvertedLists::get_ids(size_t list_no) const {
    const size_t n = sizes[list_no];
    if (n == 0) {
        return nullptr;
    }
    std::unique_ptr<idx_t[]> ids(new idx_t[n]);
    const size_t n_pages = list_pages(list_no);
    const size_t max_pages = std::max<size_t>(1, kMaxIOBytes / page_size);
    const size_t id_offset = record_size - sizeof(idx_t);
    IOBuffer buf(std::min(n_pages, max_pages) * page_size);

    for (size_t page = 0; page < n_pages;) {
        size_t n_read = std::min(
                {n_pages - page,
                 max_pages,
                 pages_left_in_extent(list_no, page)});
        read_pages(list_no, page, n_read, buf.get());
        for (size_t i = page * records_per_page;
             i < std::min(n, (page + n_read) * records_per_page);
             i++) {
            const size_t pos = i - page * records_per_page;
            const uint8_t* rec = buf.get() +
                    pos / records_per_page * page_size +
                    pos % records_per_page * record_size;
            memcpy(&ids[i], rec + id_offset, sizeof(idx_t));
        }
        page += n_read;
    }
    return ids.release();
}

void RaBitQSSDInvertedLists::release_ids(size_t, const idx_t* ids) const {
    delete[] ids;
}

idx_t RaBitQSSDInvertedLists::get_single_id(size_t list_no, size_t offset)
        const {
    FAISS_THROW_IF_NOT(offset < sizes[list_no]);
    IOBuffer buf(page_size);
    read_pages(list_no, offset / records_per_page, 1, buf.get());
    idx_t id;
    memcpy(&id,
           buf.get() + offset % records_per_page * record_size + record_size -
                   sizeof(idx_t),
           sizeof(idx_t));
    return id;
}

size_t RaBitQSSDInvertedLists::add_entries(
        size_t,
        size_t,
        const idx_t*,
        const uint8_t*) {
    FAISS_THROW_MSG(
            "RaBitQSSDInvertedLists: add vectors with "
            "IndexIVFRaBitQSSD::add");
}

void RaBitQSSDInvertedLists::update_entries(
        size_t,
        size_t,
        size_t,
        const idx_t*,
        const uint8_t*) {
    FAISS_THROW_MSG("RaBitQSSDInvertedLists: update_entries not supported");
}

void RaBitQSSDInvertedLists::resize(size_t list_no, size_t new_size) {
    FAISS_THROW_IF_NOT_MSG(
            new_size == sizes[list_no],
            "RaBitQSSDInvertedLists: lists cannot be resized");
}

void RaBitQSSDInvertedLists::reset() {
    FAISS_THROW_IF_NOT_MSG(
            !read_only, "RaBitQSSDInvertedLists: opened read-only");
    std::fill(sizes.begin(), sizes.end(), 0);
    for (auto& table : mem_data) {
        table.resize(0);
    }
    for (auto& list_extents : extents) {
        list_extents.clear();
    }
    total_pages = 0;
    if (fd >= 0) {
        FAISS_THROW_IF_NOT_FMT(
                ftruncate(fd, kSectorSize) == 0,
                "could not truncate %s: %s",
                filename.c_str(),
                strerror(errno));
    }
}

void RaBitQSSDInvertedLists::reserve(size_t list_no, size_t n_more) {
    FAISS_THROW_IF_NOT_MSG(
            !read_only, "RaBitQSSDInvertedLists: opened read-only");
    const size_t needed =
            (sizes[list_no] + n_more + records_per_page - 1) / records_per_page;
    std::vector<Extent>& list_extents = extents[list_no];
    size_t capacity = 0;
    for (const Extent& e : list_extents) {
        capacity += e.n_pages;
    }
    if (needed <= capacity) {
        return;
    }
    // grow by at least half the capacity, so that a list made by many
    // appends has few extents while at most a third of its pages are unused
    const size_t n_pages = std::max(needed - capacity, capacity / 2);
    size_t first_page;
    {
        std::lock_guard<std::mutex> lock(alloc_mutex);
        first_page = total_pages;
        total_pages += n_pages;
    }
    if (!list_extents.empty() &&
        list_extents.back().first_page + list_extents.back().n_pages ==
                first_page) {
        list_extents.back().n_pages += n_pages;
    } else {
        list_extents.push_back(Extent{first_page, n_pages});
    }
}

void RaBitQSSDInvertedLists::append(
        size_t list_no,
        size_t n,
        const uint8_t* mem_codes_in,
        const float* factors,
        const uint8_t* records) {
    if (n == 0) {
        return;
    }
    FAISS_THROW_IF_NOT_MSG(
            fd >= 0, "RaBitQSSDInvertedLists: the file is not open");
    reserve(list_no, n);
    const size_t n0 = sizes[list_no];
    const size_t n1 = n0 + n;

    // write the records to the pages [page0, page1)
    const size_t page0 = n0 / records_per_page;
    const size_t page1 = (n1 + records_per_page - 1) / records_per_page;
    const size_t max_pages = std::max<size_t>(1, kMaxIOBytes / page_size);
    IOBuffer buf(std::min(page1 - page0, max_pages) * page_size);
    size_t i = 0; // next record to write
    for (size_t page = page0; page < page1;) {
        const size_t n_write = std::min(
                {page1 - page, max_pages, pages_left_in_extent(list_no, page)});
        memset(buf.get(), 0, n_write * page_size);
        // the first page may already hold records of this list
        size_t slot = n0 + i - page * records_per_page;
        if (slot > 0) {
            read_pages(list_no, page, 1, buf.get());
        }
        for (size_t j = 0; j < n_write; j++) {
            uint8_t* page_buf = buf.get() + j * page_size;
            for (; slot < records_per_page && i < n; slot++, i++) {
                memcpy(page_buf + slot * record_size,
                       records + i * record_size,
                       record_size);
            }
            slot = 0;
        }
        write_pages(list_no, page, n_write, buf.get());
        page += n_write;
    }

    // in-memory batches: the sign bits in the FastScan layout, then the
    // factors. pq4_pack_codes_range only ORs the new codes in, so the bytes
    // after the used ones are kept at 0.
    const size_t n_batches = (n1 + kBatchSize - 1) / kBatchSize;
    AlignedTableTightAlloc<uint8_t>& table = mem_data[list_no];
    if (table.size() < n_batches * batch_bytes) {
        const size_t old_size = table.size();
        table.resize(std::max(n_batches * batch_bytes, 2 * old_size));
        memset(table.get() + old_size, 0, table.size() - old_size);
    }
    const size_t nsq = mem_dim / 4;
    pq4_pack_codes_range(
            mem_codes_in,
            nsq,
            n0,
            n1,
            kBatchSize,
            nsq,
            table.get(),
            mem_dim / 8,
            batch_bytes);
    const size_t code_bytes = kBatchSize * mem_dim / 8;
    for (size_t j = 0; j < n; j++) {
        const size_t pos = n0 + j;
        uint8_t* batch = table.get() + pos / kBatchSize * batch_bytes;
        for (size_t k = 0; k < kMemFactors; k++) {
            memcpy(batch + code_bytes +
                           (k * kBatchSize + pos % kBatchSize) * sizeof(float),
                   factors + j * kMemFactors + k,
                   sizeof(float));
        }
    }

    sizes[list_no] = n1;
}

size_t RaBitQSSDInvertedLists::page_offset(size_t list_no, size_t page_no)
        const {
    for (const Extent& e : extents[list_no]) {
        if (page_no < e.n_pages) {
            return kSectorSize + (e.first_page + page_no) * page_size;
        }
        page_no -= e.n_pages;
    }
    FAISS_THROW_FMT(
            "RaBitQSSDInvertedLists: page beyond the end of list %zu", list_no);
}

size_t RaBitQSSDInvertedLists::pages_left_in_extent(
        size_t list_no,
        size_t page_no) const {
    for (const Extent& e : extents[list_no]) {
        if (page_no < e.n_pages) {
            return e.n_pages - page_no;
        }
        page_no -= e.n_pages;
    }
    FAISS_THROW_FMT(
            "RaBitQSSDInvertedLists: page beyond the end of list %zu", list_no);
}

void RaBitQSSDInvertedLists::read_pages(
        size_t list_no,
        size_t page_no,
        size_t n_pages,
        uint8_t* buf) const {
    FAISS_THROW_IF_NOT_MSG(
            fd >= 0, "RaBitQSSDInvertedLists: the file is not open");
    FAISS_ASSERT(n_pages <= pages_left_in_extent(list_no, page_no));
    pread_all(
            fd,
            buf,
            n_pages * page_size,
            page_offset(list_no, page_no),
            filename);
}

void RaBitQSSDInvertedLists::write_pages(
        size_t list_no,
        size_t page_no,
        size_t n_pages,
        const uint8_t* buf) {
    FAISS_ASSERT(n_pages <= pages_left_in_extent(list_no, page_no));
    pwrite_all(
            fd,
            buf,
            n_pages * page_size,
            page_offset(list_no, page_no),
            filename);
}

void RaBitQSSDInvertedLists::write_header() {
    FileHeader h;
    memcpy(h.magic, kFileMagic, sizeof(kFileMagic));
    h.version = kFileVersion;
    h.reserved = 0;
    h.d = d;
    h.mem_dim = mem_dim;
    h.ex_bits = ex_bits;
    h.record_size = record_size;
    h.page_size = page_size;
    h.file_id = file_id;
    IOBuffer buf(kSectorSize);
    memset(buf.get(), 0, kSectorSize);
    memcpy(buf.get(), &h, sizeof(h));
    pwrite_all(fd, buf.get(), kSectorSize, 0, filename);
}

void RaBitQSSDInvertedLists::check_header() const {
    IOBuffer buf(kSectorSize);
    pread_all(fd, buf.get(), kSectorSize, 0, filename);
    FileHeader h;
    memcpy(&h, buf.get(), sizeof(h));
    FAISS_THROW_IF_NOT_FMT(
            memcmp(h.magic, kFileMagic, sizeof(kFileMagic)) == 0,
            "%s is not a RaBitQSSDInvertedLists file",
            filename.c_str());
    FAISS_THROW_IF_NOT_FMT(
            h.version == kFileVersion,
            "%s: unsupported file version %u",
            filename.c_str(),
            h.version);
    FAISS_THROW_IF_NOT_FMT(
            h.d == d && h.mem_dim == mem_dim && h.ex_bits == ex_bits &&
                    h.record_size == record_size && h.page_size == page_size,
            "%s does not match the inverted lists: file has d=%" PRIu64
            " mem_dim=%" PRIu64 " ex_bits=%" PRIu64
            ", lists have d=%zu mem_dim=%zu ex_bits=%zu",
            filename.c_str(),
            h.d,
            h.mem_dim,
            h.ex_bits,
            d,
            mem_dim,
            ex_bits);
    FAISS_THROW_IF_NOT_FMT(
            h.file_id == file_id,
            "%s was written for another index (file id %016" PRIx64
            ", expected %016" PRIx64 ")",
            filename.c_str(),
            h.file_id,
            file_id);
}

void RaBitQSSDInvertedLists::open_file() {
    FAISS_THROW_IF_NOT_MSG(
            fd < 0, "RaBitQSSDInvertedLists: the file is already open");
    fd = open_direct(filename, read_only ? O_RDONLY : O_RDWR);
    FAISS_THROW_IF_NOT_FMT(
            fd >= 0,
            "could not open %s with O_DIRECT: %s",
            filename.c_str(),
            strerror(errno));
    check_header();
}

/*********************************************************
 * I/O hook
 *********************************************************/

RaBitQSSDInvertedListsIOHook::RaBitQSSDInvertedListsIOHook()
        : InvertedListsIOHook("ildr", typeid(RaBitQSSDInvertedLists).name()) {}

void RaBitQSSDInvertedListsIOHook::write(
        const InvertedLists* ils_in,
        IOWriter* f) const {
    const auto* ils = dynamic_cast<const RaBitQSSDInvertedLists*>(ils_in);
    FAISS_THROW_IF_NOT(ils);
    uint32_t h = fourcc("ildr");
    WRITE1(h);
    WRITE1(ils->nlist);
    WRITE1(ils->d);
    WRITE1(ils->mem_dim);
    WRITE1(ils->ex_bits);
    WRITE1(ils->file_id);
    WRITE1(ils->total_pages);
    {
        std::vector<char> x(ils->filename.begin(), ils->filename.end());
        WRITEVECTOR(x);
    }
    WRITEVECTOR(ils->sizes);
    {
        std::vector<size_t> n_extents, flat_extents;
        for (const auto& list_extents : ils->extents) {
            n_extents.push_back(list_extents.size());
            for (const auto& e : list_extents) {
                flat_extents.push_back(e.first_page);
                flat_extents.push_back(e.n_pages);
            }
        }
        WRITEVECTOR(n_extents);
        WRITEVECTOR(flat_extents);
    }
    for (size_t l = 0; l < ils->nlist; l++) {
        if (ils->sizes[l] > 0) {
            WRITEANDCHECK(
                    ils->mem_data[l].get(),
                    ils->list_batches(l) * ils->batch_bytes);
        }
    }
}

InvertedLists* RaBitQSSDInvertedListsIOHook::read(IOReader* f, int io_flags)
        const {
    auto ils = std::make_unique<RaBitQSSDInvertedLists>();
    ils->read_only = io_flags & IO_FLAG_READ_ONLY;
    READ1(ils->nlist);
    READ1(ils->d);
    READ1(ils->mem_dim);
    READ1(ils->ex_bits);
    ils->set_derived_values();
    READ1(ils->file_id);
    READ1(ils->total_pages);
    {
        std::vector<char> x;
        READVECTOR(x);
        ils->filename.assign(x.begin(), x.end());
    }
    if (io_flags & IO_FLAG_ONDISK_SAME_DIR) {
        // the file is next to the index file, under its original base name
        FileIOReader* reader = dynamic_cast<FileIOReader*>(f);
        FAISS_THROW_IF_NOT_MSG(
                reader,
                "IO_FLAG_ONDISK_SAME_DIR only supported "
                "when reading from file");
        std::string dirname = "./";
        size_t slash = reader->name.find_last_of('/');
        if (slash != std::string::npos) {
            dirname = reader->name.substr(0, slash + 1);
        }
        std::string basename = ils->filename;
        slash = basename.find_last_of('/');
        if (slash != std::string::npos) {
            basename = basename.substr(slash + 1);
        }
        ils->filename = dirname + basename;
    }

    const size_t nlist = ils->nlist;
    const size_t byte_limit = get_deserialization_vector_byte_limit();
    READVECTOR(ils->sizes);
    FAISS_THROW_IF_NOT_FMT(
            ils->sizes.size() == nlist,
            "RaBitQSSDInvertedLists: %zu list sizes for nlist=%zu",
            ils->sizes.size(),
            nlist);
    for (size_t size : ils->sizes) {
        FAISS_THROW_IF_NOT_MSG(
                size < byte_limit, "RaBitQSSDInvertedLists: invalid size");
    }
    {
        std::vector<size_t> n_extents, flat_extents;
        READVECTOR(n_extents);
        READVECTOR(flat_extents);
        FAISS_THROW_IF_NOT(n_extents.size() == nlist);
        ils->extents.resize(nlist);
        size_t k = 0;
        for (size_t l = 0; l < nlist; l++) {
            size_t capacity = 0;
            for (size_t j = 0; j < n_extents[l]; j++) {
                FAISS_THROW_IF_NOT(k + 2 <= flat_extents.size());
                RaBitQSSDInvertedLists::Extent e{
                        flat_extents[k], flat_extents[k + 1]};
                k += 2;
                FAISS_THROW_IF_NOT_MSG(
                        e.first_page <= ils->total_pages &&
                                e.n_pages <= ils->total_pages - e.first_page,
                        "RaBitQSSDInvertedLists: extent out of the file");
                capacity += e.n_pages;
                ils->extents[l].push_back(e);
            }
            FAISS_THROW_IF_NOT_MSG(
                    ils->list_pages(l) <= capacity,
                    "RaBitQSSDInvertedLists: list larger than its extents");
        }
        FAISS_THROW_IF_NOT(k == flat_extents.size());
    }

    ils->mem_data.resize(nlist);
    for (size_t l = 0; l < nlist; l++) {
        const size_t nbytes = mul_no_overflow(
                ils->list_batches(l),
                ils->batch_bytes,
                "RaBitQSSDInvertedLists batches");
        FAISS_THROW_IF_NOT(nbytes < byte_limit);
        ils->mem_data[l].resize(nbytes);
        if (nbytes > 0) {
            READANDCHECK(ils->mem_data[l].get(), nbytes);
        }
    }

    if (!(io_flags & IO_FLAG_SKIP_IVF_DATA)) {
        ils->open_file();
    }
    return ils.release();
}

} // namespace faiss
