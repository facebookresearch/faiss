/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <mutex>
#include <string>
#include <vector>

#include <faiss/invlists/InvertedLists.h>
#include <faiss/invlists/InvertedListsIOHook.h>
#include <faiss/utils/AlignedTable.h>

namespace faiss {

struct DiskIOBackend;

/** Inverted lists of IndexIVFRaBitQSSD: the RaBitQ code of each vector is
 * split between memory and a file on an SSD.
 *
 * Memory holds each list as batches of 32 vectors. A batch is the sign bits
 * of dimensions [0, mem_dim) of its vectors, packed in the FastScan block
 * layout, followed by kMemFactors arrays of 32 floats (f_add, f_rescale,
 * f_error).
 *
 * The file holds one fixed-size record per vector:
 *
 *     sign bits of dimensions [mem_dim, d)  ((d - mem_dim) / 8 bytes)
 *     extra-bit code                         (d * ex_bits / 8 bytes)
 *     f_add_ex, f_rescale_ex                 (2 floats)
 *     id                                     (int64)
 *
 * The file starts with a kSectorSize-byte header. Records are packed into
 * pages of page_size bytes, the smallest multiple of kSectorSize that holds
 * one record, and never straddle two pages. The pages of a list are stored
 * in a few extents, allocated as the list grows, so the records of a list
 * are in add order and a page is one aligned read.
 *
 * The file is opened with O_DIRECT, so that searches read from the device
 * and not from the page cache. Searches read it with libaio, one context per
 * search thread. The contexts are kept between searches and destroyed with
 * the inverted lists.
 *
 * The codes are written by IndexIVFRaBitQSSD::add, through append(); the
 * generic add_entries / update_entries / get_codes calls are not supported.
 */
struct RaBitQSSDInvertedLists : InvertedLists {
    /// alignment and size unit of all file accesses
    static constexpr size_t kSectorSize = 4096;
    /// vectors per in-memory batch, the FastScan block size
    static constexpr size_t kBatchSize = 32;
    /// floats stored in memory per vector
    static constexpr size_t kMemFactors = 3;

    size_t d = 0;       ///< dimensionality
    size_t mem_dim = 0; ///< dimensions whose sign bits stay in memory
    size_t ex_bits = 0; ///< extra bits per dimension, in the file

    // derived from d, mem_dim and ex_bits by set_derived_values()
    size_t batch_bytes = 0;      ///< bytes per in-memory batch
    size_t record_size = 0;      ///< bytes per record in the file
    size_t page_size = 0;        ///< bytes per page in the file
    size_t records_per_page = 0; ///< records per page

    std::string filename; ///< path of the file
    bool read_only = false;

    /// a run of pages of the file, owned by one list
    struct Extent {
        size_t first_page; ///< in the file
        size_t n_pages;
    };

    std::vector<size_t> sizes; ///< number of vectors in each list

    /// in-memory batches of each list. The capacity of a table may exceed
    /// the bytes in use, batch_bytes per started batch.
    std::vector<AlignedTableTightAlloc<uint8_t>> mem_data;

    /// pages of each list, in list order
    std::vector<std::vector<Extent>> extents;

    size_t total_pages = 0; ///< pages allocated in the file
    uint64_t file_id = 0;   ///< random tag, also written to the file header

    /** Create the inverted lists and their file (truncating it).
     *
     * @param nlist     number of lists
     * @param d         dimensionality, a multiple of 64
     * @param mem_dim   dimensions whose sign bits stay in memory, a
     *                  multiple of 64 in [64, d]
     * @param ex_bits   extra bits per dimension, in [1, 8]
     * @param filename  path of the file
     */
    RaBitQSSDInvertedLists(
            size_t nlist,
            size_t d,
            size_t mem_dim,
            size_t ex_bits,
            const char* filename);

    /// for the I/O functions
    RaBitQSSDInvertedLists();

    RaBitQSSDInvertedLists(const RaBitQSSDInvertedLists&) = delete;
    RaBitQSSDInvertedLists& operator=(const RaBitQSSDInvertedLists&) = delete;

    ~RaBitQSSDInvertedLists() override;

    /// check d, mem_dim and ex_bits, and compute the derived sizes
    void set_derived_values();

    /// number of in-memory batches of a list
    size_t list_batches(size_t list_no) const;

    /// pages of the file used by a list
    size_t list_pages(size_t list_no) const;

    /// byte offset in the file of page page_no of a list
    size_t page_offset(size_t list_no, size_t page_no) const;

    /// descriptor of the file, opened with O_DIRECT
    int file_descriptor() const;

#ifndef SWIG
    /** A libaio context for one search thread, for up to max_events reads
     * in flight. Creating and destroying a context takes tens of
     * milliseconds, so the contexts are kept for the next searches.
     * Thread-safe.
     */
    std::unique_ptr<DiskIOBackend> acquire_io(int max_events) const;

    /// give back a context of acquire_io(), with no read in flight
    void release_io(std::unique_ptr<DiskIOBackend> io) const noexcept;
#endif

    /*************************
     * InvertedLists interface */

    size_t list_size(size_t list_no) const override;

    /// not supported: the codes are split between memory and the file
    const uint8_t* get_codes(size_t list_no) const override;

    /// read the ids of a list from the file. Release with release_ids.
    const idx_t* get_ids(size_t list_no) const override;

    void release_ids(size_t list_no, const idx_t* ids) const override;

    /// read one id from the file
    idx_t get_single_id(size_t list_no, size_t offset) const override;

    /// not supported, use IndexIVFRaBitQSSD::add
    size_t add_entries(
            size_t list_no,
            size_t n_entry,
            const idx_t* ids,
            const uint8_t* code) override;

    /// not supported
    void update_entries(
            size_t list_no,
            size_t offset,
            size_t n_entry,
            const idx_t* ids,
            const uint8_t* code) override;

    /// only new_size == list_size(list_no) is supported
    void resize(size_t list_no, size_t new_size) override;

    /// empty all lists and truncate the file to its header
    void reset() override;

    /*************************
     * split-code interface   */

    /// make room in the file for n_more vectors in a list. append() does it
    /// as needed; calling it first with the total count of a batch of
    /// appends gives the list a single extent.
    void reserve(size_t list_no, size_t n_more);

    /** Append vectors to a list. Not thread-safe for the same list.
     *
     * @param n           number of vectors
     * @param mem_codes   sign bits of dimensions [0, mem_dim), mem_dim / 8
     *                    bytes per vector
     * @param factors     kMemFactors floats per vector
     * @param records     record_size bytes per vector, in the file format
     */
    void append(
            size_t list_no,
            size_t n,
            const uint8_t* mem_codes,
            const float* factors,
            const uint8_t* records);

    /// open the file of deserialized inverted lists and check its header
    void open_file();

   private:
    int fd = -1;
    std::mutex alloc_mutex; ///< protects total_pages

    /// contexts of acquire_io() that are not in use
    mutable std::vector<std::unique_ptr<DiskIOBackend>> io_pool;
    mutable std::mutex io_pool_mutex; ///< protects io_pool

    /// pages from page_no to the end of its extent
    size_t pages_left_in_extent(size_t list_no, size_t page_no) const;

    /// read or write n_pages consecutive pages of a list, which must lie in
    /// one extent, with a blocking call
    void read_pages(
            size_t list_no,
            size_t page_no,
            size_t n_pages,
            uint8_t* buf) const;
    void write_pages(
            size_t list_no,
            size_t page_no,
            size_t n_pages,
            const uint8_t* buf);

    void write_header();
    void check_header() const;
};

struct RaBitQSSDInvertedListsIOHook : InvertedListsIOHook {
    RaBitQSSDInvertedListsIOHook();
    void write(const InvertedLists* ils, IOWriter* f) const override;
    InvertedLists* read(IOReader* f, int io_flags) const override;
};

} // namespace faiss
