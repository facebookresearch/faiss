/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#pragma once

#include <faiss/invlists/InvertedLists.h>

#include <rocksdb/db.h>

namespace faiss_rocksdb {

struct RocksDBInvertedListsIterator : faiss::InvertedListsIterator {
    RocksDBInvertedListsIterator(
            rocksdb::DB* db,
            size_t list_no,
            size_t code_size);
    virtual bool is_available() const override;
    virtual void next() override;
    virtual std::pair<faiss::idx_t, const uint8_t*> get_id_and_codes() override;

   private:
    std::unique_ptr<rocksdb::Iterator> it;
    size_t list_no;
    size_t code_size;
    std::vector<uint8_t> codes; // buffer for returning codes in next()
};

struct RocksDBInvertedLists : faiss::InvertedLists {
    RocksDBInvertedLists(
            const char* db_directory,
            size_t nlist,
            size_t code_size);

    size_t list_size(size_t list_no) const override;
    const uint8_t* get_codes(size_t list_no) const override;
    const faiss::idx_t* get_ids(size_t list_no) const override;

    size_t add_entries(
            size_t list_no,
            size_t n_entry,
            const faiss::idx_t* ids,
            const uint8_t* code) override;

    void update_entries(
            size_t list_no,
            size_t offset,
            size_t n_entry,
            const faiss::idx_t* ids,
            const uint8_t* code) override;

    void resize(size_t list_no, size_t new_size) override;

    faiss::InvertedListsIterator* get_iterator(
            size_t list_no,
            void* inverted_list_context) const override;

   private:
    std::unique_ptr<rocksdb::DB> db_;
};

} // namespace faiss_rocksdb
