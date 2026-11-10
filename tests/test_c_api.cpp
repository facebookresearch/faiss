/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <unordered_set>
#include <vector>

#include <gtest/gtest.h>

#include <c_api/IndexBinaryIVF_c.h>
#include <c_api/IndexIVF_c.h>
#include <faiss/IndexBinaryFlat.h>
#include <faiss/IndexBinaryIVF.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>

namespace {

// Allocating backends (such as HStackInvertedLists) require every get_ids()
// result to be released. Keep outstanding copies owned even if a test fails.
struct AllocatingInvertedLists : faiss::ArrayInvertedLists {
    mutable std::unordered_set<const faiss::idx_t*> outstanding;
    mutable size_t acquisitions = 0;
    mutable size_t releases = 0;

    using ArrayInvertedLists::ArrayInvertedLists;

    const faiss::idx_t* get_ids(size_t list_no) const override {
        auto* ids = new faiss::idx_t[list_size(list_no)];
        std::copy_n(
                ArrayInvertedLists::get_ids(list_no), list_size(list_no), ids);
        outstanding.insert(ids);
        ++acquisitions;
        return ids;
    }

    void release_ids(size_t, const faiss::idx_t* ids) const override {
        EXPECT_EQ(outstanding.erase(ids), 1);
        delete[] ids;
        ++releases;
    }

    ~AllocatingInvertedLists() override {
        for (auto* ids : outstanding) {
            delete[] ids;
        }
    }
};

template <typename Index, typename CIndex>
void check_copy_ids(
        Index& index,
        void (*copy_ids)(const CIndex*, size_t, idx_t*)) {
    AllocatingInvertedLists invlists(index.nlist, index.code_size);
    index.replace_invlists(&invlists, false);
    const std::vector<faiss::idx_t> expected{7, 11, 42};
    std::vector<uint8_t> codes(expected.size() * index.code_size);
    invlists.add_entries(1, expected.size(), expected.data(), codes.data());

    for (int i = 0; i < 3; ++i) {
        std::vector<idx_t> actual(expected.size(), -1);
        copy_ids(reinterpret_cast<const CIndex*>(&index), 1, actual.data());
        EXPECT_EQ(actual, expected);
        EXPECT_EQ(invlists.acquisitions, i + 1);
        EXPECT_EQ(invlists.releases, invlists.acquisitions);
        EXPECT_TRUE(invlists.outstanding.empty());
    }

    // An empty list requires no output buffer, but its borrowed IDs still
    // need to be released if the backend allocates them.
    copy_ids(reinterpret_cast<const CIndex*>(&index), 0, nullptr);
    EXPECT_EQ(invlists.acquisitions, 4);
    EXPECT_EQ(invlists.releases, invlists.acquisitions);
    EXPECT_TRUE(invlists.outstanding.empty());
}

} // namespace

TEST(CApiInvertedLists, FloatIdsAreReleased) {
    faiss::IndexFlatL2 quantizer(8);
    faiss::IndexIVFFlat index(&quantizer, 8, 2);
    check_copy_ids(index, faiss_IndexIVF_invlists_get_ids);
}

TEST(CApiInvertedLists, BinaryIdsAreReleased) {
    faiss::IndexBinaryFlat quantizer(8);
    faiss::IndexBinaryIVF index(&quantizer, 8, 2);
    check_copy_ids(index, faiss_IndexBinaryIVF_invlists_get_ids);
}
