/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <random>
#include <tuple>
#include <vector>

#include <unistd.h>

#include <pthread.h>

#include <gtest/gtest.h>

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/index_io.h>
#include <faiss/invlists/OnDiskInvertedLists.h>
#include <faiss/utils/random.h>

namespace {

struct Tempfilename {
    static pthread_mutex_t mutex;

    std::string filename = "/tmp/faiss_tmp_XXXXXX";

    Tempfilename() {
        pthread_mutex_lock(&mutex);
        int fd = mkstemp(&filename[0]);
        close(fd);
        pthread_mutex_unlock(&mutex);
    }

    ~Tempfilename() {
        if (access(filename.c_str(), F_OK) == 0) {
            unlink(filename.c_str());
        }
    }

    const char* c_str() {
        return filename.c_str();
    }
};

pthread_mutex_t Tempfilename::mutex = PTHREAD_MUTEX_INITIALIZER;

// The lists and the free slots must tile [0, totsize) without overlap, and
// the free slots must be sorted and coalesced.
void check_ondisk_slots(const faiss::OnDiskInvertedLists& od) {
    std::vector<std::tuple<size_t, size_t, bool>> ranges; // offset, end, free
    for (const auto& l : od.lists) {
        if (l.capacity > 0) {
            size_t nbytes = l.capacity * (sizeof(faiss::idx_t) + od.code_size);
            ranges.emplace_back(l.offset, l.offset + nbytes, false);
        }
    }
    size_t prev_slot_offset = 0;
    for (const auto& s : od.slots) {
        EXPECT_GE(s.offset, prev_slot_offset);
        prev_slot_offset = s.offset;
        ranges.emplace_back(s.offset, s.offset + s.capacity, true);
    }
    std::sort(ranges.begin(), ranges.end());
    size_t end = 0;
    bool prev_free = false;
    for (const auto& [offset, range_end, is_free] : ranges) {
        if (offset != end) {
            ADD_FAILURE() << (offset < end ? "overlap" : "unused space")
                          << " at offset " << offset;
            return;
        }
        if (prev_free && is_free) {
            ADD_FAILURE() << "free slots not coalesced at offset " << offset;
            return;
        }
        end = range_end;
        prev_free = is_free;
    }
    EXPECT_EQ(end, od.totsize);
}

void expect_same_lists(
        const faiss::InvertedLists& a,
        const faiss::InvertedLists& b) {
    ASSERT_EQ(a.nlist, b.nlist);
    ASSERT_EQ(a.code_size, b.code_size);
    for (size_t l = 0; l < a.nlist; l++) {
        size_t n = b.list_size(l);
        ASSERT_EQ(a.list_size(l), n) << "list " << l;
        if (n == 0) {
            continue;
        }
        std::vector<faiss::idx_t> ids_a(a.get_ids(l), a.get_ids(l) + n);
        std::vector<faiss::idx_t> ids_b(b.get_ids(l), b.get_ids(l) + n);
        EXPECT_EQ(ids_a, ids_b) << "list " << l;
        size_t nbytes = n * a.code_size;
        std::vector<uint8_t> codes_a(a.get_codes(l), a.get_codes(l) + nbytes);
        std::vector<uint8_t> codes_b(b.get_codes(l), b.get_codes(l) + nbytes);
        EXPECT_EQ(codes_a, codes_b) << "list " << l;
    }
}

const faiss::OnDiskInvertedLists& ondisk_invlists(
        const faiss::IndexIVF& index) {
    return *dynamic_cast<const faiss::OnDiskInvertedLists*>(index.invlists);
}

} // namespace

TEST(ONDISK, test_add) {
    int d = 8;
    int nlist = 30, nq = 200, nb = 1500, k = 10;
    faiss::IndexFlatL2 quantizer(d);
    {
        std::vector<float> x(d * nlist);
        faiss::float_rand(x.data(), d * nlist, 12345);
        quantizer.add(nlist, x.data());
    }
    std::vector<float> xb(d * nb);
    faiss::float_rand(xb.data(), d * nb, 23456);

    faiss::IndexIVFFlat index(&quantizer, d, nlist);
    index.add(nb, xb.data());

    std::vector<float> xq(d * nq);
    faiss::float_rand(xq.data(), d * nq, 34567);

    std::vector<float> ref_D(nq * k);
    std::vector<faiss::idx_t> ref_I(nq * k);

    index.search(nq, xq.data(), k, ref_D.data(), ref_I.data());

    Tempfilename filename, filename2;

    {
        faiss::IndexIVFFlat index2(&quantizer, d, nlist);

        faiss::OnDiskInvertedLists ivf(
                index.nlist, index.code_size, filename.c_str());

        index2.replace_invlists(&ivf);

        index2.add(nb, xb.data());

        std::vector<float> new_D(nq * k);
        std::vector<faiss::idx_t> new_I(nq * k);

        index2.search(nq, xq.data(), k, new_D.data(), new_I.data());

        EXPECT_EQ(ref_D, new_D);
        EXPECT_EQ(ref_I, new_I);

        write_index(&index2, filename2.c_str());
    }

    {
        faiss::Index* index3 = faiss::read_index(filename2.c_str());

        std::vector<float> new_D(nq * k);
        std::vector<faiss::idx_t> new_I(nq * k);

        index3->search(nq, xq.data(), k, new_D.data(), new_I.data());

        EXPECT_EQ(ref_D, new_D);
        EXPECT_EQ(ref_I, new_I);

        delete index3;
    }
}

// WARN this test will run multithreaded only in opt mode
TEST(ONDISK, make_invlists_threaded) {
    int nlist = 100;
    int code_size = 32;
    int nadd = 1000000;

    Tempfilename filename;

    faiss::OnDiskInvertedLists ivf(nlist, code_size, filename.c_str());

    std::vector<int> list_nos(nadd);

    std::mt19937 rng;
    std::uniform_real_distribution<> distrib;
    for (int i = 0; i < nadd; i++) {
        double d = distrib(rng);
        list_nos[i] = int(nlist * d * d); // skewed distribution
    }

#pragma omp parallel
    {
        std::vector<uint8_t> code(32);
#pragma omp for
        for (int i = 0; i < nadd; i++) {
            int list_no = list_nos[i];
            int* ar = (int*)code.data();
            ar[0] = i;
            ar[1] = list_no;
            ivf.add_entry(list_no, i, code.data());
        }
    }

    int ntot = 0;
    for (int i = 0; i < nlist; i++) {
        int size = ivf.list_size(i);
        const faiss::idx_t* ids = ivf.get_ids(i);
        const uint8_t* codes = ivf.get_codes(i);
        for (int j = 0; j < size; j++) {
            faiss::idx_t id = ids[j];
            const int* ar = (const int*)&codes[code_size * j];
            EXPECT_EQ(ar[0], id);
            EXPECT_EQ(ar[1], i);
            EXPECT_EQ(list_nos[id], i);
            ntot++;
        }
    }
    EXPECT_EQ(ntot, nadd);
}

namespace {

struct MergedOnDiskIndex {
    int d = 8, nlist = 10;
    faiss::IndexFlatL2 quantizer;
    faiss::IndexIVFFlat ref; // in-memory reference
    std::unique_ptr<faiss::IndexIVFFlat> index;
    Tempfilename ivfdata;
    faiss::idx_t next_id = 0;

    // build two in-memory blocks and merge them into an on-disk index, as
    // contrib/ondisk.py merge_ondisk does
    MergedOnDiskIndex() : quantizer(d), ref(&quantizer, d, nlist) {
        std::vector<float> x(d * nlist);
        faiss::float_rand(x.data(), x.size(), 123);
        quantizer.add(nlist, x.data());
        ref.is_trained = true;

        faiss::IndexIVFFlat block1(&quantizer, d, nlist);
        faiss::IndexIVFFlat block2(&quantizer, d, nlist);
        add(block1, 150);
        add(block2, 100);

        index.reset(new faiss::IndexIVFFlat(&quantizer, d, nlist));
        auto od = new faiss::OnDiskInvertedLists(
                nlist, index->code_size, ivfdata.c_str());
        const faiss::InvertedLists* ils[2] = {block1.invlists, block2.invlists};
        index->ntotal = od->merge_from_multiple(ils, 2);
        index->replace_invlists(od, true);
    }

    // add the same n vectors to dst and to the reference index
    void add(faiss::IndexIVF& dst, int n) {
        std::vector<float> x(d * n);
        faiss::float_rand(x.data(), x.size(), 1000 + next_id);
        std::vector<faiss::idx_t> ids(n);
        for (int i = 0; i < n; i++) {
            ids[i] = next_id++;
        }
        dst.add_with_ids(n, x.data(), ids.data());
        ref.add_with_ids(n, x.data(), ids.data());
    }

    void check() {
        check_ondisk_slots(ondisk_invlists(*index));
        expect_same_lists(*index->invlists, *ref.invlists);
    }
};

} // namespace

TEST(ONDISK, merge_then_add) {
    MergedOnDiskIndex mi;
    mi.check();
    // a merged index is compact
    EXPECT_TRUE(ondisk_invlists(*mi.index).slots.empty());
    for (int n : {1, 7, 40, 300}) {
        mi.add(*mi.index, n);
        mi.check();
    }

    int nq = 20, k = 5;
    std::vector<float> xq(mi.d * nq);
    faiss::float_rand(xq.data(), xq.size(), 789);
    std::vector<float> D(nq * k), ref_D(nq * k);
    std::vector<faiss::idx_t> I(nq * k), ref_I(nq * k);
    mi.index->nprobe = mi.ref.nprobe = 3;
    mi.index->search(nq, xq.data(), k, D.data(), I.data());
    mi.ref.search(nq, xq.data(), k, ref_D.data(), ref_I.data());
    EXPECT_EQ(I, ref_I);
    EXPECT_EQ(D, ref_D);
}

TEST(ONDISK, merge_write_read_then_add) {
    MergedOnDiskIndex mi;
    Tempfilename index_file;
    for (int round = 0; round < 3; round++) {
        faiss::write_index(mi.index.get(), index_file.c_str());
        mi.index.reset(
                dynamic_cast<faiss::IndexIVFFlat*>(
                        faiss::read_index(index_file.c_str())));
        mi.check();
        mi.add(*mi.index, 30);
        mi.check();
    }
}

// Files written by merge_from_multiple before it was fixed contain a free
// slot that covers all the merged lists. They must be repaired when read.
TEST(ONDISK, read_merged_with_stale_free_slot) {
    MergedOnDiskIndex mi;
    auto od = dynamic_cast<faiss::OnDiskInvertedLists*>(mi.index->invlists);
    od->slots.push_back(faiss::OnDiskInvertedLists::Slot(0, od->totsize));

    Tempfilename index_file;
    faiss::write_index(mi.index.get(), index_file.c_str());
    mi.index.reset(
            dynamic_cast<faiss::IndexIVFFlat*>(
                    faiss::read_index(index_file.c_str())));
    mi.check();
    for (int n : {1, 20, 100}) {
        mi.add(*mi.index, n);
        mi.check();
    }
}

// Shrinking a list reallocates it, possibly at its own offset. The ids must
// still be moved.
TEST(ONDISK, remove_ids) {
    int d = 2, nlist = 1;
    faiss::IndexFlatL2 quantizer(d);
    {
        std::vector<float> x(d * nlist);
        faiss::float_rand(x.data(), x.size(), 123);
        quantizer.add(nlist, x.data());
    }
    faiss::IndexIVFFlat ref(&quantizer, d, nlist);
    faiss::IndexIVFFlat index(&quantizer, d, nlist);
    Tempfilename ivfdata;
    index.replace_invlists(
            new faiss::OnDiskInvertedLists(
                    nlist, index.code_size, ivfdata.c_str()),
            true);

    int nb = 64;
    std::vector<float> xb(d * nb);
    faiss::float_rand(xb.data(), xb.size(), 456);
    for (int round = 0; round < 3; round++) {
        index.add(nb, xb.data());
        ref.add(nb, xb.data());
        // shrink the list to 3 entries, below 1/8 of its capacity
        faiss::IDSelectorRange sel(3, ref.ntotal);
        EXPECT_EQ(index.remove_ids(sel), ref.remove_ids(sel));
        check_ondisk_slots(ondisk_invlists(index));
        expect_same_lists(*index.invlists, *ref.invlists);
    }
}
