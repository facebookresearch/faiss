/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdio>
#include <cstdlib>

#include <gtest/gtest.h>

#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexFlat.h>
#include <faiss/MetaIndexes.h>
#include <faiss/FaissAssert.h>
#include <faiss/VectorTransform.h>
#include <faiss/OnDiskInvertedLists.h>


namespace faiss {

// Main function to test

// Merge index1 into index0. Works on IndexIVF's and IndexIVF's
// embedded in a IndexPreTransform

void merge_into(Index *index0, Index *index1, bool shift_ids) {
    FAISS_THROW_IF_NOT (index0->d == index1->d);
    IndexIVF *ivf0 = dynamic_cast<IndexIVF *>(index0);
    IndexIVF *ivf1 = dynamic_cast<IndexIVF *>(index1);

    if (!ivf0) {
        IndexPreTransform *pt0 = dynamic_cast<IndexPreTransform *>(index0);
        IndexPreTransform *pt1 = dynamic_cast<IndexPreTransform *>(index1);

        // minimal sanity check
        FAISS_THROW_IF_NOT (pt0 && pt1);
        FAISS_THROW_IF_NOT (pt0->chain.size() == pt1->chain.size());
        for (int i = 0; i < pt0->chain.size(); i++) {
            FAISS_THROW_IF_NOT (typeid(pt0->chain[i]) == typeid(pt1->chain[i]));
        }

        ivf0 = dynamic_cast<IndexIVF *>(pt0->index);
        ivf1 = dynamic_cast<IndexIVF *>(pt1->index);
    }

    FAISS_THROW_IF_NOT (ivf0);
    FAISS_THROW_IF_NOT (ivf1);

    ivf0->merge_from (*ivf1, shift_ids ? ivf0->ntotal : 0);

    // useful for IndexPreTransform
    index0->ntotal = ivf0->ntotal;
    index1->ntotal = ivf1->ntotal;
}

};


struct Tempfilename {

    static pthread_mutex_t mutex;

    std::string filename;

    Tempfilename (const char *prefix = nullptr) {
        pthread_mutex_lock (&mutex);
        filename = tempnam (nullptr, prefix);
        pthread_mutex_unlock (&mutex);
    }

    ~Tempfilename () {
        if (access (filename.c_str(), F_OK)) {
            unlink (filename.c_str());
        }
    }

    const char *c_str() {
        return filename.c_str();
    }

};

pthread_mutex_t Tempfilename::mutex = PTHREAD_MUTEX_INITIALIZER;


// parameters to use for the test
int d = 64;
size_t nb = 1000;
size_t nq = 100;
int nindex = 4;
int k = 10;
int nlist = 40;

typedef faiss::Index::idx_t idx_t;

struct CommonData {

    std::vector <float> database;
    std::vector <float> queries;
    std::vector<idx_t> ids;
    faiss::IndexFlatL2 quantizer;

    CommonData(): database (nb * d), queries (nq * d), ids(nb), quantizer (d) {

        for (size_t i = 0; i < nb * d; i++) {
            database[i] = drand48();
        }
        for (size_t i = 0; i < nq * d; i++) {
            queries[i] = drand48();
        }
        for (int i = 0; i < nb; i++) {
            ids[i] = 123 + 456 * i;
        }
        { // just to train the quantizer
            faiss::IndexIVFFlat iflat (&quantizer, d, nlist);
            iflat.train(nb, database.data());
        }
    }
};

CommonData cd;



/// perform a search on shards, then merge and search again and
/// compare results.
int compare_merged (faiss::IndexShards *index_shards, bool shift_ids,
                    bool standard_merge = true)
{

    std::vector<idx_t> refI(k * nq);
    std::vector<float> refD(k * nq);

    index_shards->search(nq, cd.queries.data(), k, refD.data(), refI.data());
    Tempfilename filename;

    std::vector<idx_t> newI(k * nq);
    std::vector<float> newD(k * nq);

    if (standard_merge) {

        for (int i = 1; i < nindex; i++) {
            merge_into(index_shards->at(0), index_shards->at(i), shift_ids);
        }

        index_shards->sync_with_shard_indexes();
    } else {
        std::vector<const faiss::InvertedLists *> lists;
        faiss::IndexIVF *index0 = nullptr;
        size_t ntotal = 0;
        for (int i = 0; i < nindex; i++) {
            auto index_ivf = dynamic_cast<faiss::IndexIVF*>(index_shards->at(i));
            assert (index_ivf);
            if (i == 0) {
                index0 = index_ivf;
            }
            lists.push_back (index_ivf->invlists);
            ntotal += index_ivf->ntotal;
        }

        auto il = new faiss::OnDiskInvertedLists(
                        index0->nlist, index0->code_size,
                        filename.c_str());

        il->merge_from(lists.data(), lists.size());

        index0->replace_invlists(il, true);
        index0->ntotal = ntotal;
    }
    // search only on first index
    index_shards->at(0)->search(nq, cd.queries.data(),
                                k, newD.data(), newI.data());

    size_t ndiff = 0;
    for (size_t i = 0; i < k * nq; i++) {
        if (refI[i] != newI[i]) {
            ndiff ++;
        }
    }
    return ndiff;
}


// test on IVFFlat with implicit numbering
TEST(MERGE, merge_flat_no_ids) {
    faiss::IndexShards index_shards(d);
    index_shards.own_fields = true;
    for (int i = 0; i < nindex; i++) {
        index_shards.add_shard (
            new faiss::IndexIVFFlat (&cd.quantizer, d, nlist));
    }
    EXPECT_TRUE(index_shards.is_trained);
    index_shards.add(nb, cd.database.data());
    size_t prev_ntotal = index_shards.ntotal;
    int ndiff = compare_merged(&index_shards, true);
    EXPECT_EQ (prev_ntotal, index_shards.ntotal);
    EXPECT_EQ(0, ndiff);
}


// test on IVFFlat, explicit ids
TEST(MERGE, merge_flat) {
    faiss::IndexShards index_shards(d, false, false);
    index_shards.own_fields = true;

    for (int i = 0; i < nindex; i++) {
        index_shards.add_shard (
             new faiss::IndexIVFFlat (&cd.quantizer, d, nlist));
    }

    EXPECT_TRUE(index_shards.is_trained);
    index_shards.add_with_ids(nb, cd.database.data(), cd.ids.data());
    int ndiff = compare_merged(&index_shards, false);
    EXPECT_GE(0, ndiff);
}

// test on IVFFlat and a VectorTransform
TEST(MERGE, merge_flat_vt) {
    faiss::IndexShards index_shards(d, false, false);
    index_shards.own_fields = true;

    // here we have to retrain because of the vectorTransform
    faiss::RandomRotationMatrix rot(d, d);
    rot.init(1234);
    faiss::IndexFlatL2 quantizer (d);

    { // just to train the quantizer
        faiss::IndexIVFFlat iflat (&quantizer, d, nlist);
        faiss::IndexPreTransform ipt (&rot, &iflat);
        ipt.train(nb, cd.database.data());
    }

    for (int i = 0; i < nindex; i++) {
        faiss::IndexPreTransform * ipt = new faiss::IndexPreTransform (
             new faiss::RandomRotationMatrix (rot),
             new faiss::IndexIVFFlat (&quantizer, d, nlist)
        );
        ipt->own_fields = true;
        index_shards.add_shard (ipt);
    }
    EXPECT_TRUE(index_shards.is_trained);
    index_shards.add_with_ids(nb, cd.database.data(), cd.ids.data());
    size_t prev_ntotal = index_shards.ntotal;
    int ndiff = compare_merged(&index_shards, false);
    EXPECT_EQ (prev_ntotal, index_shards.ntotal);
    EXPECT_GE(0, ndiff);
}


// put the merged invfile on disk
TEST(MERGE, merge_flat_ondisk) {
    faiss::IndexShards index_shards(d, false, false);
    index_shards.own_fields = true;
    Tempfilename filename;

    for (int i = 0; i < nindex; i++) {
        auto ivf = new faiss::IndexIVFFlat (&cd.quantizer, d, nlist);
        if (i == 0) {
            auto il = new faiss::OnDiskInvertedLists (
                ivf->nlist, ivf->code_size,
                filename.c_str());
            ivf->replace_invlists(il, true);
        }
        index_shards.add_shard (ivf);
    }

    EXPECT_TRUE(index_shards.is_trained);
    index_shards.add_with_ids(nb, cd.database.data(), cd.ids.data());
    int ndiff = compare_merged(&index_shards, false);

    EXPECT_EQ(ndiff, 0);
}

// non use ondisk specific merge
TEST(MERGE, merge_flat_ondisk_2) {
    faiss::IndexShards index_shards(d, false, false);
    index_shards.own_fields = true;

    for (int i = 0; i < nindex; i++) {
        index_shards.add_shard (
             new faiss::IndexIVFFlat (&cd.quantizer, d, nlist));
    }
    EXPECT_TRUE(index_shards.is_trained);
    index_shards.add_with_ids(nb, cd.database.data(), cd.ids.data());
    int ndiff = compare_merged(&index_shards, false, false);
    EXPECT_GE(0, ndiff);
}
