/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <cstdio>
#include <cstdlib>

#include <memory>
#include <vector>

#include <gtest/gtest.h>

#include <faiss/IndexIVF.h>
#include <faiss/AutoTune.h>
#include <faiss/VectorTransform.h>


/*************************************************************
 * The functions to test, that can be useful in FANN
 *************************************************************/

/* Returns the cluster the embeddings belong to.
 *
 * @param index      Index, which should be an IVF index
 *                   (otherwise there are no clusters)
 * @param embeddings object descriptors for which the centroids should be found,
 *                   size num_objects * d
 * @param cebtroid_ids
 *                   cluster id each object belongs to, size num_objects
 */
void Search_centroid(faiss::Index *index,
                     const float* embeddings, int num_objects,
                     int64_t* centroid_ids)
{
    const float *x = embeddings;
    std::unique_ptr<float[]> del;
    if (auto index_pre = dynamic_cast<faiss::IndexPreTransform*>(index)) {
        x = index_pre->apply_chain(num_objects, x);
        del.reset((float*)x);
        index = index_pre->index;
    }
    faiss::IndexIVF* index_ivf = dynamic_cast<faiss::IndexIVF*>(index);
    assert(index_ivf);
    index_ivf->quantizer->assign(num_objects, x, centroid_ids);
}



/* Returns the cluster the embeddings belong to.
 *
 * @param index      Index, which should be an IVF index
 *                   (otherwise there are no clusters)
 * @param query_centroid_ids
 *                   centroid ids corresponding to the query vectors (size n)
 * @param result_centroid_ids
 *                   centroid ids corresponding to the results (size n * k)
 * other arguments are the same as the standard search function
 */
void search_and_retrun_centroids(faiss::Index *index,
                                 size_t n,
                                 const float* xin,
                                 long k,
                                 float *distances,
                                 int64_t* labels,
                                 int64_t* query_centroid_ids,
                                 int64_t* result_centroid_ids)
{
    const float *x = xin;
    std::unique_ptr<float []> del;
    if (auto index_pre = dynamic_cast<faiss::IndexPreTransform*>(index)) {
        x = index_pre->apply_chain(n, x);
        del.reset((float*)x);
        index = index_pre->index;
    }
    faiss::IndexIVF* index_ivf = dynamic_cast<faiss::IndexIVF*>(index);
    assert(index_ivf);

    size_t nprobe = index_ivf->nprobe;
    std::vector<long> cent_nos (n * nprobe);
    std::vector<float> cent_dis (n * nprobe);
    index_ivf->quantizer->search(
        n, x, nprobe, cent_dis.data(), cent_nos.data());

    if (query_centroid_ids) {
        for (size_t i = 0; i < n; i++)
            query_centroid_ids[i] = cent_nos[i * nprobe];
    }

    index_ivf->search_preassigned (n, x, k,
                                   cent_nos.data(), cent_dis.data(),
                                   distances, labels, true);

    for (size_t i = 0; i < n * k; i++) {
        int64_t label = labels[i];
        if (label < 0) {
            if (result_centroid_ids)
                result_centroid_ids[i] = -1;
        } else {
            long list_no = label >> 32;
            long list_index = label & 0xffffffff;
            if (result_centroid_ids)
                result_centroid_ids[i] = list_no;
            labels[i] = index_ivf->invlists->get_single_id(list_no, list_index);
        }
    }
}

/*************************************************************
 * Test utils
 *************************************************************/

// return an IndexIVF that may be embedded in an IndexPreTransform
faiss::IndexIVF * get_IndexIVF(faiss::Index *index) {
    if (auto index_pre = dynamic_cast<faiss::IndexPreTransform*>(index)) {
        index = index_pre->index;
    }
    faiss::IndexIVF*  index_ivf = dynamic_cast<faiss::IndexIVF*>(index);
    bool t = index_ivf != nullptr;
    assert(index_ivf);
    return index_ivf;
}



// dimension of the vectors to index
int d = 64;

// size of the database we plan to index
size_t nb = 8000;

// nb of queries
size_t nq = 200;

std::vector<float> make_data(size_t n)
{
    std::vector <float> database (n * d);
    for (size_t i = 0; i < n * d; i++) {
        database[i] = drand48();
    }
    return database;
}

std::unique_ptr<faiss::Index> make_index(const char *index_type,
                                         const std::vector<float> & x) {

    auto index = std::unique_ptr<faiss::Index> (
            faiss::index_factory(d, index_type));
    index->train(nb, x.data());
    index->add(nb, x.data());
    return index;
}

/*************************************************************
 * Test functions for a given index type
 *************************************************************/

bool test_Search_centroid(const char *index_key) {
    std::vector<float> xb = make_data(nb); // database vectors
    auto index = make_index(index_key, xb);

    /* First test: find the centroids associated to the database
       vectors and make sure that each vector does indeed appear in
       the inverted list corresponding to its centroid */

    std::vector<int64_t> centroid_ids (nb);
    Search_centroid(index.get(), xb.data(), nb, centroid_ids.data());

    const faiss::IndexIVF * ivf = get_IndexIVF(index.get());

    for(int i = 0; i < nb; i++) {
        bool found = false;
        int list_no = centroid_ids[i];
        int list_size = ivf->invlists->list_size (list_no);
        auto * list = ivf->invlists->get_ids (list_no);

        for(int j = 0; j < list_size; j++) {
            if (list[j] == i) {
                found = true;
                break;
            }
        }
        if(!found) return false;
    }
    return true;
}

int test_search_and_return_centroids(const char *index_key) {
    std::vector<float> xb = make_data(nb); // database vectors
    auto index = make_index(index_key, xb);

    std::vector<int64_t> centroid_ids (nb);
    Search_centroid(index.get(), xb.data(), nb, centroid_ids.data());

    faiss::IndexIVF * ivf = get_IndexIVF(index.get());
    ivf->nprobe = 4;

    std::vector<float> xq = make_data(nq); // database vectors

    int k = 5;

    // compute a reference search result

    std::vector<long> refI (nq * k);
    std::vector<float> refD (nq * k);
    index->search (nq, xq.data(), k, refD.data(), refI.data());

    // compute search result

    std::vector<long> newI (nq * k);
    std::vector<float> newD (nq * k);

    std::vector<int64_t> query_centroid_ids (nq);
    std::vector<int64_t> result_centroid_ids (nq * k);

    search_and_retrun_centroids(index.get(),
                                nq, xq.data(), k,
                                newD.data(), newI.data(),
                                query_centroid_ids.data(),
                                result_centroid_ids.data());

    // first verify that we have the same result as the standard search

    if (newI != refI) {
        return 1;
    }

    // then check if the result ids are indeed in the inverted list
    // they are supposed to be in

    for(int i = 0; i < nq * k; i++) {
        int list_no = result_centroid_ids[i];
        int result_no = newI[i];

        if (result_no < 0) continue;

        bool found = false;

        int list_size = ivf->invlists->list_size (list_no);
        auto * list = ivf->invlists->get_ids (list_no);

        for(int j = 0; j < list_size; j++) {
            if (list[j] == result_no) {
                found = true;
                break;
            }
        }
        if(!found) return 2;
    }
    return 0;
}

/*************************************************************
 * Test entry points
 *************************************************************/

TEST(test_Search_centroid, IVFFlat) {
    bool ok = test_Search_centroid("IVF32,Flat");
    EXPECT_TRUE(ok);
}

TEST(test_Search_centroid, PCAIVFFlat) {
    bool ok = test_Search_centroid("PCA16,IVF32,Flat");
    EXPECT_TRUE(ok);
}

TEST(test_search_and_return_centroids, IVFFlat) {
    int err = test_search_and_return_centroids("IVF32,Flat");
    EXPECT_NE(err, 1);
    EXPECT_NE(err, 2);
}

TEST(test_search_and_return_centroids, PCAIVFFlat) {
    int err = test_search_and_return_centroids("PCA16,IVF32,Flat");
    EXPECT_NE(err, 1);
    EXPECT_NE(err, 2);
}
