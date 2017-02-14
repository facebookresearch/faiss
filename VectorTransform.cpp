
/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the CC-by-NC license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved
// -*- c++ -*-

#include "VectorTransform.h"

#include <cstdio>
#include <cmath>
#include <cstring>

#include "utils.h"
#include "FaissAssert.h"
#include "IndexPQ.h"

using namespace faiss;


extern "C" {

// this is to keep the clang syntax checker happy
#ifndef FINTEGER
#define FINTEGER int
#endif


/* declare BLAS functions, see http://www.netlib.org/clapack/cblas/ */

int sgemm_ (
        const char *transa, const char *transb, FINTEGER *m, FINTEGER *
        n, FINTEGER *k, const float *alpha, const float *a,
        FINTEGER *lda, const float *b,
        FINTEGER *ldb, float *beta,
        float *c, FINTEGER *ldc);

int ssyrk_ (
        const char *uplo, const char *trans, FINTEGER *n, FINTEGER *k,
        float *alpha, float *a, FINTEGER *lda,
        float *beta, float *c, FINTEGER *ldc);

/* Lapack functions from http://www.netlib.org/clapack/old/single/ */

int ssyev_ (
        const char *jobz, const char *uplo, FINTEGER *n, float *a,
        FINTEGER *lda, float *w, float *work, FINTEGER *lwork,
        FINTEGER *info);

int sgesvd_(
        const char *jobu, const char *jobvt, FINTEGER *m, FINTEGER *n,
        float *a, FINTEGER *lda, float *s, float *u, FINTEGER *ldu, float *vt,
        FINTEGER *ldvt, float *work, FINTEGER *lwork, FINTEGER *info);

}

/*********************************************
 * VectorTransform
 *********************************************/



float * VectorTransform::apply (Index::idx_t n, const float * x) const
{
    float * xt = new float[n * d_out];
    apply_noalloc (n, x, xt);
    return xt;
}


void VectorTransform::train (idx_t, const float *) {
    // does nothing by default
}


void VectorTransform::reverse_transform (
             idx_t , const float *,
             float *) const
{
    FAISS_ASSERT (!"reverse transform not implemented");
}




/*********************************************
 * LinearTransform
 *********************************************/
/// both d_in > d_out and d_out < d_in are supported
LinearTransform::LinearTransform (int d_in, int d_out,
                                  bool have_bias):
    VectorTransform (d_in, d_out), have_bias (have_bias),
    max_points_per_d (1 << 20), verbose (false)
{}

void LinearTransform::apply_noalloc (Index::idx_t n, const float * x,
                               float * xt) const
{
    FAISS_ASSERT(is_trained || !"Transformation not trained yet");

    float c_factor;
    if (have_bias) {
        FAISS_ASSERT (b.size() == d_out || !"Bias not initialized");
        float * xi = xt;
        for (int i = 0; i < n; i++)
            for(int j = 0; j < d_out; j++)
                *xi++ = b[j];
        c_factor = 1.0;
    } else {
        c_factor = 0.0;
    }

    FAISS_ASSERT (A.size() == d_out * d_in ||
            !"Transformation matrix not initialized");

    float one = 1;
    FINTEGER nbiti = d_out, ni = n, di = d_in;
    sgemm_ ("Transposed", "Not transposed",
            &nbiti, &ni, &di,
            &one, A.data(), &di, x, &di, &c_factor, xt, &nbiti);

}


void LinearTransform::transform_transpose (idx_t n, const float * y,
                                           float *x) const
{
    if (have_bias) { // allocate buffer to store bias-corrected data
        float *y_new = new float [n * d_out];
        const float *yr = y;
        float *yw = y_new;
        for (idx_t i = 0; i < n; i++) {
            for (int j = 0; j < d_out; j++) {
                *yw++ = *yr++ - b [j];
            }
        }
        y = y_new;
    }

    {
        FINTEGER dii = d_in, doi = d_out, ni = n;
        float one = 1.0, zero = 0.0;
        sgemm_ ("Not", "Not", &dii, &ni, &doi,
                &one, A.data (), &dii, y, &doi, &zero, x, &dii);
    }

    if (have_bias) delete [] y;
}

const float * LinearTransform::maybe_subsample_train_set (
            Index::idx_t *n, const float *x)
{
    if (*n <= max_points_per_d * d_in) return x;

    size_t n2 = max_points_per_d * d_in;
    if (verbose) {
        printf ("  Input training set too big, sampling "
                "%ld / %ld vectors\n", n2, *n);
    }
    std::vector<int> subset (*n);
    rand_perm (subset.data (), *n, 1234);
    float *x_subset = new float[n2 * d_in];
    for (long i = 0; i < n2; i++)
        memcpy (&x_subset[i * d_in],
                &x[subset[i] * size_t(d_in)],
                sizeof (x[0]) * d_in);
    *n = n2;
    return x_subset;
}


/*********************************************
 * RandomRotationMatrix
 *********************************************/

void RandomRotationMatrix::init (int seed)
{

    if(d_out <= d_in) {
        A.resize (d_out * d_in);
        float *q = A.data();
        float_randn(q, d_out * d_in, seed);
        matrix_qr(d_in, d_out, q);
    } else {
        A.resize (d_out * d_out);
        float *q = A.data();
        float_randn(q, d_out * d_out, seed);
        matrix_qr(d_out, d_out, q);
        // remove columns
        int i, j;
        for (i = 0; i < d_out; i++) {
            for(j = 0; j < d_in; j++) {
                q[i * d_in + j] = q[i * d_out + j];
            }
        }
        A.resize(d_in * d_out);
    }

}

void RandomRotationMatrix::reverse_transform (idx_t n, const float * xt,
                                              float *x) const
{
    transform_transpose (n, xt, x);
}

/*********************************************
 * PCAMatrix
 *********************************************/

PCAMatrix::PCAMatrix (int d_in, int d_out,
                      float eigen_power, bool random_rotation):
    LinearTransform(d_in, d_out, true),
    eigen_power(eigen_power), random_rotation(random_rotation)
{
    is_trained = false;
    max_points_per_d = 1000;
    balanced_bins = 0;
}


void PCAMatrix::train (Index::idx_t n, const float *x)
{
    const float * x_in = x;

    x = maybe_subsample_train_set(&n, x);

    // compute mean
    mean.clear(); mean.resize(d_in, 0.0);
    if (have_bias) { // we may want to skip the bias
        const float *xi = x;
        for (int i = 0; i < n; i++) {
            for(int j = 0; j < d_in; j++)
                mean[j] += *xi++;
        }
        for(int j = 0; j < d_in; j++)
            mean[j] /= n;
    }
    if(verbose) {
        printf("mean=[");
        for(int j = 0; j < d_in; j++) printf("%g ", mean[j]);
        printf("]\n");
    }

    if(n >= d_in) {
        // compute covariance matrix, store it in PCA matrix
        PCAMat.resize(d_in * d_in);
        float * cov = PCAMat.data();
        { // initialize with  mean * mean^T term
            float *ci = cov;
            for(int i = 0; i < d_in; i++) {
                for(int j = 0; j < d_in; j++)
                    *ci++ = - n * mean[i] * mean[j];
            }
        }
        {
            FINTEGER di = d_in, ni = n;
            float one = 1.0;
            ssyrk_ ("Up", "Non transposed",
                    &di, &ni, &one, (float*)x, &di, &one, cov, &di);

        }
        if(verbose && d_in <= 10) {
            float *ci = cov;
            printf("cov=\n");
            for(int i = 0; i < d_in; i++) {
                for(int j = 0; j < d_in; j++)
                    printf("%10g ", *ci++);
                printf("\n");
            }
        }

        { // compute eigenvalues and vectors
            eigenvalues.resize(d_in);
            FINTEGER info = 0, lwork = -1, di = d_in;
            float workq;

            ssyev_ ("Vectors as well", "Upper",
                    &di, cov, &di, eigenvalues.data(), &workq, &lwork, &info);
            lwork = FINTEGER(workq);
            float *work = new float[lwork];

            ssyev_ ("Vectors as well", "Upper",
                    &di, cov, &di, eigenvalues.data(), work, &lwork, &info);

            if (info != 0) {
                fprintf (stderr, "WARN ssyev info returns %d, "
                         "a very bad PCA matrix is learnt\n",
                         int(info));

            }

            delete [] work;

            if(verbose && d_in <= 10) {
                printf("info=%ld new eigvals=[", long(info));
                for(int j = 0; j < d_in; j++) printf("%g ", eigenvalues[j]);
                printf("]\n");

                float *ci = cov;
                printf("eigenvecs=\n");
                for(int i = 0; i < d_in; i++) {
                    for(int j = 0; j < d_in; j++)
                        printf("%10.4g ", *ci++);
                    printf("\n");
                }
            }

        }

        // revert order of eigenvectors & values

        for(int i = 0; i < d_in / 2; i++) {

            std::swap(eigenvalues[i], eigenvalues[d_in - 1 - i]);
            float *v1 = cov + i * d_in;
            float *v2 = cov + (d_in - 1 - i) * d_in;
            for(int j = 0; j < d_in; j++)
                std::swap(v1[j], v2[j]);
        }

    } else {
        FAISS_ASSERT(!"Gramm matrix version not implemented "
               "--  provide more training examples than dimensions");
    }


    if (x != x_in) delete [] x;

    prepare_Ab();
    is_trained = true;
}

void PCAMatrix::copy_from (const PCAMatrix & other)
{
    FAISS_ASSERT (other.is_trained);
    mean = other.mean;
    eigenvalues = other.eigenvalues;
    PCAMat = other.PCAMat;
    prepare_Ab ();
    is_trained = true;
}

void PCAMatrix::prepare_Ab ()
{

    if (!random_rotation) {
        A = PCAMat;
        A.resize(d_out * d_in); // strip off useless dimensions

        // first scale the components
        if (eigen_power != 0) {
            float *ai = A.data();
            for (int i = 0; i < d_out; i++) {
                float factor = pow(eigenvalues[i], eigen_power);
                for(int j = 0; j < d_in; j++)
                    *ai++ *= factor;
            }
        }

        if (balanced_bins != 0) {
            FAISS_ASSERT (d_out % balanced_bins == 0);
            int dsub = d_out / balanced_bins;
            std::vector <float> Ain;
            std::swap(A, Ain);
            A.resize(d_out * d_in);

            std::vector <float> accu(balanced_bins);
            std::vector <int> counter(balanced_bins);

            // greedy assignment
            for (int i = 0; i < d_out; i++) {
                // find best bin
                int best_j = -1;
                float min_w = 1e30;
                for (int j = 0; j < balanced_bins; j++) {
                    if (counter[j] < dsub && accu[j] < min_w) {
                        min_w = accu[j];
                        best_j = j;
                    }
                }
                int row_dst = best_j * dsub + counter[best_j];
                accu[best_j] += eigenvalues[i];
                counter[best_j] ++;
                memcpy (&A[row_dst * d_in], &Ain[i * d_in],
                        d_in * sizeof (A[0]));
            }

            if (verbose) {
                printf("  bin accu=[");
                for (int i = 0; i < balanced_bins; i++)
                    printf("%g ", accu[i]);
                printf("]\n");
            }
        }


    } else {
        FAISS_ASSERT (balanced_bins == 0 ||
                      !"both balancing bins and applying a random rotation "
                      "does not make sense");
        RandomRotationMatrix rr(d_out, d_out);

        rr.init(5);

        // apply scaling on the rotation matrix (right multiplication)
        if (eigen_power != 0) {
            for (int i = 0; i < d_out; i++) {
                float factor = pow(eigenvalues[i], eigen_power);
                for(int j = 0; j < d_out; j++)
                   rr.A[j * d_out + i] *= factor;
            }
        }

        A.resize(d_in * d_out);
        {
            FINTEGER dii = d_in, doo = d_out;
            float one = 1.0, zero = 0.0;

            sgemm_ ("Not", "Not", &dii, &doo, &doo,
                    &one, PCAMat.data(), &dii, rr.A.data(), &doo, &zero,
                    A.data(), &dii);

        }

    }

    b.clear(); b.resize(d_out);

    for (int i = 0; i < d_out; i++) {
        float accu = 0;
        for (int j = 0; j < d_in; j++)
            accu -= mean[j] * A[j + i * d_in];
        b[i] = accu;
    }

}

void PCAMatrix::reverse_transform (idx_t n, const float * xt,
                                   float *x) const
{
    FAISS_ASSERT (eigen_power == 0 ||
                  !"reverse only implemented for orthogonal transforms");
    transform_transpose (n, xt, x);
}

/*********************************************
 * OPQMatrix
 *********************************************/


OPQMatrix::OPQMatrix (int d, int M, int d2):
    LinearTransform (d, d2 == -1 ? d : d2, false), M(M),
    niter (50),
    niter_pq (4), niter_pq_0 (40),
    verbose(false)
{
    is_trained = false;
    max_points_per_d = 1000;
}



void OPQMatrix::train (Index::idx_t n, const float *x)
{

    const float * x_in = x;

    x = maybe_subsample_train_set (&n, x);

    // To support d_out > d_in, we pad input vectors with 0s to d_out
    size_t d = d_out <= d_in ? d_in : d_out;
    size_t d2 = d_out;

#if 0
    // what this test shows: the only way of getting bit-exact
    // reproducible results with sgeqrf and sgesvd seems to be forcing
    // single-threading.
    { // test repro
        std::vector<float> r (d * d);
        float * rotation = r.data();
        float_randn (rotation, d * d, 1234);
        printf("CS0: %016lx\n",
               ivec_checksum (128*128, (int*)rotation));
        matrix_qr (d, d, rotation);
        printf("CS1: %016lx\n",
               ivec_checksum (128*128, (int*)rotation));
        return;
    }
#endif

    if (verbose) {
        printf ("OPQMatrix::train: training an OPQ rotation matrix "
                "for M=%d from %ld vectors in %dD -> %dD\n",
                M, n, d_in, d_out);
    }

    std::vector<float> xtrain (n * d);
    // center x
    {
        std::vector<float> sum (d);
        const float *xi = x;
        for (size_t i = 0; i < n; i++) {
            for (int j = 0; j < d_in; j++)
                sum [j] += *xi++;
        }
        for (int i = 0; i < d; i++) sum[i] /= n;
        float *yi = xtrain.data();
        xi = x;
        for (size_t i = 0; i < n; i++) {
            for (int j = 0; j < d_in; j++)
                *yi++ = *xi++ - sum[j];
            yi += d - d_in;
        }
    }
    float *rotation;

    if (A.size () == 0) {
        A.resize (d * d);
        rotation = A.data();
        if (verbose)
            printf("  OPQMatrix::train: making random %ld*%ld rotation\n",
                   d, d);
        float_randn (rotation, d * d, 1234);
        matrix_qr (d, d, rotation);
        // we use only the d * d2 upper part of the matrix
        A.resize (d * d2);
    } else {
        FAISS_ASSERT (A.size() == d * d2);
        rotation = A.data();
    }


    std::vector<float>
        xproj (d2 * n), pq_recons (d2 * n), xxr (d * n),
        tmp(d * d * 4);

    std::vector<uint8_t> codes (M * n);
    ProductQuantizer pq_regular (d2, M, 8);
    double t0 = getmillisecs();
    for (int iter = 0; iter < niter; iter++) {

        { // torch.mm(xtrain, rotation:t())
            FINTEGER di = d, d2i = d2, ni = n;
            float zero = 0, one = 1;
            sgemm_ ("Transposed", "Not transposed",
                    &d2i, &ni, &di,
                    &one, rotation, &di,
                    xtrain.data(), &di,
                    &zero, xproj.data(), &d2i);
        }

        pq_regular.cp.max_points_per_centroid = 1000;
        pq_regular.cp.niter = iter == 0 ? niter_pq_0 : niter_pq;
        pq_regular.cp.verbose = verbose;
        pq_regular.train (n, xproj.data());

        pq_regular.compute_codes (xproj.data(), codes.data(), n);
        pq_regular.decode (codes.data(), pq_recons.data(), n);

        float pq_err = fvec_L2sqr (pq_recons.data(), xproj.data(), n * d2) / n;

        if (verbose)
            printf ("    Iteration %d (%d PQ iterations):"
                    "%.3f s, obj=%g\n", iter, pq_regular.cp.niter,
                    (getmillisecs () - t0) / 1000.0, pq_err);

        {
            float *u = tmp.data(), *vt = &tmp [d * d];
            float *sing_val = &tmp [2 * d * d];
            FINTEGER di = d, d2i = d2, ni = n;
            float one = 1, zero = 0;

            // torch.mm(xtrain:t(), pq_recons)
            sgemm_ ("Not", "Transposed",
                    &d2i, &di, &ni,
                    &one, pq_recons.data(), &d2i,
                    xtrain.data(), &di,
                    &zero, xxr.data(), &d2i);


            FINTEGER lwork = -1, info = -1;
            float worksz;
            // workspace query
            sgesvd_ ("All", "All",
                     &d2i, &di, xxr.data(), &d2i,
                     sing_val,
                     vt, &d2i, u, &di,
                     &worksz, &lwork, &info);

            lwork = int(worksz);
            std::vector<float> work (lwork);
            // u and vt swapped
            sgesvd_ ("All", "All",
                     &d2i, &di, xxr.data(), &d2i,
                     sing_val,
                     vt, &d2i, u, &di,
                     work.data(), &lwork, &info);

            sgemm_ ("Transposed", "Transposed",
                    &di, &d2i, &d2i,
                    &one, u, &di, vt, &d2i,
                    &zero, rotation, &di);

        }
        pq_regular.train_type = ProductQuantizer::Train_hot_start;
    }

    // revert A matrix
    if (d > d_in) {
        for (long i = 0; i < d_out; i++)
            memmove (&A[i * d_in], &A[i * d], sizeof(A[0]) * d_in);
        A.resize (d_in * d_out);
    }

    if (x != x_in)
        delete [] x;

    is_trained = true;
}




void OPQMatrix::reverse_transform (idx_t n, const float * xt,
                                   float *x) const
{
    transform_transpose (n, xt, x);
}

/*********************************************
 * IndexPreTransform
 *********************************************/

IndexPreTransform::IndexPreTransform ():
    index(nullptr), own_fields (false)
{
}


IndexPreTransform::IndexPreTransform (
        Index * index):
    Index (index->d, index->metric_type),
    index (index), own_fields (false)
{
    is_trained = index->is_trained;
    set_typename();
}




IndexPreTransform::IndexPreTransform (
        VectorTransform * ltrans,
        Index * index):
    Index (index->d, index->metric_type),
    index (index), own_fields (false)
{
    is_trained = index->is_trained;
    prepend_transform (ltrans);
    set_typename();
}

void IndexPreTransform::prepend_transform (VectorTransform *ltrans)
{
    FAISS_ASSERT (ltrans->d_out == d);
    is_trained = is_trained && ltrans->is_trained;
    chain.insert (chain.begin(), ltrans);
    d = ltrans->d_in;
    set_typename ();
}


void IndexPreTransform::set_typename ()
{
    // TODO correct this according to actual type
    index_typename = "PreLT[" + index->index_typename + "]";
}


IndexPreTransform::~IndexPreTransform ()
{
    if (own_fields) {
        for (int i = 0; i < chain.size(); i++)
            delete chain[i];
        delete index;
    }
}




void IndexPreTransform::train (idx_t n, const float *x)
{
    int last_untrained = 0;
    for (int i = 0; i < chain.size(); i++)
        if (!chain[i]->is_trained) last_untrained = i;
    if (!index->is_trained) last_untrained = chain.size();
    const float *prev_x = x;

    for (int i = 0; i <= last_untrained; i++) {
        if (i < chain.size()) {
            VectorTransform *ltrans = chain [i];
            if (!ltrans->is_trained)
                ltrans->train(n, prev_x);
        } else {
            index->train (n, prev_x);
        }
        if (i == last_untrained) break;

        float * xt = chain[i]->apply (n, prev_x);
        if (prev_x != x) delete [] prev_x;
        prev_x = xt;
    }

    if (prev_x != x) delete [] prev_x;
    is_trained = true;
}


const float *IndexPreTransform::apply_chain (idx_t n, const float *x) const
{
    const float *prev_x = x;
    for (int i = 0; i < chain.size(); i++) {
        float * xt = chain[i]->apply (n, prev_x);
        if (prev_x != x) delete [] prev_x;
        prev_x = xt;
    }
    return prev_x;
}

void IndexPreTransform::add (idx_t n, const float *x)
{
    FAISS_ASSERT (is_trained);
    const float *xt = apply_chain (n, x);
    index->add (n, xt);
    if (xt != x) delete [] xt;
    ntotal = index->ntotal;
}

void IndexPreTransform::add_with_ids (idx_t n, const float * x,
                                      const long *xids)
{
    FAISS_ASSERT (is_trained);
    const float *xt = apply_chain (n, x);
    index->add_with_ids (n, xt, xids);
    if (xt != x) delete [] xt;
    ntotal = index->ntotal;
}




void IndexPreTransform::search (idx_t n, const float *x, idx_t k,
                               float *distances, idx_t *labels) const
{
    FAISS_ASSERT (is_trained);
    const float *xt = apply_chain (n, x);
    index->search (n, xt, k, distances, labels);
    if (xt != x) delete [] xt;
}


void IndexPreTransform::reset () {
    index->reset();
    ntotal = 0;
}

long IndexPreTransform::remove_ids (const IDSelector & sel) {
    long nremove = index->remove_ids (sel);
    ntotal = index->ntotal;
    return nremove;
}


void IndexPreTransform::reconstruct_n (idx_t i0, idx_t ni, float *recons) const
{
    float *x = chain.empty() ? recons : new float [ni * index->d];
    // initial reconstruction
    index->reconstruct_n (i0, ni, x);

    // revert transformations from last to first
    for (int i = chain.size() - 1; i >= 0; i--) {
        float *x_pre = i == 0 ? recons : new float [chain[i]->d_in * ni];
        chain [i]->reverse_transform (ni, x, x_pre);
        delete [] x;
        x = x_pre;
    }
}



/*********************************************
 * RemapDimensionsTransform
 *********************************************/


RemapDimensionsTransform::RemapDimensionsTransform (
        int d_in, int d_out, const int *map_in):
    VectorTransform (d_in, d_out)
{
    map.resize (d_out);
    for (int i = 0; i < d_out; i++) {
        map[i] = map_in[i];
        FAISS_ASSERT (map[i] == -1 || (map[i] >= 0 && map[i] < d_in));
    }
}

RemapDimensionsTransform::RemapDimensionsTransform (
      int d_in, int d_out, bool uniform): VectorTransform (d_in, d_out)
{
    map.resize (d_out, -1);

    if (uniform) {
        if (d_in < d_out) {
            for (int i = 0; i < d_in; i++) {
                map [i * d_out / d_in] = i;
        }
        } else {
            for (int i = 0; i < d_out; i++) {
                map [i] = i * d_in / d_out;
            }
        }
    } else {
        for (int i = 0; i < d_in && i < d_out; i++)
            map [i] = i;
    }
}


void RemapDimensionsTransform::apply_noalloc (idx_t n, const float * x,
                                              float *xt) const
{
    for (idx_t i = 0; i < n; i++) {
        for (int j = 0; j < d_out; j++) {
            xt[j] = map[j] < 0 ? 0 : x[map[j]];
        }
        x += d_in;
        xt += d_out;
    }
}

void RemapDimensionsTransform::reverse_transform (idx_t n, const float * xt,
                                                  float *x) const
{
    memset (x, 0, sizeof (*x) * n * d_in);
    for (idx_t i = 0; i < n; i++) {
        for (int j = 0; j < d_out; j++) {
            if (map[j] >= 0) x[map[j]] = xt[j];
        }
        x += d_in;
        xt += d_out;
    }
}
