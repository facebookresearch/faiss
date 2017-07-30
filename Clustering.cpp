/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

/* Copyright 2004-present Facebook. All Rights Reserved.
   kmeans clustering routines
*/

#include "Clustering.h"



#include <cmath>
#include <cstdio>
#include <cstring>

#include "utils.h"
#include "FaissAssert.h"
#include "IndexFlat.h"

namespace faiss {

ClusteringParameters::ClusteringParameters ():
    niter(25),
    nredo(1),
    verbose(false), spherical(false),
    update_index(false),
    min_points_per_centroid(39),
    max_points_per_centroid(256),
    seed(1234)
{}
// 39 corresponds to 10000 / 256 -> to avoid warnings on PQ tests with randu10k


Clustering::Clustering (int d, int k):
    d(d), k(k) {}

Clustering::Clustering (int d, int k, const ClusteringParameters &cp):
    ClusteringParameters (cp), d(d), k(k) {}



static double imbalance_factor (int n, int k, long *assign) {
    std::vector<int> hist(k, 0);
    for (int i = 0; i < n; i++)
        hist[assign[i]]++;

    double tot = 0, uf = 0;

    for (int i = 0 ; i < k ; i++) {
        tot += hist[i];
        uf += hist[i] * (double) hist[i];
    }
    uf = uf * k / (tot * tot);

    return uf;
}




void Clustering::train (idx_t nx, const float *x_in, Index & index) {
    FAISS_THROW_IF_NOT_MSG (nx >= k,
                    "need at least as many training points as clusters");

    double t0 = getmillisecs();

    // yes it is the user's responsibility, but it may spare us some
    // hard-to-debug reports.
    for (size_t i = 0; i < nx * d; i++) {
      FAISS_THROW_IF_NOT_MSG (finite (x_in[i]),
                        "input contains NaN's or Inf's");
    }

    const float *x = x_in;
    ScopeDeleter<float> del1;

    if (nx > k * max_points_per_centroid) {
        if (verbose)
            printf("Sampling a subset of %ld / %ld for training\n",
                   k * max_points_per_centroid, nx);
        std::vector<int> perm (nx);
        rand_perm (perm.data (), nx, seed);
        nx = k * max_points_per_centroid;
        float * x_new = new float [nx * d];
        for (idx_t i = 0; i < nx; i++)
            memcpy (x_new + i * d, x + perm[i] * d, sizeof(x_new[0]) * d);
        x = x_new;
        del1.set (x);
    } else if (nx < k * min_points_per_centroid) {
        fprintf (stderr,
                 "WARNING clustering %ld points to %ld centroids: "
                 "please provide at least %ld training points\n",
                 nx, k, idx_t(k) * min_points_per_centroid);
    }


    if (verbose)
        printf("Clustering %d points in %ldD to %ld clusters, "
               "redo %d times, %d iterations\n",
               int(nx), d, k, nredo, niter);


    idx_t * assign = new idx_t[nx];
    ScopeDeleter<idx_t> del (assign);
    float * dis = new float[nx];
    ScopeDeleter<float> del2(dis);

    float best_err = 1e50;
    double t_search_tot = 0;
    if (verbose) {
        printf("  Preprocessing in %.2f s\n",
               (getmillisecs() - t0)/1000.);
    }
    t0 = getmillisecs();

    for (int redo = 0; redo < nredo; redo++) {

        std::vector<float> buf_centroids;

        std::vector<float> &cur_centroids =
            nredo == 1 ? centroids : buf_centroids;

        if (verbose && nredo > 1) {
            printf("Outer iteration %d / %d\n", redo, nredo);
        }

        if (cur_centroids.size() == 0) {
            // initialize centroids with random points from the dataset
            cur_centroids.resize (d * k);
            std::vector<int> perm (nx);

            rand_perm (perm.data(), nx, seed + 1 + redo * 15486557L);
#pragma omp parallel for
            for (int i = 0; i < k ; i++)
                memcpy (&cur_centroids[i * d], x + perm[i] * d,
                        d * sizeof (float));
        } else { // assume user provides some meaningful initialization
            FAISS_THROW_IF_NOT (cur_centroids.size() == d * k);
            FAISS_THROW_IF_NOT_MSG (nredo == 1,
                              "will redo with same initialization");
        }

        if (spherical)
            fvec_renorm_L2 (d, k, cur_centroids.data());

        if (!index.is_trained)
            index.train (k, cur_centroids.data());

        FAISS_THROW_IF_NOT (index.ntotal == 0);
        index.add (k, cur_centroids.data());
        float err = 0;
        for (int i = 0; i < niter; i++) {
            double t0s = getmillisecs();
            index.search (nx, x, 1, dis, assign);
            t_search_tot += getmillisecs() - t0s;

            err = 0;
            for (int j = 0; j < nx; j++)
                err += dis[j];
            obj.push_back (err);

            int nsplit = km_update_centroids (x, cur_centroids.data(),
                                              assign, d, k, nx);

            if (verbose) {
                printf ("  Iteration %d (%.2f s, search %.2f s): "
                        "objective=%g imbalance=%.3f nsplit=%d       \r",
                        i, (getmillisecs() - t0) / 1000.0,
                        t_search_tot / 1000,
                        err, imbalance_factor (nx, k, assign),
                        nsplit);
                fflush (stdout);
            }

            if (spherical)
                fvec_renorm_L2 (d, k, cur_centroids.data());

            index.reset ();
            if (update_index)
                index.train (k, cur_centroids.data());

            assert (index.ntotal == 0);
            index.add (k, cur_centroids.data());
        }
        if (verbose) printf("\n");
        if (nredo > 1) {
            if (err < best_err) {
                if (verbose)
                    printf ("Objective improved: keep new clusters\n");
                centroids = buf_centroids;
                best_err = err;
            }
            index.reset ();
        }
    }

}

float kmeans_clustering (size_t d, size_t n, size_t k,
                         const float *x,
                         float *centroids)
{
    Clustering clus (d, k);
    clus.verbose = d * n * k > (1L << 30);
    // display logs if > 1Gflop per iteration
    IndexFlatL2 index (d);
    clus.train (n, x, index);
    memcpy(centroids, clus.centroids.data(), sizeof(*centroids) * d * k);
    return clus.obj.back();
}

} // namespace faiss
