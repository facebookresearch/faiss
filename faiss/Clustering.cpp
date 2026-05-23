/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/Clustering.h>
#include <faiss/VectorTransform.h>
#include <faiss/impl/AuxIndexStructures.h>

#include <cinttypes>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <limits>

#include <faiss/IndexFlat.h>
#include <faiss/impl/ClusteringHelpers.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/kmeans1d.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/random.h>
#include <faiss/utils/utils.h>

namespace faiss {

Clustering::Clustering(int d_, int k_) : d(d_), k(k_) {}

Clustering::Clustering(int d_, int k_, const ClusteringParameters& cp)
        : ClusteringParameters(cp), d(d_), k(k_) {}

void Clustering::post_process_centroids() {
    if (spherical) {
        fvec_renorm_L2(d, k, centroids.data());
    }

    if (int_centroids) {
        for (size_t i = 0; i < centroids.size(); i++) {
            centroids[i] = roundf(centroids[i]);
        }
    }
}

void Clustering::train(
        idx_t nx,
        const float* x_in,
        Index& index,
        const float* weights) {
    train_encoded(
            nx,
            reinterpret_cast<const uint8_t*>(x_in),
            nullptr,
            index,
            weights);
}

void Clustering::train_encoded(
        idx_t nx,
        const uint8_t* x_in,
        const Index* codec,
        Index& index,
        const float* weights) {
    FAISS_THROW_IF_NOT_FMT(
            nx >= static_cast<idx_t>(k),
            "Number of training points (%" PRId64
            ") should be at least "
            "as large as number of clusters (%zd)",
            nx,
            k);

    FAISS_THROW_IF_NOT_FMT(
            (!codec || static_cast<size_t>(codec->d) == d),
            "Codec dimension %d not the same as data dimension %d",
            int(codec->d),
            int(d));

    FAISS_THROW_IF_NOT_FMT(
            static_cast<size_t>(index.d) == d,
            "Index dimension %d not the same as data dimension %d",
            int(index.d),
            int(d));

    double t0 = getmillisecs();

    if (!codec && check_input_data_for_NaNs) {
        // Check for NaNs in input data. Normally it is the user's
        // responsibility, but it may spare us some hard-to-debug
        // reports.
        const float* x = reinterpret_cast<const float*>(x_in);
        for (size_t i = 0; i < nx * d; i++) {
            FAISS_THROW_IF_NOT_MSG(
                    std::isfinite(x[i]), "input contains NaN's or Inf's");
        }
    }

    const uint8_t* x = x_in;
    std::unique_ptr<uint8_t[]> del1;
    std::unique_ptr<float[]> del3;
    size_t line_size = codec ? codec->sa_code_size() : sizeof(float) * d;

    if (static_cast<size_t>(nx) > k * max_points_per_centroid) {
        uint8_t* x_new;
        float* weights_new;
        nx = detail::subsample_training_set(
                *this, nx, x, line_size, weights, &x_new, &weights_new);
        del1.reset(x_new);
        x = x_new;
        del3.reset(weights_new);
        weights = weights_new;
    } else if (static_cast<size_t>(nx) < k * min_points_per_centroid) {
        fprintf(stderr,
                "WARNING clustering %" PRId64
                " points to %zd centroids: "
                "please provide at least %" PRId64 " training points\n",
                nx,
                k,
                idx_t(k) * min_points_per_centroid);
    }

    if (static_cast<size_t>(nx) == k) {
        // this is a corner case, just copy training set to clusters
        if (verbose) {
            printf("Number of training points (%" PRId64
                   ") same as number of "
                   "clusters, just copying\n",
                   nx);
        }
        centroids.resize(d * k);
        if (!codec) {
            memcpy(centroids.data(), x_in, sizeof(float) * d * k);
        } else {
            codec->sa_decode(nx, x_in, centroids.data());
        }

        // one fake iteration...
        ClusteringIterationStats stats = {0.0, 0.0, 0.0, 1.0, 0};
        iteration_stats.push_back(stats);

        index.reset();
        index.add(k, centroids.data());
        return;
    }

    if (verbose) {
        printf("Clustering %" PRId64
               " points in %zdD to %zd clusters, "
               "redo %d times, %d iterations\n",
               nx,
               d,
               k,
               nredo,
               niter);
        if (codec) {
            printf("Input data encoded in %zd bytes per vector\n",
                   codec->sa_code_size());
        }
    }

    std::unique_ptr<idx_t[]> assign(new idx_t[nx]);
    std::unique_ptr<float[]> dis(new float[nx]);

    // remember best iteration for redo
    bool lower_is_better = !is_similarity_metric(index.metric_type);
    float best_obj = lower_is_better ? HUGE_VALF : -HUGE_VALF;
    std::vector<ClusteringIterationStats> best_iteration_stats;
    std::vector<float> best_centroids;

    // support input centroids

    FAISS_THROW_IF_NOT_MSG(
            centroids.size() % d == 0,
            "size of provided input centroids not a multiple of dimension");

    size_t n_input_centroids = centroids.size() / d;

    if (verbose && n_input_centroids > 0) {
        printf("  Using %zd centroids provided as input (%sfrozen)\n",
               n_input_centroids,
               frozen_centroids ? "" : "not ");
    }

    double t_search_tot = 0;
    if (verbose) {
        printf("  Preprocessing in %.2f s\n", (getmillisecs() - t0) / 1000.);
    }
    t0 = getmillisecs();

    // initialize seed
    const uint64_t actual_seed = detail::get_actual_rng_seed(seed);

    // temporary buffer to decode vectors during the optimization
    std::vector<float> decode_buffer(codec ? d * decode_block_size : 0);

    for (int redo = 0; redo < nredo; redo++) {
        if (verbose && nredo > 1) {
            printf("Outer iteration %d / %d\n", redo, nredo);
        }

        // initialize centroids using the selected method
        centroids.resize(d * k);

        size_t k_to_init = k - n_input_centroids;
        if (k_to_init > 0) {
            // Fast path for RANDOM initialization - preserves exact original
            // behavior
            if (init_method == ClusteringInitMethod::RANDOM) {
                std::vector<int> perm(nx);
                rand_perm(perm.data(), nx, actual_seed + 1 + redo * 15486557L);
                for (size_t i = 0; i < k_to_init; i++) {
                    if (!codec) {
                        memcpy(centroids.data() + (n_input_centroids + i) * d,
                               x + perm[n_input_centroids + i] * line_size,
                               line_size);
                    } else {
                        codec->sa_decode(
                                1,
                                x + perm[n_input_centroids + i] * line_size,
                                centroids.data() + (n_input_centroids + i) * d);
                    }
                }
            } else {
                // For k-means++ and AFK-MC², we need all vectors decoded
                const float* x_float = nullptr;
                std::vector<float> x_decoded;

                if (!codec) {
                    x_float = reinterpret_cast<const float*>(x);
                } else {
                    // Decode all vectors for initialization
                    x_decoded.resize(nx * d);
                    codec->sa_decode(nx, x, x_decoded.data());
                    x_float = x_decoded.data();
                }

                ClusteringInitialization initializer(d, k_to_init);
                initializer.method = init_method;
                initializer.seed = actual_seed + 1 + redo * 15486557L;
                initializer.afkmc2_chain_length = afkmc2_chain_length;
                initializer.init_centroids(
                        nx,
                        x_float,
                        centroids.data() + n_input_centroids * d,
                        n_input_centroids,
                        n_input_centroids > 0 ? centroids.data() : nullptr);
            }
        }

        post_process_centroids();

        // prepare the index

        if (index.ntotal != 0) {
            index.reset();
        }

        if (!index.is_trained) {
            index.train(k, centroids.data());
        }

        index.add(k, centroids.data());

        // k-means iterations

        float obj = 0;
        for (int i = 0; i < niter; i++) {
            double t0s = getmillisecs();

            if (!codec) {
                index.search(
                        nx,
                        reinterpret_cast<const float*>(x),
                        1,
                        dis.get(),
                        assign.get());
            } else {
                // search by blocks of decode_block_size vectors
                size_t code_size = codec->sa_code_size();
                for (size_t i0 = 0; i0 < static_cast<size_t>(nx);
                     i0 += decode_block_size) {
                    size_t i1 = i0 + decode_block_size;
                    if (i1 > static_cast<size_t>(nx)) {
                        i1 = nx;
                    }
                    codec->sa_decode(
                            i1 - i0, x + code_size * i0, decode_buffer.data());
                    index.search(
                            i1 - i0,
                            decode_buffer.data(),
                            1,
                            dis.get() + i0,
                            assign.get() + i0);
                }
            }

            InterruptCallback::check();
            t_search_tot += getmillisecs() - t0s;

            // accumulate objective
            obj = 0;
            for (idx_t j = 0; j < nx; j++) {
                obj += dis[j];
            }

            // update the centroids
            std::vector<float> hassign(k);

            size_t k_frozen = frozen_centroids ? n_input_centroids : 0;
            detail::compute_centroids(
                    d,
                    k,
                    nx,
                    k_frozen,
                    x,
                    codec,
                    assign.get(),
                    weights,
                    hassign.data(),
                    centroids.data());

            int nsplit = detail::split_clusters(
                    d, k, nx, k_frozen, hassign.data(), centroids.data());

            // collect statistics
            ClusteringIterationStats stats = {
                    obj,
                    (getmillisecs() - t0) / 1000.0,
                    t_search_tot / 1000,
                    imbalance_factor(nx, static_cast<int>(k), assign.get()),
                    nsplit};
            iteration_stats.push_back(stats);

            if (verbose) {
                printf("  Iteration %d (%.2f s, search %.2f s): "
                       "objective=%g imbalance=%.3f nsplit=%d       \r",
                       i,
                       stats.time,
                       stats.time_search,
                       stats.obj,
                       stats.imbalance_factor,
                       nsplit);
                fflush(stdout);
            }

            post_process_centroids();

            // add centroids to index for the next iteration (or for output)

            index.reset();
            if (update_index) {
                index.train(k, centroids.data());
            }

            index.add(k, centroids.data());
            InterruptCallback::check();

            // Early stopping: if objective didn't change, we've converged.
            // Safe to access iteration_stats[size - 2] because we push_back
            // above, so size >= i + 1, and when i > 0 we have size >= 2.
            if (i > 0) {
                float prev_obj =
                        iteration_stats[iteration_stats.size() - 2].obj;

                double change = (prev_obj == 0)
                        ? std::numeric_limits<double>::max()
                        : std::abs(prev_obj - obj) / std::abs(prev_obj);

                if (change >= 0 && change <= early_stop_threshold) {
                    if (verbose) {
                        printf("\n  Converged at iteration %d: "
                               "objective did not change\n",
                               i);
                    }
                    break;
                }
            }
        }

        if (verbose) {
            printf("\n");
        }
        if (nredo > 1) {
            if ((lower_is_better && obj < best_obj) ||
                (!lower_is_better && obj > best_obj)) {
                if (verbose) {
                    printf("Objective improved: keep new clusters\n");
                }
                best_centroids = centroids;
                best_iteration_stats = iteration_stats;
                best_obj = obj;
            }
            index.reset();
        }
    }
    if (nredo > 1) {
        centroids = best_centroids;
        iteration_stats = best_iteration_stats;
        index.reset();
        index.add(k, best_centroids.data());
    }
}

Clustering1D::Clustering1D(int k_) : Clustering(1, k_) {}

Clustering1D::Clustering1D(int k_, const ClusteringParameters& cp)
        : Clustering(1, k_, cp) {}

void Clustering1D::train_exact(idx_t n, const float* x) {
    const float* xt = x;

    std::unique_ptr<uint8_t[]> del;
    if (static_cast<size_t>(n) > k * max_points_per_centroid) {
        uint8_t* x_new;
        float* weights_new;
        n = detail::subsample_training_set(
                *this,
                n,
                (uint8_t*)x,
                sizeof(float) * d,
                nullptr,
                &x_new,
                &weights_new);
        del.reset(x_new);
        xt = (float*)x_new;
    }

    centroids.resize(k);
    double uf = kmeans1d(xt, n, k, centroids.data());

    ClusteringIterationStats stats = {0.0, 0.0, 0.0, uf, 0};
    iteration_stats.push_back(stats);
}

float kmeans_clustering(
        size_t d,
        size_t n,
        size_t k,
        const float* x,
        float* centroids) {
    Clustering clus(static_cast<int>(d), static_cast<int>(k));
    clus.verbose = d * n * k > (size_t(1) << 30);
    // display logs if > 1Gflop per iteration
    IndexFlatL2 index(d);
    clus.train(n, x, index);
    memcpy(centroids, clus.centroids.data(), sizeof(*centroids) * d * k);
    return clus.iteration_stats.back().obj;
}

/******************************************************************************
 * ProgressiveDimClustering implementation
 ******************************************************************************/

ProgressiveDimClusteringParameters::ProgressiveDimClusteringParameters() {
    progressive_dim_steps = 10;
    apply_pca = true; // seems a good idea to do this by default
    niter = 10;       // reduce nb of iterations per step
}

Index* ProgressiveDimIndexFactory::operator()(int dim) {
    return new IndexFlatL2(dim);
}

ProgressiveDimClustering::ProgressiveDimClustering(int d_, int k_)
        : d(d_), k(k_) {}

ProgressiveDimClustering::ProgressiveDimClustering(
        int d_,
        int k_,
        const ProgressiveDimClusteringParameters& cp)
        : ProgressiveDimClusteringParameters(cp), d(d_), k(k_) {}

namespace {

void copy_columns(idx_t n, idx_t d1, const float* src, idx_t d2, float* dest) {
    idx_t d = std::min(d1, d2);
    for (idx_t i = 0; i < n; i++) {
        memcpy(dest, src, sizeof(float) * d);
        src += d1;
        dest += d2;
    }
}

} // namespace

void ProgressiveDimClustering::train(
        idx_t n,
        const float* x,
        ProgressiveDimIndexFactory& factory) {
    int d_prev = 0;

    PCAMatrix pca(static_cast<int>(d), static_cast<int>(d));

    std::vector<float> xbuf;
    if (apply_pca) {
        if (verbose) {
            printf("Training PCA transform\n");
        }
        pca.train(n, x);
        if (verbose) {
            printf("Apply PCA\n");
        }
        xbuf.resize(n * d);
        pca.apply_noalloc(n, x, xbuf.data());
        x = xbuf.data();
    }

    for (int iter = 0; iter < progressive_dim_steps; iter++) {
        int di = int(pow(d, (1. + iter) / progressive_dim_steps));
        if (verbose) {
            printf("Progressive dim step %d: cluster in dimension %d\n",
                   iter,
                   di);
        }
        std::unique_ptr<Index> clustering_index(factory(di));

        Clustering clus(di, static_cast<int>(k), *this);
        if (d_prev > 0) {
            // copy warm-start centroids (padded with 0s)
            clus.centroids.resize(k * di);
            copy_columns(
                    k, d_prev, centroids.data(), di, clus.centroids.data());
        }
        std::vector<float> xsub(n * di);
        copy_columns(n, d, x, di, xsub.data());

        clus.train(n, xsub.data(), *clustering_index.get());

        centroids = clus.centroids;
        iteration_stats.insert(
                iteration_stats.end(),
                clus.iteration_stats.begin(),
                clus.iteration_stats.end());

        d_prev = di;
    }

    if (apply_pca) {
        if (verbose) {
            printf("Revert PCA transform on centroids\n");
        }
        std::vector<float> cent_transformed(d * k);
        pca.reverse_transform(k, centroids.data(), cent_transformed.data());
        cent_transformed.swap(centroids);
    }
}

} // namespace faiss
