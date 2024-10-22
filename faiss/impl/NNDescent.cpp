/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/impl/NNDescent.h>

#include <mutex>
#include <string>

#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/DistanceComputer.h>

namespace faiss {

using LockGuard = std::lock_guard<std::mutex>;

namespace nndescent {

void gen_random(std::mt19937& rng, int* addr, const int size, const int N);

Nhood::Nhood(int l, int s, std::mt19937& rng, int N) {
    M = s;
    nn_new.resize(s * 2);
    gen_random(rng, nn_new.data(), (int)nn_new.size(), N);
}

/// Copy operator
Nhood& Nhood::operator=(const Nhood& other) {
    M = other.M;
    std::copy(
            other.nn_new.begin(),
            other.nn_new.end(),
            std::back_inserter(nn_new));
    nn_new.reserve(other.nn_new.capacity());
    pool.reserve(other.pool.capacity());
    return *this;
}

/// Copy constructor
Nhood::Nhood(const Nhood& other) {
    M = other.M;
    std::copy(
            other.nn_new.begin(),
            other.nn_new.end(),
            std::back_inserter(nn_new));
    nn_new.reserve(other.nn_new.capacity());
    pool.reserve(other.pool.capacity());
}

/// Insert a point into the candidate pool
void Nhood::insert(int id, float dist) {
    LockGuard guard(lock);
    if (dist > pool.front().distance)
        return;
    for (int i = 0; i < pool.size(); i++) {
        if (id == pool[i].id)
            return;
    }
    if (pool.size() < pool.capacity()) {
        pool.push_back(Neighbor(id, dist, true));
        std::push_heap(pool.begin(), pool.end());
    } else {
        std::pop_heap(pool.begin(), pool.end());
        pool[pool.size() - 1] = Neighbor(id, dist, true);
        std::push_heap(pool.begin(), pool.end());
    }
}

/// In local join, two objects are compared only if at least
/// one of them is new.
template <typename C>
void Nhood::join(C callback) const {
    for (int const i : nn_new) {
        for (int const j : nn_new) {
            if (i < j) {
                callback(i, j);
            }
        }
        for (int j : nn_old) {
            callback(i, j);
        }
    }
}

void gen_random(std::mt19937& rng, int* addr, const int size, const int N) {
    for (int i = 0; i < size; ++i) {
        addr[i] = rng() % (N - size);
    }
    std::sort(addr, addr + size);
    for (int i = 1; i < size; ++i) {
        if (addr[i] <= addr[i - 1]) {
            addr[i] = addr[i - 1] + 1;
        }
    }
    int off = rng() % N;
    for (int i = 0; i < size; ++i) {
        addr[i] = (addr[i] + off) % N;
    }
}

// Insert a new point into the candidate pool in ascending order
int insert_into_pool(Neighbor* addr, int size, Neighbor nn) {
    // find the location to insert
    int left = 0, right = size - 1;
    if (addr[left].distance > nn.distance) {
        memmove((char*)&addr[left + 1], &addr[left], size * sizeof(Neighbor));
        addr[left] = nn;
        return left;
    }
    if (addr[right].distance < nn.distance) {
        addr[size] = nn;
        return size;
    }
    while (left < right - 1) {
        int mid = (left + right) / 2;
        if (addr[mid].distance > nn.distance)
            right = mid;
        else
            left = mid;
    }
    // check equal ID

    while (left > 0) {
        if (addr[left].distance < nn.distance)
            break;
        if (addr[left].id == nn.id)
            return size + 1;
        left--;
    }
    if (addr[left].id == nn.id || addr[right].id == nn.id)
        return size + 1;
    memmove((char*)&addr[right + 1],
            &addr[right],
            (size - right) * sizeof(Neighbor));
    addr[right] = nn;
    return right;
}

} // namespace nndescent

using namespace nndescent;

constexpr int NUM_EVAL_POINTS = 100;

NNDescent::NNDescent(const int d, const int K) : K(K), d(d) {
    L = K + 50;
}

NNDescent::~NNDescent() {}

void NNDescent::join(DistanceComputer& qdis) {
    idx_t check_period = InterruptCallback::get_period_hint(d * search_L);
    for (idx_t i0 = 0; i0 < (idx_t)ntotal; i0 += check_period) {
        idx_t i1 = std::min(i0 + check_period, (idx_t)ntotal);
#pragma omp parallel for default(shared) schedule(dynamic, 100)
        for (idx_t n = i0; n < i1; n++) {
            graph[n].join([&](int i, int j) {
                if (i != j) {
                    float dist = qdis.symmetric_dis(i, j);
                    graph[i].insert(j, dist);
                    graph[j].insert(i, dist);
                }
            });
        }
        InterruptCallback::check();
    }
}

/// Sample neighbors for each node to peform local join later
/// Store them in nn_new and nn_old
void NNDescent::update() {
    // Step 1.
    // Clear all nn_new and nn_old
#pragma omp parallel for
    for (int i = 0; i < ntotal; i++) {
        std::vector<int>().swap(graph[i].nn_new);
        std::vector<int>().swap(graph[i].nn_old);
    }

    // Step 2.
    // Compute the number of neighbors which is new i.e. flag is true
    // in the candidate pool. This must not exceed the sample number S.
    // That means We only select S new neighbors.
#pragma omp parallel for
    for (int n = 0; n < ntotal; ++n) {
        auto& nn = graph[n];
        std::sort(nn.pool.begin(), nn.pool.end());

        if (nn.pool.size() > L)
            nn.pool.resize(L);
        nn.pool.reserve(L); // keep the pool size be L

        int maxl = std::min(nn.M + S, (int)nn.pool.size());
        int c = 0;
        int l = 0;

        while ((l < maxl) && (c < S)) {
            if (nn.pool[l].flag) {
                ++c;
            }
            ++l;
        }
        nn.M = l;
    }

    // Step 3.
    // Find reverse links for each node
    // Randomly choose R reverse links.
#pragma omp parallel
    {
        std::mt19937 rng(random_seed * 5081 + omp_get_thread_num());
#pragma omp for
        for (int n = 0; n < ntotal; ++n) {
            auto& node = graph[n];
            auto& nn_new = node.nn_new;
            auto& nn_old = node.nn_old;

            for (int l = 0; l < node.M; ++l) {
                auto& nn = node.pool[l];
                auto& other = graph[nn.id]; // the other side of the edge

                if (nn.flag) { // the node is inserted newly
                    // push the neighbor into nn_new
                    nn_new.push_back(nn.id);
                    // push itself into other.rnn_new if it is not in
                    // the candidate pool of the other side
                    if (nn.distance > other.pool.back().distance) {
                        LockGuard guard(other.lock);
                        if (other.rnn_new.size() < R) {
                            other.rnn_new.push_back(n);
                        } else {
                            int pos = rng() % R;
                            other.rnn_new[pos] = n;
                        }
                    }
                    nn.flag = false;

                } else { // the node is old
                    // push the neighbor into nn_old
                    nn_old.push_back(nn.id);
                    // push itself into other.rnn_old if it is not in
                    // the candidate pool of the other side
                    if (nn.distance > other.pool.back().distance) {
                        LockGuard guard(other.lock);
                        if (other.rnn_old.size() < R) {
                            other.rnn_old.push_back(n);
                        } else {
                            int pos = rng() % R;
                            other.rnn_old[pos] = n;
                        }
                    }
                }
            }
            // make heap to join later (in join() function)
            std::make_heap(node.pool.begin(), node.pool.end());
        }
    }

    // Step 4.
    // Combine the forward and the reverse links
    // R = 0 means no reverse links are used.
#pragma omp parallel for
    for (int i = 0; i < ntotal; ++i) {
        auto& nn_new = graph[i].nn_new;
        auto& nn_old = graph[i].nn_old;
        auto& rnn_new = graph[i].rnn_new;
        auto& rnn_old = graph[i].rnn_old;

        nn_new.insert(nn_new.end(), rnn_new.begin(), rnn_new.end());
        nn_old.insert(nn_old.end(), rnn_old.begin(), rnn_old.end());
        if (nn_old.size() > R * 2) {
            nn_old.resize(R * 2);
            nn_old.reserve(R * 2);
        }

        std::vector<int>().swap(graph[i].rnn_new);
        std::vector<int>().swap(graph[i].rnn_old);
    }
}

void NNDescent::nndescent(DistanceComputer& qdis, bool verbose) {
    int num_eval_points = std::min(NUM_EVAL_POINTS, ntotal);
    std::vector<int> eval_points(num_eval_points);
    std::vector<std::vector<int>> acc_eval_set(num_eval_points);
    std::mt19937 rng(random_seed * 6577 + omp_get_thread_num());
    gen_random(rng, eval_points.data(), eval_points.size(), ntotal);
    generate_eval_set(qdis, eval_points, acc_eval_set, ntotal);
    for (int it = 0; it < iter; it++) {
        join(qdis);
        update();

        if (verbose) {
            float recall = eval_recall(eval_points, acc_eval_set);
            printf("Iter: %d, recall@%d: %lf\n", it, K, recall);
        }
    }
}

/// Sample a small number of points to evaluate the quality of KNNG built
void NNDescent::generate_eval_set(
        DistanceComputer& qdis,
        std::vector<int>& c,
        std::vector<std::vector<int>>& v,
        int N) {
#pragma omp parallel for
    for (int i = 0; i < c.size(); i++) {
        std::vector<Neighbor> tmp;
        for (int j = 0; j < N; j++) {
            if (c[i] == j) {
                continue; // skip itself
            }
            float dist = qdis.symmetric_dis(c[i], j);
            tmp.push_back(Neighbor(j, dist, true));
        }

        std::partial_sort(tmp.begin(), tmp.begin() + K, tmp.end());
        for (int j = 0; j < K; j++) {
            v[i].push_back(tmp[j].id);
        }
    }
}

/// Evaluate the quality of KNNG built
float NNDescent::eval_recall(
        std::vector<int>& eval_points,
        std::vector<std::vector<int>>& acc_eval_set) {
    float mean_acc = 0.0f;
    for (size_t i = 0; i < eval_points.size(); i++) {
        float acc = 0;
        std::vector<Neighbor>& g = graph[eval_points[i]].pool;
        std::vector<int>& v = acc_eval_set[i];
        for (size_t j = 0; j < g.size(); j++) {
            for (size_t k = 0; k < v.size(); k++) {
                if (g[j].id == v[k]) {
                    acc++;
                    break;
                }
            }
        }
        mean_acc += acc / v.size();
    }
    return mean_acc / eval_points.size();
}

/// Initialize the KNN graph randomly
void NNDescent::init_graph(DistanceComputer& qdis) {
    graph.reserve(ntotal);
    {
        std::mt19937 rng(random_seed * 6007);
        for (int i = 0; i < ntotal; i++) {
            graph.push_back(Nhood(L, S, rng, (int)ntotal));
        }
    }
#pragma omp parallel
    {
        std::mt19937 rng(random_seed * 7741 + omp_get_thread_num());
#pragma omp for
        for (int i = 0; i < ntotal; i++) {
            std::vector<int> tmp(S);

            gen_random(rng, tmp.data(), S, ntotal);

            for (int j = 0; j < S; j++) {
                int id = tmp[j];
                if (id == i) {
                    continue;
                }
                float dist = qdis.symmetric_dis(i, id);

                graph[i].pool.push_back(Neighbor(id, dist, true));
            }
            std::make_heap(graph[i].pool.begin(), graph[i].pool.end());
            graph[i].pool.reserve(L);
        }
    }
}

void NNDescent::build(DistanceComputer& qdis, const int n, bool verbose) {
    FAISS_THROW_IF_NOT_MSG(L >= K, "L should be >= K in NNDescent.build");
    FAISS_THROW_IF_NOT_FMT(
            n > NUM_EVAL_POINTS,
            "NNDescent.build cannot build a graph smaller than %d",
            int(NUM_EVAL_POINTS));

    if (verbose) {
        printf("Parameters: K=%d, S=%d, R=%d, L=%d, iter=%d\n",
               K,
               S,
               R,
               L,
               iter);
    }

    ntotal = n;
    init_graph(qdis);
    nndescent(qdis, verbose);

    final_graph.resize(ntotal * K);

    // Store the neighbor link structure into final_graph
    // Clear the old graph
    for (int i = 0; i < ntotal; i++) {
        std::sort(graph[i].pool.begin(), graph[i].pool.end());
        for (int j = 0; j < K; j++) {
            FAISS_ASSERT(graph[i].pool[j].id < ntotal);
            final_graph[i * K + j] = graph[i].pool[j].id;
        }
    }
    std::vector<Nhood>().swap(graph);
    has_built = true;

    if (verbose) {
        printf("Added %d points into the index\n", ntotal);
    }
}

void NNDescent::search(
        DistanceComputer& qdis,
        const int topk,
        idx_t* indices,
        float* dists,
        VisitedTable& vt) const {
    FAISS_THROW_IF_NOT_MSG(has_built, "The index is not build yet.");
    int L_2 = std::max(search_L, topk);

    // candidate pool, the K best items is the result.
    std::vector<Neighbor> retset(L_2 + 1);

    // Randomly choose L_2 points to initialize the candidate pool
    std::vector<int> init_ids(L_2);
    std::mt19937 rng(random_seed);

    gen_random(rng, init_ids.data(), L_2, ntotal);
    for (int i = 0; i < L_2; i++) {
        int id = init_ids[i];
        float dist = qdis(id);
        retset[i] = Neighbor(id, dist, true);
    }

    // Maintain the candidate pool in ascending order
    std::sort(retset.begin(), retset.begin() + L_2);

    int k = 0;

    // Stop until the smallest position updated is >= L_2
    while (k < L_2) {
        int nk = L_2;

        if (retset[k].flag) {
            retset[k].flag = false;
            int n = retset[k].id;

            for (int m = 0; m < K; ++m) {
                int id = final_graph[n * K + m];
                if (vt.get(id)) {
                    continue;
                }

                vt.set(id);
                float dist = qdis(id);
                if (dist >= retset[L_2 - 1].distance) {
                    continue;
                }

                Neighbor nn(id, dist, true);
                int r = insert_into_pool(retset.data(), L_2, nn);

                if (r < nk)
                    nk = r;
            }
        }
        if (nk <= k) {
            k = nk;
        } else {
            ++k;
        }
    }
    for (size_t i = 0; i < topk; i++) {
        indices[i] = retset[i].id;
        dists[i] = retset[i].distance;
    }

    vt.advance();
}

void NNDescent::reset() {
    has_built = false;
    ntotal = 0;
    final_graph.resize(0);
}

} // namespace faiss
