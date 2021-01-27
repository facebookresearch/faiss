/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/impl/NNDescent.h>

#include <iostream>
#include <mutex>
#include <string>

#include <faiss/impl/AuxIndexStructures.h>

namespace faiss {

constexpr int NUM_EVAL_POINTS = 100;
inline int insert_into_pool(Neighbor *addr, int size, Neighbor nn);

NNDescent::NNDescent(const int d, const int K) : d(d), K(K), rng(2021) {

  ntotal = 0;
  has_built = false;
  S = 10;
  R = 100;
  L = K;
  iter = 10;
  search_L = 0;
}

NNDescent::~NNDescent() {}

void NNDescent::join(DistanceComputer &qdis) {
#pragma omp parallel for default(shared) schedule(dynamic, 100)
  for (int n = 0; n < ntotal; n++) {
    graph[n].join([&](int i, int j) {
      if (i != j) {
        float dist = qdis.symmetric_dis(i, j);
        graph[i].insert(j, dist);
        graph[j].insert(i, dist);
      }
    });
  }
}

void NNDescent::update() {
#pragma omp parallel for
  for (int i = 0; i < ntotal; i++) {
    std::vector<int>().swap(graph[i].nn_new);
    std::vector<int>().swap(graph[i].nn_old);
  }

#pragma omp parallel for
  for (int n = 0; n < ntotal; ++n) {
    auto &nn = graph[n];
    std::sort(nn.pool.begin(), nn.pool.end());
    if (nn.pool.size() > L)
      nn.pool.resize(L);
    nn.pool.reserve(L);
    int maxl = std::min(nn.M + S, (int)nn.pool.size());
    int c = 0;
    int l = 0;

    while ((l < maxl) && (c < S)) {
      if (nn.pool[l].flag)
        ++c;
      ++l;
    }
    nn.M = l;
  }
#pragma omp parallel for
  for (int n = 0; n < ntotal; ++n) {
    auto &nnhd = graph[n];
    auto &nn_new = nnhd.nn_new;
    auto &nn_old = nnhd.nn_old;
    for (int l = 0; l < nnhd.M; ++l) {
      auto &nn = nnhd.pool[l];
      auto &nhood_o = graph[nn.id]; // nn on the other side of the edge

      if (nn.flag) {
        nn_new.push_back(nn.id);
        if (nn.distance > nhood_o.pool.back().distance) {
          LockGuard guard(nhood_o.lock);
          if (nhood_o.rnn_new.size() < R)
            nhood_o.rnn_new.push_back(n);
          else {
            int pos = rand() % R;
            nhood_o.rnn_new[pos] = n;
          }
        }
        nn.flag = false;
      } else {
        nn_old.push_back(nn.id);
        if (nn.distance > nhood_o.pool.back().distance) {
          LockGuard guard(nhood_o.lock);
          if (nhood_o.rnn_old.size() < R)
            nhood_o.rnn_old.push_back(n);
          else {
            int pos = rand() % R;
            nhood_o.rnn_old[pos] = n;
          }
        }
      }
    }
    std::make_heap(nnhd.pool.begin(), nnhd.pool.end());
  }
#pragma omp parallel for
  for (int i = 0; i < ntotal; ++i) {
    auto &nn_new = graph[i].nn_new;
    auto &nn_old = graph[i].nn_old;
    auto &rnn_new = graph[i].rnn_new;
    auto &rnn_old = graph[i].rnn_old;
    if (R && rnn_new.size() > R) {
      std::random_shuffle(rnn_new.begin(), rnn_new.end());
      rnn_new.resize(R);
    }
    nn_new.insert(nn_new.end(), rnn_new.begin(), rnn_new.end());
    if (R && rnn_old.size() > R) {
      std::random_shuffle(rnn_old.begin(), rnn_old.end());
      rnn_old.resize(R);
    }
    nn_old.insert(nn_old.end(), rnn_old.begin(), rnn_old.end());
    if (nn_old.size() > R * 2) {
      nn_old.resize(R * 2);
      nn_old.reserve(R * 2);
    }
    std::vector<int>().swap(graph[i].rnn_new);
    std::vector<int>().swap(graph[i].rnn_old);
  }
}

void NNDescent::nndescent(DistanceComputer &qdis, bool verbose) {
  int num_eval_points = std::min(NUM_EVAL_POINTS, ntotal);
  std::vector<int> eval_points(num_eval_points);
  std::vector<std::vector<int>> acc_eval_set(num_eval_points);
  gen_random(rng, &eval_points[0], eval_points.size(), ntotal);
  generate_eval_set(qdis, eval_points, acc_eval_set, ntotal);
  for (int it = 0; it < iter; it++) {
    join(qdis);
    update();

    if (verbose) {
      float recall = eval_recall(eval_points, acc_eval_set);
      std::cout << "iter: " << it << " recall: " << recall << std::endl;
    }
  }
}

void NNDescent::generate_eval_set(DistanceComputer &qdis, std::vector<int> &c,
                                  std::vector<std::vector<int>> &v, int N) {
#pragma omp parallel for
  for (int i = 0; i < c.size(); i++) {
    std::vector<Neighbor> tmp;
    for (int j = 0; j < N; j++) {
      float dist = qdis.symmetric_dis(c[i], j);
      tmp.push_back(Neighbor(j, dist, true));
    }

    std::partial_sort(tmp.begin(), tmp.begin() + K, tmp.end());
    for (int j = 0; j < K; j++) {
      v[i].push_back(tmp[j].id);
    }
  }
}

float NNDescent::eval_recall(std::vector<int> &eval_points,
                             std::vector<std::vector<int>> &acc_eval_set) {
  float mean_acc = 0.0f;
  for (size_t i = 0; i < eval_points.size(); i++) {
    float acc = 0;
    auto &g = graph[eval_points[i]].pool;
    auto &v = acc_eval_set[i];
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

void NNDescent::init_graph(DistanceComputer &qdis) {

  graph.reserve(ntotal);
  for (int i = 0; i < ntotal; i++) {
    graph.push_back(Nhood(L, S, rng, (int)ntotal));
  }
#pragma omp parallel for
  for (int i = 0; i < ntotal; i++) {
    std::vector<int> tmp(S);

    gen_random(rng, tmp.data(), S, ntotal);

    for (int j = 0; j < S; j++) {
      int id = tmp[j];
      if (id == i)
        continue;
      float dist = qdis.symmetric_dis(i, id);

      graph[i].pool.push_back(Neighbor(id, dist, true));
    }
    std::make_heap(graph[i].pool.begin(), graph[i].pool.end());
    graph[i].pool.reserve(L);
  }
}

void NNDescent::build(DistanceComputer &qdis, const int n, bool verbose) {
  FAISS_THROW_IF_NOT_MSG(L >= K, "L should be >= K in NNDescent.build");

  if (verbose) {
    printf("Parameters: K=%d, S=%d, R=%d, L=%d, iter=%d\n", K, S, R, L, iter);
  }

  ntotal = n;
  init_graph(qdis);
  nndescent(qdis, verbose);

  final_graph.resize(ntotal * K);

  for (int i = 0; i < ntotal; i++) {
    // std::vector<int> tmp;
    std::sort(graph[i].pool.begin(), graph[i].pool.end());
    for (int j = 0; j < K; j++) {
      // tmp.push_back(graph[i].pool[j].id);
      FAISS_ASSERT(graph[i].pool[j].id < ntotal);
      final_graph[i * K + j] = graph[i].pool[j].id;
    }

    std::vector<Neighbor>().swap(graph[i].pool);
    std::vector<int>().swap(graph[i].nn_new);
    std::vector<int>().swap(graph[i].nn_old);
    std::vector<int>().swap(graph[i].rnn_new);
    std::vector<int>().swap(graph[i].rnn_new);
  }
  std::vector<Nhood>().swap(graph);
  has_built = true;

  std::cout << "Added " << ntotal << " points into index" << std::endl;
}

void NNDescent::search(DistanceComputer &qdis, const int topk, idx_t *indices,
                       float *dists, VisitedTable &vt) const {

  FAISS_THROW_IF_NOT_MSG(has_built, "The index is not build yet.");
  FAISS_THROW_IF_NOT_MSG(search_L >= topk,
                         "search_L should be >= k in NNDescent.search");

  std::vector<Neighbor> retset(search_L + 1);
  std::vector<int> init_ids(search_L);
  gen_random(rng, init_ids.data(), search_L, ntotal);

  for (int i = 0; i < search_L; i++) {
    int id = init_ids[i];
    float dist = qdis(id);
    retset[i] = Neighbor(id, dist, true);
  }

  std::sort(retset.begin(), retset.begin() + search_L);
  int k = 0;
  while (k < search_L) {
    int nk = search_L;

    if (retset[k].flag) {
      retset[k].flag = false;
      int n = retset[k].id;

      for (int m = 0; m < K; ++m) {
        int id = final_graph[n * K + m];
        if (vt.get(id))
          continue;

        vt.set(id);
        float dist = qdis(id);
        if (dist >= retset[search_L - 1].distance)
          continue;

        Neighbor nn(id, dist, true);
        int r = insert_into_pool(retset.data(), search_L, nn);

        if (r < nk)
          nk = r;
      }
      // lock to here
    }
    if (nk <= k)
      k = nk;
    else
      ++k;
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
  std::vector<int>().swap(final_graph);
}

void gen_random(std::mt19937 &rng, int *addr, const int size, const int N) {
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

inline int insert_into_pool(Neighbor *addr, int size, Neighbor nn) {
  // find the location to insert
  int left = 0, right = size - 1;
  if (addr[left].distance > nn.distance) {
    memmove((char *)&addr[left + 1], &addr[left], size * sizeof(Neighbor));
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
  memmove((char *)&addr[right + 1], &addr[right],
          (size - right) * sizeof(Neighbor));
  addr[right] = nn;
  return right;
}

} // namespace faiss
