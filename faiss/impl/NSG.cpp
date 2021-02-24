/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <faiss/impl/NSG.h>

#include <algorithm>
#include <mutex>
#include <stack>
#include <string>

#include <faiss/impl/AuxIndexStructures.h>

namespace faiss {

namespace nsg {

DistanceComputer *storage_distance_computer(const Index *storage) {
  if (storage->metric_type == METRIC_INNER_PRODUCT) {
    return new NegativeDistanceComputer(storage->get_distance_computer());
  } else {
    return storage->get_distance_computer();
  }
}

}  // namespace faiss::nsg

using namespace nsg;

using LockGuard = std::lock_guard<std::mutex>;

struct Neighbor {
  int id;
  float distance;
  bool flag;

  Neighbor() = default;
  Neighbor(int id, float distance, bool f)
      : id(id), distance(distance), flag(f) {}

  inline bool operator<(const Neighbor &other) const {
    return distance < other.distance;
  }
};

struct SimpleNeighbor {
  int id;
  float distance;

  SimpleNeighbor() = default;
  SimpleNeighbor(int id, float distance) : id(id), distance(distance) {}

  inline bool operator<(const SimpleNeighbor &other) const {
    return distance < other.distance;
  }
};

inline int insert_into_pool(Neighbor *addr, int K, Neighbor nn) {
  // find the location to insert
  int left = 0, right = K - 1;
  if (addr[left].distance > nn.distance) {
    memmove((char *)&addr[left + 1], &addr[left], K * sizeof(Neighbor));
    addr[left] = nn;
    return left;
  }
  if (addr[right].distance < nn.distance) {
    addr[K] = nn;
    return K;
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
      return K + 1;
    left--;
  }
  if (addr[left].id == nn.id || addr[right].id == nn.id)
    return K + 1;
  memmove((char *)&addr[right + 1], &addr[right],
          (K - right) * sizeof(Neighbor));
  addr[right] = nn;
  return right;
}

NSG::NSG(int R) : R(R) {
  L = R;
  C = R * 10;
  ntotal = 0;
  is_built = false;
}

void NSG::search(DistanceComputer &dis, int k, idx_t *I, float *D,
                 VisitedTable &vt) const {

  FAISS_THROW_IF_NOT(is_built);
  FAISS_THROW_IF_NOT(final_graph);

  int pool_size = std::max(search_L, k);
  std::vector<Neighbor> retset, tmp;
  search_on_graph<false, int>(*final_graph, dis, vt, enterpoint, pool_size,
                              retset, tmp);

  std::partial_sort(retset.begin(), retset.begin() + k,
                    retset.begin() + pool_size);

  for (size_t i = 0; i < k; i++) {
    I[i] = retset[i].id;
    D[i] = retset[i].distance;
  }
}

void NSG::build(Index *storage, idx_t n, const nsg::Graph<idx_t> &knn_graph, bool verbose) {
  FAISS_THROW_IF_NOT(!is_built && ntotal == 0);

  if (verbose) { 
    printf("R=%d, L=%d, C=%d\n", R, L, C);
  }

  ntotal = n;
  init_graph(storage, knn_graph);

  // SimpleNeighbor *graph = new SimpleNeighbor[n * R];
  nsg::Graph<SimpleNeighbor> tmp_graph(n, R);

  link(storage, knn_graph, tmp_graph);

  final_graph = std::make_shared<nsg::Graph<int>>(n, R);
  for (int i = 0; i < n; i++) {
    for (int j = 0; j < R; j++) {
      final_graph->at(i, j) = tmp_graph.at(i, j).id;
    }
  }

  tree_grow(storage);
  check_graph();
  is_built = true;

  if (verbose) { 
    int max = 0, min = 1e6;
    double avg = 0;
    for (int i = 0; i < n; i++) {
      int size = 0;
      while (size < R && final_graph->at(i, size) != EMPTY_ID) {
        size += 1;
      }
      max = std::max(size, max);
      min = std::min(size, min);
      avg += size;
    }
    avg = avg / n;
    printf("Degree Statistics: Max = %d, Min = %d, Avg = %lf\n",
           max, min, avg);
  }  
}

void NSG::reset() {
  final_graph.reset();
  ntotal = 0;
  is_built = false;
}

void NSG::init_graph(Index *storage, const nsg::Graph<idx_t> &knn_graph) {
  int d = storage->d;
  int n = storage->ntotal;

  float *center = new float[d];
  float *tmp = new float[d];

  for (int i = 0; i < d; i++) {
    center[i] = 0;
  }

  for (int i = 0; i < n; i++) {
    storage->reconstruct(i, tmp);
    for (int j = 0; j < d; j++) {
      center[j] += tmp[j];
    }
  }

  for (int i = 0; i < d; i++) {
    center[i] /= n;
  }

  std::vector<Neighbor> retset, pool;
  // random initialize navigating point
  int ep = rand() % n;
  DistanceComputer *dis = storage_distance_computer(storage);
  ScopeDeleter1<DistanceComputer> del(dis);

  dis->set_query(center);
  VisitedTable vt(ntotal);

  // Do not collect the visited set
  search_on_graph<false, idx_t>(knn_graph, *dis, vt, ep, L, retset, pool);

  enterpoint = retset[0].id;
}

template <bool collect_fullset, class index_t>
void NSG::search_on_graph(const nsg::Graph<index_t> &graph,
                          DistanceComputer &dis, VisitedTable &vt, int ep,
                          int pool_size, std::vector<Neighbor> &retset,
                          std::vector<Neighbor> &fullset) const {
  retset.resize(pool_size + 1);
  std::vector<int> init_ids(pool_size);

  int num_ids = 0;
  for (int i = 0; i < init_ids.size() && i < graph.K; i++) {
    int id = (int)graph.at(ep, i);
    if (id < 0 || id >= ntotal)
      continue;

    init_ids[i] = id;
    vt.set(id);
    num_ids += 1;
  }

  while (num_ids < pool_size) {
    int id = rand() % ntotal;
    if (vt.get(id))
      continue;

    init_ids[num_ids] = id;
    num_ids++;
    vt.set(id);
  }

  for (int i = 0; i < init_ids.size(); i++) {
    int id = init_ids[i];

    float dist = dis(id);
    retset[i] = Neighbor(id, dist, true);

    if (collect_fullset)
      fullset.push_back(retset[i]);
  }

  std::sort(retset.begin(), retset.begin() + pool_size);

  int k = 0;
  while (k < pool_size) {
    int updated_pos = pool_size;

    if (retset[k].flag) {
      retset[k].flag = false;
      int n = retset[k].id;

      for (int m = 0; m < graph.K; m++) {
        int id = (int)graph.at(n, m);
        if (id == EMPTY_ID || vt.get(id))
          continue;
        vt.set(id);

        float dist = dis(id);
        Neighbor nn(id, dist, true);
        if (collect_fullset)
          fullset.push_back(nn);
        if (dist >= retset[pool_size - 1].distance)
          continue;
        int r = insert_into_pool(retset.data(), pool_size, nn);

        updated_pos = std::min(updated_pos, r);
      }
    }

    k = (updated_pos <= k) ? updated_pos : (k + 1);
  }
}

void NSG::link(Index *storage, const nsg::Graph<idx_t> &knn_graph,
               nsg::Graph<SimpleNeighbor> &graph) {

#pragma omp parallel
  {
    float *vec = new float[storage->d];
    ScopeDeleter<float> del(vec);
    std::vector<Neighbor> pool, tmp;

    VisitedTable vt(ntotal);
    DistanceComputer *dis = storage_distance_computer(storage);
    ScopeDeleter1<DistanceComputer> del1(dis);

#pragma omp for schedule(dynamic, 100)
    for (int i = 0; i < ntotal; i++) {
      storage->reconstruct(i, vec);
      dis->set_query(vec);

      search_on_graph<true, idx_t>(knn_graph, *dis, vt, enterpoint, L, tmp,
                                   pool);

      sync_prune(i, pool, *dis, vt, knn_graph, graph);

      pool.clear();
      tmp.clear();
      vt.advance();
    }
  }

  std::vector<std::mutex> locks(ntotal);
#pragma omp parallel
  {
    DistanceComputer *dis = storage_distance_computer(storage);
    ScopeDeleter1<DistanceComputer> del(dis);
#pragma omp for schedule(dynamic, 100)
    for (int i = 0; i < ntotal; ++i) {
      add_reverse_links(i, locks, *dis, graph);
    }
  }
}

void NSG::sync_prune(int q, std::vector<Neighbor> &pool, DistanceComputer &dis,
                     VisitedTable &vt, const nsg::Graph<idx_t> &knn_graph,
                     nsg::Graph<SimpleNeighbor> &graph) {

  for (int i = 0; i < knn_graph.K; i++) {
    int id = knn_graph.at(q, i);
    if (vt.get(id))
      continue;

    float dist = dis.symmetric_dis(q, id);
    pool.push_back(Neighbor(id, dist, true));
  }

  std::sort(pool.begin(), pool.end());

  std::vector<Neighbor> result;

  int start = 0;
  if (pool[start].id == q)
    start++;
  result.push_back(pool[start]);

  while (result.size() < R && (++start) < pool.size() && start < C) {
    auto &p = pool[start];
    bool occlude = false;
    for (int t = 0; t < result.size(); t++) {
      if (p.id == result[t].id) {
        occlude = true;
        break;
      }
      float djk = dis.symmetric_dis(result[t].id, p.id);
      if (djk < p.distance /* dik */) {
        occlude = true;
        break;
      }
    }
    if (!occlude)
      result.push_back(p);
  }

  for (size_t i = 0; i < R; i++) {
    if (i < result.size()) {
      graph.at(q, i).id = result[i].id;
      graph.at(q, i).distance = result[i].distance;
    } else {
      graph.at(q, i).id = EMPTY_ID;
    }
  }
}

void NSG::add_reverse_links(int q, std::vector<std::mutex> &locks,
                            DistanceComputer &dis,
                            nsg::Graph<SimpleNeighbor> &graph) {

  for (size_t i = 0; i < R; i++) {
    if (graph.at(q, i).id == EMPTY_ID)
      break;

    SimpleNeighbor sn(q, graph.at(q, i).distance);
    int des = graph.at(q, i).id;

    std::vector<SimpleNeighbor> tmp_pool;
    int dup = 0;
    {
      LockGuard guard(locks[des]);
      for (int j = 0; j < R; j++) {
        if (graph.at(des, j).id == EMPTY_ID)
          break;
        if (q == graph.at(des, j).id) {
          dup = 1;
          break;
        }
        tmp_pool.push_back(graph.at(des, j));
      }
    }

    if (dup)
      continue;

    tmp_pool.push_back(sn);
    if (tmp_pool.size() > R) {
      std::vector<SimpleNeighbor> result;
      int start = 0;
      std::sort(tmp_pool.begin(), tmp_pool.end());
      result.push_back(tmp_pool[start]);

      while (result.size() < R && (++start) < tmp_pool.size()) {
        auto &p = tmp_pool[start];
        bool occlude = false;

        for (int t = 0; t < result.size(); t++) {
          if (p.id == result[t].id) {
            occlude = true;
            break;
          }
          float djk = dis.symmetric_dis(result[t].id, p.id);
          if (djk < p.distance /* dik */) {
            occlude = true;
            break;
          }
        }
        if (!occlude)
          result.push_back(p);
      }
      {
        LockGuard guard(locks[des]);
        for (int t = 0; t < result.size(); t++) {
          graph.at(des, t) = result[t];
        }
      }
    } else {
      LockGuard guard(locks[des]);
      for (int t = 0; t < R; t++) {
        if (graph.at(des, t).id == EMPTY_ID) {
          graph.at(des, t) = sn;
          if (t + 1 < R)
            graph.at(des, t + 1).id = EMPTY_ID;
          break;
        }
      }
    }
  }
}

void NSG::tree_grow(Index *storage) {
  int root = enterpoint;
  VisitedTable vt(ntotal);

  int cnt;
  while (cnt < ntotal) {
    dfs(vt, root, cnt);
    if (cnt >= ntotal)
      break;

    find_root(storage, vt, root);
  }
}

void NSG::dfs(VisitedTable &vt, int root, int &cnt) {
  int node = root;
  std::stack<int> s;
  s.push(root);

  cnt = 0;
  if (!vt.get(root))
    cnt++;
  vt.set(root);

  while (!s.empty()) {
    int next = EMPTY_ID;
    for (int i = 0; i < R; i++) {
      int id = final_graph->at(node, i);
      if (id != EMPTY_ID && !vt.get(id)) {
        next = id;
        break;
      }
    }

    if (next == EMPTY_ID) {
      s.pop();
      if (s.empty())
        break;
      node = s.top();
      continue;
    }
    node = next;
    vt.set(node);
    s.push(node);
    cnt++;
  }
}

void NSG::find_root(Index *storage, VisitedTable &vt, int &root) {
  int id = EMPTY_ID;
  for (int i = 0; i < ntotal; i++) {
    if (!vt.get(i)) {
      id = i;
      break;
    }
  }

  if (id == EMPTY_ID)
    return; // No Unlinked Node

  std::vector<Neighbor> tmp, pool;

  DistanceComputer *dis = storage_distance_computer(storage);
  ScopeDeleter1<DistanceComputer> del1(dis);
  float *vec = new float[storage->d];
  ScopeDeleter<float> del(vec);

  storage->reconstruct(id, vec);
  dis->set_query(vec);

  {
    VisitedTable vt2(ntotal);
    search_on_graph<true, int>(*final_graph, *dis, vt2, enterpoint, L, tmp,
                               pool);
  }

  std::sort(pool.begin(), pool.end());

  int found = 0;
  for (int i = 0; i < pool.size(); i++) {
    if (vt.get(pool[i].id)) {
      root = pool[i].id;
      found = 1;
      break;
    }
  }
  if (found == 0) {
    while (true) {
      int rid = rand() % ntotal;
      if (vt.get(rid)) {
        root = rid;
        break;
      }
    }
  }
  final_graph->at(root, R - 1) = id;
}

void NSG::check_graph() {
  for (int i = 0; i < ntotal; i++) {
    for (int j = 0; j < R; j++) {
      int id = final_graph->at(i, j);
      FAISS_THROW_IF_NOT(id < ntotal && (id >= 0 || id == EMPTY_ID));
    }
  }
}

} // namespace faiss
