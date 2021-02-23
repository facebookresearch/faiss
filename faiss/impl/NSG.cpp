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


/**************************************************************
 * Auxiliary structures
 **************************************************************/

/// set implementation optimized for fast access.
struct VisitedTable {
  std::vector<uint8_t> visited;
  int visno;

  explicit VisitedTable(int size) : visited(size), visno(1) {}

  /// set flog #no to true
  void set(int no) { visited[no] = visno; }

  /// get flag #no
  bool get(int no) const { return visited[no] == visno; }

  /// reset all flags to false
  void advance() {
    visno++;
    if (visno == 250) {
      // 250 rather than 255 because sometimes we use visno and visno+1
      memset(visited.data(), 0, sizeof(visited[0]) * visited.size());
      visno = 1;
    }
  }
};

/* Wrap the distance computer into one that negates the
   distances. This makes supporting INNER_PRODUCE search easier */

struct NegativeDistanceComputer : DistanceComputer {

  /// owned by this
  DistanceComputer *basedis;

  explicit NegativeDistanceComputer(DistanceComputer *basedis)
      : basedis(basedis) {}

  void set_query(const float *x) override { basedis->set_query(x); }

  /// compute distance of vector i to current query
  float operator()(idx_t i) override { return -(*basedis)(i); }

  /// compute distance between two stored vectors
  float symmetric_dis(idx_t i, idx_t j) override {
    return -basedis->symmetric_dis(i, j);
  }

  virtual ~NegativeDistanceComputer() { delete basedis; }
};

DistanceComputer *storage_distance_computer(const Index *storage) {
  if (storage->metric_type == METRIC_INNER_PRODUCT) {
    return new NegativeDistanceComputer(storage->get_distance_computer());
  } else {
    return storage->get_distance_computer();
  }
}


// It needs to be smaller than 0
const int EMPTY_ID = -1;

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
  is_built = false;
}

void NSG::search(DistanceComputer &dis, int k, idx_t *I, float *D,
                 VisitedTable &vt) const {

  FAISS_THROW_IF_NOT(is_built);
  int pool_size = std::max(search_L, k);

  std::vector<Neighbor> retset, tmp;
  search_on_graph<false>(compact_graph.data(), width, dis, vt,
                         enter_point, pool_size, retset, tmp);
  std::partial_sort(retset.begin(), retset.begin() + k,
                    retset.begin() + pool_size);

  for (size_t i = 0; i < k; i++) {
    I[i] = retset[i].id;
    D[i] = retset[i].distance;
  }
}

void NSG::build(Index *storage, idx_t n, int *knn_graph, int GK) {
  printf("R=%d, L=%d, C=%d\n", R, L, C);

  ntotal = n;
  init_graph(storage, knn_graph, GK);
  SimpleNeighbor *graph = new SimpleNeighbor[n * R];
  link(storage, knn_graph, GK, graph);

  final_graph.resize(n);
  for (int i = 0; i < n; i++) {
    final_graph[i].reserve(R);
    for (int j = 0; j < R; j++) {
      final_graph[i].push_back(graph[i * R + j].id);
    }
  }
  delete[] graph;

  // tree_grow(storage);
  is_built = true;

  int max = 0, min = 1e6;
  double avg = 0;
  for (int i = 0; i < n; i++) {
    int size = 0;
    while (size < final_graph[i].size() && final_graph[i][size] >= 0) {
      size += 1;
    }
    max = std::max(size, max);
    min = std::min(size, min);
    avg += size;
  }
  avg = avg / n;
  printf("Degree Statistics: Max = %d, Min = %d, Avg = %lf\n", max, min, avg);

  width = max;
  compact(width);
}

void NSG::reset() {
  compact_graph.clear();
  ntotal = 0;
  is_built = false;
}

void NSG::init_graph(Index *storage, int *knn_graph, int GK) {
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
  search_on_graph<false>(knn_graph, GK, *dis, vt, ep, L, retset, pool);

  enter_point = retset[0].id;
}

template <bool collect_fullset>
void NSG::search_on_graph(const int *graph, int GK, DistanceComputer &dis,
                          VisitedTable &vt, int ep, int pool_size,
                          std::vector<Neighbor> &retset,
                          std::vector<Neighbor> &fullset) const {
  retset.resize(pool_size + 1);
  std::vector<int> init_ids(pool_size);

  int num_ids = 0;
  for (int i = 0; i < init_ids.size() && i < GK; i++) {
    int id = graph[ep * GK + i];
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

      for (int m = 0; m < GK; m++) {
        int id = graph[n * GK + m];
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

void NSG::link(Index *storage, int *knn_graph, int GK, SimpleNeighbor *graph) {

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

      search_on_graph<true>(knn_graph, GK, *dis, vt, enter_point, L, tmp, pool);

      sync_prune(i, pool, *dis, vt, knn_graph, GK, graph);

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
                     VisitedTable &vt, const int *knn_graph, int GK,
                     SimpleNeighbor *graph) {

  for (int i = 0; i < GK; i++) {
    int id = knn_graph[q * GK + i];
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

  SimpleNeighbor *neighbors = graph + q * R;
  for (size_t i = 0; i < R; i++) {
    if (i < result.size()) {
      neighbors[i].id = result[i].id;
      neighbors[i].distance = result[i].distance;
    } else {
      neighbors[i].id = EMPTY_ID;
    }
  }
}

void NSG::add_reverse_links(int q, std::vector<std::mutex> &locks,
                            DistanceComputer &dis,
                            SimpleNeighbor *graph) {
SimpleNeighbor *src_pool = graph + q * R;

for (size_t i = 0; i < R; i++) {
  if (src_pool[i].id == EMPTY_ID)
    break;

  SimpleNeighbor sn(q, src_pool[i].distance);
  int des = src_pool[i].id;
  SimpleNeighbor *des_pool = graph + des * R;

  std::vector<SimpleNeighbor> tmp_pool;
  int dup = 0;
  {
    LockGuard guard(locks[des]);
    for (int j = 0; j < R; j++) {
      if (des_pool[j].id == EMPTY_ID)
        break;
      if (q == des_pool[j].id) {
        dup = 1;
        break;
      }
      tmp_pool.push_back(des_pool[j]);
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
        des_pool[t] = result[t];
      }
    }
  } else {
    LockGuard guard(locks[des]);
    for (int t = 0; t < R; t++) {
      if (des_pool[t].id == EMPTY_ID) {
        des_pool[t] = sn;
        if (t + 1 < R)
          des_pool[t + 1].id = EMPTY_ID;
        break;
      }
    }
  }
}

}

void NSG::tree_grow(Index *storage) {
  int root = enter_point;
  VisitedTable vt(ntotal);

  int cnt = 0;
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

  if (!vt.get(root))
    cnt++;
  vt.set(root);

  while (!s.empty()) {
    int next = EMPTY_ID;
    for (const int &id : final_graph[node]) {
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
  // int id = EMPTY_ID;
  // for (int i = 0; i < ntotal; i++) {
  //   if (!vt.get(i)) {
  //     id = i;
  //     break;
  //   }
  // }

  // if (id == EMPTY_ID)
  //   return; // No Unlinked Node

  // std::vector<Neighbor> tmp, pool;

  // DistanceComputer *dis = storage_distance_computer(storage);
  // ScopeDeleter1<DistanceComputer> del(dis);
  // float *vec = new float[storage->d];
  // ScopeDeleter<float> del(vec);

  // storage->reconstruct(id, vec);
  // dis->set_query(vec);

  // {
  //   VisitedTable vt2(ntotal);
  //   search_on_graph<true>(final_graph.data(), R, *dis, vt2, enter_point, L, tmp, pool);
  // }

  // std::sort(pool.begin(), pool.end());

  // int found = 0;
  // for (int i = 0; i < pool.size(); i++) {
  //   if (vt.get(pool[i].id)) {
  //     root = pool[i].id;
  //     found = 1;
  //     break;
  //   }
  // }
  // if (found == 0) {
  //   while (true) {
  //     int rid = rand() % ntotal;
  //     if (vt.get(rid)) {
  //       root = rid;
  //       break;
  //     }
  //   }
  // }
  // final_graph[root].push_back(id);
}

void NSG::compact(int width) {
  compact_graph.resize(ntotal * width);
  for (int i = 0; i < ntotal; i++) {
    for (int j = 0; j < width; j++) {
      if (j < final_graph[i].size()) {
        int id = final_graph[i][j];
        compact_graph[i * width + j] = id;
        FAISS_THROW_IF_NOT(id < ntotal && (id >= 0 || id == EMPTY_ID));
      } else {
        compact_graph[i * width + j] = EMPTY_ID;
      }
    }
    final_graph[i].clear();
  }
  final_graph.clear();
}

} // namespace faiss
