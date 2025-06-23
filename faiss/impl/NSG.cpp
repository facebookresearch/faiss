/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/NSG.h>

#include <algorithm>
#include <memory>
#include <mutex>
#include <stack>

#include <faiss/impl/DistanceComputer.h>

namespace faiss {

namespace {

using LockGuard = std::lock_guard<std::mutex>;

// It needs to be smaller than 0
constexpr int EMPTY_ID = -1;

} // anonymous namespace

namespace nsg {

DistanceComputer* storage_distance_computer(const Index* storage) {
    if (is_similarity_metric(storage->metric_type)) {
        return new NegativeDistanceComputer(storage->get_distance_computer());
    } else {
        return storage->get_distance_computer();
    }
}

struct Neighbor {
    int32_t id;
    float distance;
    bool flag;

    Neighbor() = default;
    Neighbor(int id, float distance, bool f)
            : id(id), distance(distance), flag(f) {}

    inline bool operator<(const Neighbor& other) const {
        return distance < other.distance;
    }
};

struct Node {
    int32_t id;
    float distance;

    Node() = default;
    Node(int id, float distance) : id(id), distance(distance) {}

    inline bool operator<(const Node& other) const {
        return distance < other.distance;
    }

    // to keep the compiler happy
    inline bool operator<(int other) const {
        return id < other;
    }
};

inline int insert_into_pool(Neighbor* addr, int K, Neighbor nn) {
    // find the location to insert
    int left = 0, right = K - 1;
    if (addr[left].distance > nn.distance) {
        memmove(&addr[left + 1], &addr[left], K * sizeof(Neighbor));
        addr[left] = nn;
        return left;
    }
    if (addr[right].distance < nn.distance) {
        addr[K] = nn;
        return K;
    }
    while (left < right - 1) {
        int mid = (left + right) / 2;
        if (addr[mid].distance > nn.distance) {
            right = mid;
        } else {
            left = mid;
        }
    }
    // check equal ID

    while (left > 0) {
        if (addr[left].distance < nn.distance) {
            break;
        }
        if (addr[left].id == nn.id) {
            return K + 1;
        }
        left--;
    }
    if (addr[left].id == nn.id || addr[right].id == nn.id) {
        return K + 1;
    }
    memmove(&addr[right + 1], &addr[right], (K - right) * sizeof(Neighbor));
    addr[right] = nn;
    return right;
}

} // namespace nsg

using namespace nsg;

NSG::NSG(int R) : R(R), rng(0x0903) {
    L = R + 32;
    C = R + 100;
    srand(0x1998);
}

void NSG::search(
        DistanceComputer& dis,
        int k,
        idx_t* I,
        float* D,
        VisitedTable& vt) const {
    FAISS_THROW_IF_NOT(is_built);
    FAISS_THROW_IF_NOT(final_graph);

    int pool_size = std::max(search_L, k);
    std::vector<Neighbor> retset;
    std::vector<Node> tmp;
    search_on_graph<false>(
            *final_graph, dis, vt, enterpoint, pool_size, retset, tmp);

    for (size_t i = 0; i < k; i++) {
        I[i] = retset[i].id;
        D[i] = retset[i].distance;
    }
}

void NSG::build(
        Index* storage,
        idx_t n,
        const nsg::Graph<idx_t>& knn_graph,
        bool verbose) {
    FAISS_THROW_IF_NOT(!is_built && ntotal == 0);

    if (verbose) {
        printf("NSG::build R=%d, L=%d, C=%d\n", R, L, C);
    }

    ntotal = n;
    init_graph(storage, knn_graph);

    std::vector<int> degrees(n, 0);
    {
        nsg::Graph<Node> tmp_graph(n, R);

        link(storage, knn_graph, tmp_graph, verbose);

        final_graph = std::make_shared<nsg::Graph<int>>(n, R);
        std::fill_n(final_graph->data, n * R, EMPTY_ID);

#pragma omp parallel for
        for (int i = 0; i < n; i++) {
            int cnt = 0;
            for (int j = 0; j < R; j++) {
                int id = tmp_graph.at(i, j).id;
                if (id != EMPTY_ID) {
                    final_graph->at(i, cnt) = id;
                    cnt += 1;
                }
                degrees[i] = cnt;
            }
        }
    }

    int num_attached = tree_grow(storage, degrees);
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
               max,
               min,
               avg);
        printf("Attached nodes: %d\n", num_attached);
    }
}

void NSG::reset() {
    final_graph.reset();
    ntotal = 0;
    is_built = false;
}

void NSG::init_graph(Index* storage, const nsg::Graph<idx_t>& knn_graph) {
    int d = storage->d;
    int n = storage->ntotal;

    std::unique_ptr<float[]> center(new float[d]);
    std::unique_ptr<float[]> tmp(new float[d]);
    std::fill_n(center.get(), d, 0.0f);

    for (int i = 0; i < n; i++) {
        storage->reconstruct(i, tmp.get());
        for (int j = 0; j < d; j++) {
            center[j] += tmp[j];
        }
    }

    for (int i = 0; i < d; i++) {
        center[i] /= n;
    }

    std::vector<Neighbor> retset;
    std::vector<Node> tmpset;

    // random initialize navigating point
    int ep = rng.rand_int(n);
    std::unique_ptr<DistanceComputer> dis(storage_distance_computer(storage));

    dis->set_query(center.get());
    VisitedTable vt(ntotal);

    // Do not collect the visited nodes
    search_on_graph<false>(knn_graph, *dis, vt, ep, L, retset, tmpset);

    // set enterpoint
    enterpoint = retset[0].id;
}

template <bool collect_fullset, class index_t>
void NSG::search_on_graph(
        const nsg::Graph<index_t>& graph,
        DistanceComputer& dis,
        VisitedTable& vt,
        int ep,
        int pool_size,
        std::vector<Neighbor>& retset,
        std::vector<Node>& fullset) const {
    RandomGenerator gen(0x1234);
    retset.resize(pool_size + 1);
    std::vector<int> init_ids(pool_size);

    int num_ids = 0;
    std::vector<index_t> neighbors(graph.K);
    size_t nneigh = graph.get_neighbors(ep, neighbors.data());
    for (int i = 0; i < init_ids.size() && i < nneigh; i++) {
        int id = (int)neighbors[i];
        if (id >= ntotal) {
            continue;
        }

        init_ids[i] = id;
        vt.set(id);
        num_ids += 1;
    }

    while (num_ids < pool_size) {
        int id = gen.rand_int(ntotal);
        if (vt.get(id)) {
            continue;
        }

        init_ids[num_ids] = id;
        num_ids++;
        vt.set(id);
    }

    for (int i = 0; i < init_ids.size(); i++) {
        int id = init_ids[i];

        float dist = dis(id);
        retset[i] = Neighbor(id, dist, true);

        if (collect_fullset) {
            fullset.emplace_back(retset[i].id, retset[i].distance);
        }
    }

    std::sort(retset.begin(), retset.begin() + pool_size);

    int k = 0;
    while (k < pool_size) {
        int updated_pos = pool_size;

        if (retset[k].flag) {
            retset[k].flag = false;
            int n = retset[k].id;

            size_t nneigh_for_n = graph.get_neighbors(n, neighbors.data());
            for (int m = 0; m < nneigh_for_n; m++) {
                int id = neighbors[m];
                if (id > ntotal || vt.get(id)) {
                    continue;
                }
                vt.set(id);

                float dist = dis(id);
                Neighbor nn(id, dist, true);
                if (collect_fullset) {
                    fullset.emplace_back(id, dist);
                }

                if (dist >= retset[pool_size - 1].distance) {
                    continue;
                }

                int r = insert_into_pool(retset.data(), pool_size, nn);

                updated_pos = std::min(updated_pos, r);
            }
        }

        k = (updated_pos <= k) ? updated_pos : (k + 1);
    }
}

void NSG::link(
        Index* storage,
        const nsg::Graph<idx_t>& knn_graph,
        nsg::Graph<Node>& graph,
        bool /* verbose */) {
#pragma omp parallel
    {
        std::unique_ptr<float[]> vec(new float[storage->d]);

        std::vector<Node> pool;
        std::vector<Neighbor> tmp;

        VisitedTable vt(ntotal);
        std::unique_ptr<DistanceComputer> dis(
                storage_distance_computer(storage));

#pragma omp for schedule(dynamic, 100)
        for (int i = 0; i < ntotal; i++) {
            storage->reconstruct(i, vec.get());
            dis->set_query(vec.get());

            // Collect the visited nodes into pool
            search_on_graph<true>(
                    knn_graph, *dis, vt, enterpoint, L, tmp, pool);

            sync_prune(i, pool, *dis, vt, knn_graph, graph);

            pool.clear();
            tmp.clear();
            vt.advance();
        }
    } // omp parallel

    std::vector<std::mutex> locks(ntotal);
#pragma omp parallel
    {
        std::unique_ptr<DistanceComputer> dis(
                storage_distance_computer(storage));

#pragma omp for schedule(dynamic, 100)
        for (int i = 0; i < ntotal; ++i) {
            add_reverse_links(i, locks, *dis, graph);
        }
    } // omp parallel
}

void NSG::sync_prune(
        int q,
        std::vector<Node>& pool,
        DistanceComputer& dis,
        VisitedTable& vt,
        const nsg::Graph<idx_t>& knn_graph,
        nsg::Graph<Node>& graph) {
    for (int i = 0; i < knn_graph.K; i++) {
        int id = knn_graph.at(q, i);
        if (id < 0 || id >= ntotal || vt.get(id)) {
            continue;
        }

        float dist = dis.symmetric_dis(q, id);
        pool.emplace_back(id, dist);
    }

    std::sort(pool.begin(), pool.end());

    std::vector<Node> result;

    int start = 0;
    if (pool[start].id == q) {
        start++;
    }
    result.push_back(pool[start]);

    while (result.size() < R && (++start) < pool.size() && start < C) {
        auto& p = pool[start];
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
        if (!occlude) {
            result.push_back(p);
        }
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

void NSG::add_reverse_links(
        int q,
        std::vector<std::mutex>& locks,
        DistanceComputer& dis,
        nsg::Graph<Node>& graph) {
    for (size_t i = 0; i < R; i++) {
        if (graph.at(q, i).id == EMPTY_ID) {
            break;
        }

        Node sn(q, graph.at(q, i).distance);
        int des = graph.at(q, i).id;

        std::vector<Node> tmp_pool;
        int dup = 0;
        {
            LockGuard guard(locks[des]);
            for (int j = 0; j < R; j++) {
                if (graph.at(des, j).id == EMPTY_ID) {
                    break;
                }
                if (q == graph.at(des, j).id) {
                    dup = 1;
                    break;
                }
                tmp_pool.push_back(graph.at(des, j));
            }
        }

        if (dup) {
            continue;
        }

        tmp_pool.push_back(sn);
        if (tmp_pool.size() > R) {
            std::vector<Node> result;
            int start = 0;
            std::sort(tmp_pool.begin(), tmp_pool.end());
            result.push_back(tmp_pool[start]);

            while (result.size() < R && (++start) < tmp_pool.size()) {
                auto& p = tmp_pool[start];
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

                if (!occlude) {
                    result.push_back(p);
                }
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
                    break;
                }
            }
        }
    }
}

int NSG::tree_grow(Index* storage, std::vector<int>& degrees) {
    int root = enterpoint;
    VisitedTable vt(ntotal);
    VisitedTable vt2(ntotal);

    int num_attached = 0;
    int cnt = 0;
    while (true) {
        cnt = dfs(vt, root, cnt);
        if (cnt >= ntotal) {
            break;
        }

        root = attach_unlinked(storage, vt, vt2, degrees);
        vt2.advance();
        num_attached += 1;
    }

    return num_attached;
}

int NSG::dfs(VisitedTable& vt, int root, int cnt) const {
    int node = root;
    std::stack<int> stack;
    stack.push(root);

    if (!vt.get(root)) {
        cnt++;
    }
    vt.set(root);

    while (!stack.empty()) {
        int next = EMPTY_ID;
        for (int i = 0; i < R; i++) {
            int id = final_graph->at(node, i);
            if (id != EMPTY_ID && !vt.get(id)) {
                next = id;
                break;
            }
        }

        if (next == EMPTY_ID) {
            stack.pop();
            if (stack.empty()) {
                break;
            }
            node = stack.top();
            continue;
        }
        node = next;
        vt.set(node);
        stack.push(node);
        cnt++;
    }

    return cnt;
}

int NSG::attach_unlinked(
        Index* storage,
        VisitedTable& vt,
        VisitedTable& vt2,
        std::vector<int>& degrees) {
    /* NOTE: This implementation is slightly different from the original paper.
     *
     * Instead of connecting the unlinked node to the nearest point in the
     * spanning tree which will increase the maximum degree of the graph and
     * also make the graph hard to maintain, this implementation links the
     * unlinked node to the nearest node of which the degree is smaller than R.
     * It will keep the degree of all nodes to be no more than `R`.
     */

    // find one unlinked node
    int id = EMPTY_ID;
    for (int i = 0; i < ntotal; i++) {
        if (!vt.get(i)) {
            id = i;
            break;
        }
    }

    if (id == EMPTY_ID) {
        return EMPTY_ID; // No Unlinked Node
    }

    std::vector<Neighbor> tmp;
    std::vector<Node> pool;

    std::unique_ptr<DistanceComputer> dis(storage_distance_computer(storage));
    std::unique_ptr<float[]> vec(new float[storage->d]);

    storage->reconstruct(id, vec.get());
    dis->set_query(vec.get());

    // Collect the visited nodes into pool
    search_on_graph<true>(
            *final_graph, *dis, vt2, enterpoint, search_L, tmp, pool);

    std::sort(pool.begin(), pool.end());

    int node;
    bool found = false;
    for (int i = 0; i < pool.size(); i++) {
        node = pool[i].id;
        if (degrees[node] < R && node != id) {
            found = true;
            break;
        }
    }

    // randomly choice annother node
    if (!found) {
        do {
            node = rng.rand_int(ntotal);
            if (vt.get(node) && degrees[node] < R && node != id) {
                found = true;
            }
        } while (!found);
    }

    int pos = degrees[node];
    final_graph->at(node, pos) = id; // replace
    degrees[node] += 1;

    return node;
}

void NSG::check_graph() const {
#pragma omp parallel for
    for (int i = 0; i < ntotal; i++) {
        for (int j = 0; j < R; j++) {
            int id = final_graph->at(i, j);
            FAISS_THROW_IF_NOT(id < ntotal && (id >= 0 || id == EMPTY_ID));
        }
    }
}

} // namespace faiss
