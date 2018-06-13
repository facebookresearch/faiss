/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "IndexHNSW.h"


#include <cstdlib>
#include <cassert>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <omp.h>

#include <unordered_set>
#include <queue>

#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <stdint.h>

#include <immintrin.h>

#include "utils.h"
#include "Heap.h"
#include "FaissAssert.h"
#include "IndexFlat.h"
#include "IndexIVFPQ.h"


extern "C" {

/* declare BLAS functions, see http://www.netlib.org/clapack/cblas/ */

int sgemm_ (const char *transa, const char *transb, FINTEGER *m, FINTEGER *
            n, FINTEGER *k, const float *alpha, const float *a,
            FINTEGER *lda, const float *b, FINTEGER *
            ldb, float *beta, float *c, FINTEGER *ldc);

}

namespace faiss {

/**************************************************************
 * Auxiliary structures
 **************************************************************/

/// set implementation optimized for fast access.
struct VisitedTable {
    std::vector<uint8_t> visited;
    int visno;

    VisitedTable(int size):
        visited(size), visno(1)
    {}
    /// set flog #no to true
    void set(int no) {
        visited[no] = visno;
    }
    /// get flag #no
    bool get(int no) const {
        return visited[no] == visno;
    }
    /// reset all flags to false
    void advance() {
        visno++;
        if (visno == 250) {
            // 250 rather than 255 because sometimes we use visno and visno+1
            memset (visited.data(), 0, sizeof(visited[0]) * visited.size());
            visno = 1;
        }
    }
};


namespace {

typedef HNSW::idx_t idx_t;
typedef HNSW::storage_idx_t storage_idx_t;
typedef HNSW::DistanceComputer DistanceComputer;
    // typedef ::faiss::VisitedTable VisitedTable;

/// to sort pairs of (id, distance) from nearest to fathest or the reverse
struct NodeDistCloser {
    float d;
    int id;
    NodeDistCloser(float d, int id): d(d), id(id) {}
    bool operator<(const NodeDistCloser &obj1) const { return d < obj1.d; }
};

struct NodeDistFarther {
    float d;
    int id;
    NodeDistFarther(float d, int id): d(d), id(id) {}
    bool operator<(const NodeDistFarther &obj1) const { return d > obj1.d; }
};


/** Heap structure that allows fast */

struct MinimaxHeap {
    int n;
    int k;
    int nvalid;

    std::vector<storage_idx_t> ids;
    std::vector<float> dis;
    typedef faiss::CMax<float, storage_idx_t> HC;

    explicit MinimaxHeap(int n): n(n), k(0), nvalid(0), ids(n), dis(n) {}

    void push(storage_idx_t i, float v)
    {
        if (k == n) {
            if (v >= dis[0]) return;
            faiss::heap_pop<HC> (k--, dis.data(), ids.data());
            nvalid--;
        }
        faiss::heap_push<HC> (++k, dis.data(), ids.data(), v, i);
        nvalid++;
    }

    float max() const
    {
        return dis[0];
    }

    int size() const {return nvalid;}

    void clear() {nvalid = k = 0; }

    int pop_min(float *vmin_out = nullptr)
    {
        assert(k > 0);
        // returns min. This is an O(n) operation
        int i = k - 1;
        while (i >= 0) {
            if (ids[i] != -1) break;
            i--;
        }
        if (i == -1) return -1;
        int imin = i;
        float vmin = dis[i];
        i--;
        while(i >= 0) {
            if (ids[i] != -1 && dis[i] < vmin) {
                vmin = dis[i];
                imin = i;
            }
            i--;
        }
        if (vmin_out) *vmin_out = vmin;
        int ret = ids[imin];
        ids[imin] = -1;
        nvalid --;
        return ret;
    }

    int count_below(float thresh) {
        float n_below = 0;
        for(int i = 0; i < k; i++) {
            if (dis[i] < thresh)
                n_below++;
        }
        return n_below;
    }

};


/**************************************************************
 * Addition subroutines
 **************************************************************/

/** Enumerate vertices from farthest to nearest from query, keep a
 * neighbor only if there is no previous neighbor that is closer to
 * that vertex than the query.
 */
void shrink_neighbor_list(DistanceComputer & qdis,
                          std::priority_queue<NodeDistFarther> &input,
                          std::vector<NodeDistFarther> &output,
                          int max_size)
{
    while (input.size() > 0) {
        NodeDistFarther v1 = input.top();
        input.pop();
        float dist_v1_q = v1.d;

        bool good = true;
        for (NodeDistFarther v2 : output) {
            float dist_v1_v2 = qdis.symmetric_dis(v2.id, v1.id);

            if (dist_v1_v2 < dist_v1_q) {
                good = false;
                break;
            }
        }
        if (good) {
            output.push_back(v1);
            if (output.size() >= max_size)
                return;
        }
    }
}




/// remove neighbors from the list to make it smaller than max_size
void shrink_neighbor_list(DistanceComputer & qdis,
                          std::priority_queue<NodeDistCloser> &resultSet1,
                          int max_size)
{

    if (resultSet1.size() < max_size) {
        return;
    }
    std::priority_queue<NodeDistFarther> resultSet;
    std::vector<NodeDistFarther> returnlist;

    while (resultSet1.size() > 0) {
        resultSet.emplace(resultSet1.top().d, resultSet1.top().id);
        resultSet1.pop();
    }

    shrink_neighbor_list (qdis, resultSet, returnlist, max_size);

    for (NodeDistFarther curen2 : returnlist) {
        resultSet1.emplace(curen2.d, curen2.id);
    }

}


/// add a link between two elements, possibly shrinking the list
/// of links to make room for it.
void add_link(HNSW & hnsw,
              DistanceComputer & qdis,
              storage_idx_t src, storage_idx_t dest,
              int level)
{
    size_t begin, end;
    hnsw.neighbor_range(src, level, &begin, &end);
    if (hnsw.neighbors[end - 1] == -1) {
        // there is enough room, find a slot to add it
        size_t i = end;
        while(i > begin) {
            if (hnsw.neighbors[i - 1] != -1) break;
            i--;
        }
        hnsw.neighbors[i] = dest;
        return;
    }

    // otherwise we let them fight out which to keep

    // copy to resultSet...
    std::priority_queue<NodeDistCloser> resultSet;
    resultSet.emplace(qdis.symmetric_dis(src, dest), dest);
    for (size_t i = begin; i < end; i++) { // HERE WAS THE BUG
        storage_idx_t neigh = hnsw.neighbors[i];
        resultSet.emplace(qdis.symmetric_dis(src, neigh), neigh);
    }

    shrink_neighbor_list(qdis, resultSet, end - begin);

    // ...and back
    size_t i = begin;
    while (resultSet.size()) {
        hnsw.neighbors[i++] = resultSet.top().id;
        resultSet.pop();
    }
    // they may have shrunk more than just by 1 element
    while(i < end) {
        hnsw.neighbors[i++] = -1;
    }
}

/// search neighbors on a single level, starting from an entry point
void search_neighbors_to_add(HNSW & hnsw,
                       DistanceComputer &qdis,
                       std::priority_queue<NodeDistCloser> &results,
                       int entry_point,
                       float d_entry_point,
                       int level,
                       VisitedTable &vt)
{
    // top is nearest candidate
    std::priority_queue<NodeDistFarther> candidates;

    NodeDistFarther ev(d_entry_point, entry_point);
    candidates.push(ev);
    results.emplace(d_entry_point, entry_point);
    vt.set(entry_point);

    while (!candidates.empty()) {
        // get nearest
        const NodeDistFarther &currEv = candidates.top();

        if (currEv.d > results.top().d) {
            break;
        }
        int currNode = currEv.id;
        candidates.pop();

        // loop over neighbors
        size_t begin, end;
        hnsw.neighbor_range(currNode, level, &begin, &end);
        for(size_t i = begin; i < end; i++) {
            storage_idx_t nodeId = hnsw.neighbors[i];
            if (nodeId < 0) break;
            if (vt.get(nodeId)) continue;
            vt.set(nodeId);

            float dis = qdis(nodeId);
            NodeDistFarther evE1(dis, nodeId);

            if (results.size() < hnsw.efConstruction ||
                results.top().d > dis) {

                results.emplace(dis, nodeId);
                candidates.emplace(dis, nodeId);
                if (results.size() > hnsw.efConstruction) {
                    results.pop();
                }
            }
        }
    }
    vt.advance();
}


/// Finds neighbors and builds links with them, starting from an entry
/// point. The own neighbor list is assumed to be locked.
void add_links_starting_from(HNSW & hnsw,
                             DistanceComputer &ptdis,
                             storage_idx_t pt_id,
                             storage_idx_t nearest,
                             float d_nearest,
                             int level,
                             omp_lock_t * locks,
                             VisitedTable &vt)
{

    std::priority_queue<NodeDistCloser> link_targets;

    search_neighbors_to_add(
            hnsw, ptdis, link_targets, nearest, d_nearest,
            level, vt);

    // but we can afford only this many neighbors
    int M = hnsw.nb_neighbors(level);

    shrink_neighbor_list(ptdis, link_targets, M);

    while (!link_targets.empty()) {
        int other_id = link_targets.top().id;

        omp_set_lock(&locks[other_id]);
        add_link(hnsw, ptdis, other_id, pt_id, level);
        omp_unset_lock(&locks[other_id]);

        add_link(hnsw, ptdis, pt_id, other_id, level);

        link_targets.pop();
    }
}

/**************************************************************
 * Searching subroutines
 **************************************************************/

/// greedily update a nearest vector at a given level
void greedy_update_nearest(const HNSW & hnsw,
                           DistanceComputer & qdis,
                           int level,
                           storage_idx_t & nearest,
                           float & d_nearest)
{
    for(;;) {
        storage_idx_t prev_nearest = nearest;

        size_t begin, end;
        hnsw.neighbor_range(nearest, level, &begin, &end);
        for(size_t i = begin; i < end; i++) {
            storage_idx_t v = hnsw.neighbors[i];
            if (v < 0) break;
            float dis = qdis(v);
            if (dis < d_nearest) {
                nearest = v;
                d_nearest = dis;
            }
        }
        if (nearest == prev_nearest) {
            return;
        }
    }

}


/** Do a BFS on the candidates list */

int search_from_candidates(const HNSW & hnsw,
                           DistanceComputer & qdis, int k,
                           idx_t *I, float * D,
                           MinimaxHeap &candidates,
                           VisitedTable &vt,
                           int level, int nres_in = 0)
{
    int nres = nres_in;
    int ndis = 0;
    for (int i = 0; i < candidates.size(); i++) {
        idx_t v1 = candidates.ids[i];
        float d = candidates.dis[i];
        FAISS_ASSERT(v1 >= 0);
        if (nres < k) {
            faiss::maxheap_push (++nres, D, I, d, v1);
        } else if (d < D[0]) {
            faiss::maxheap_pop (nres--, D, I);
            faiss::maxheap_push (++nres, D, I, d, v1);
        }
        vt.set(v1);
    }

    bool do_dis_check = hnsw.check_relative_distance;
    int nstep = 0;

    while (candidates.size() > 0) {
        float d0 = 0;
        int v0 = candidates.pop_min(&d0);

        if (do_dis_check) {
            // tricky stopping condition: there are more that ef
            // distances that are processed already that are smaller
            // than d0

            int n_dis_below = candidates.count_below(d0);
            if(n_dis_below >= hnsw.efSearch) {
                break;
            }
        }
        size_t begin, end;
        hnsw.neighbor_range(v0, level, &begin, &end);

        for (size_t j = begin; j < end; j++) {
            int v1 = hnsw.neighbors[j];
            if (v1 < 0) break;
            if (vt.get(v1)) {
                continue;
            }
            vt.set(v1);
            ndis++;
            float d = qdis(v1);
            if (nres < k) {
                faiss::maxheap_push (++nres, D, I, d, v1);
            } else if (d < D[0]) {
                faiss::maxheap_pop (nres--, D, I);
                faiss::maxheap_push (++nres, D, I, d, v1);
            }
            candidates.push(v1, d);
        }

        nstep++;
        if (!do_dis_check && nstep > hnsw.efSearch) {
            break;
        }
    }

    if (level == 0) {
#pragma omp critical
        {
            hnsw_stats.n1 ++;
            if (candidates.size() == 0)
                hnsw_stats.n2 ++;
            hnsw_stats.n3 += ndis;
        }
    }

    return nres;
}


} // anonymous namespace


/**************************************************************
 * HNSW structure implementation
 **************************************************************/


int HNSW::nb_neighbors(int layer_no) const
{
    return cum_nneighbor_per_level[layer_no + 1] -
        cum_nneighbor_per_level[layer_no];
}

void HNSW::set_nb_neighbors(int level_no, int n)
{
    FAISS_THROW_IF_NOT(levels.size() == 0);
    int cur_n = nb_neighbors(level_no);
    for (int i = level_no + 1; i < cum_nneighbor_per_level.size(); i++) {
        cum_nneighbor_per_level[i] += n - cur_n;
    }
}

int HNSW::cum_nb_neighbors(int layer_no) const
{
    return cum_nneighbor_per_level[layer_no];
}

void HNSW::neighbor_range(idx_t no, int layer_no,
                        size_t * begin, size_t * end) const
{
    size_t o = offsets[no];
    *begin = o + cum_nb_neighbors(layer_no);
    *end = o + cum_nb_neighbors(layer_no + 1);
}



HNSW::HNSW(int M): rng(12345) {
    set_default_probas(M, 1.0 / log(M));
    max_level = -1;
    entry_point = -1;
    efSearch = 16;
    check_relative_distance = true;
    efConstruction = 40;
    upper_beam = 1;
    offsets.push_back(0);
}


int HNSW::random_level()
{
    double f = rng.rand_float();
    // could be a bit faster with bissection
    for (int level = 0; level < assign_probas.size(); level++) {
        if (f < assign_probas[level]) {
            return level;
        }
        f -= assign_probas[level];
    }
    // happens with exponentially low probability
    return assign_probas.size() - 1;
}

void HNSW::set_default_probas(int M, float levelMult)
{
    int nn = 0;
    cum_nneighbor_per_level.push_back (0);
    for (int level = 0; ;level++) {
        float proba = exp(-level / levelMult) * (1 - exp(-1 / levelMult));
        if (proba < 1e-9) break;
        assign_probas.push_back(proba);
        nn += level == 0 ? M * 2 : M;
        cum_nneighbor_per_level.push_back (nn);
    }
}

void HNSW::clear_neighbor_tables(int level)
{
    for (int i = 0; i < levels.size(); i++) {
        size_t begin, end;
        neighbor_range(i, level, &begin, &end);
        for (size_t j = begin; j < end; j++)
            neighbors[j] = -1;
    }
}


void HNSW::reset() {
    max_level = -1;
    entry_point = -1;
    offsets.clear();
    offsets.push_back(0);
    levels.clear();
    neighbors.clear();
}



void HNSW::print_neighbor_stats(int level) const
{
    FAISS_THROW_IF_NOT (level < cum_nneighbor_per_level.size());
    printf("stats on level %d, max %d neighbors per vertex:\n",
           level, nb_neighbors(level));
    size_t tot_neigh = 0, tot_common = 0, tot_reciprocal = 0, n_node = 0;
#pragma omp parallel for reduction(+: tot_neigh) reduction(+: tot_common) \
                         reduction(+: tot_reciprocal) reduction(+: n_node)
    for (int i = 0; i < levels.size(); i++) {
        if (levels[i] > level) {
            n_node++;
            size_t begin, end;
            neighbor_range(i, level, &begin, &end);
            std::unordered_set<int> neighset;
            for (size_t j = begin; j < end; j++) {
                if (neighbors [j] < 0) break;
                neighset.insert(neighbors[j]);
            }
            int n_neigh = neighset.size();
            int n_common = 0;
            int n_reciprocal = 0;
            for (size_t j = begin; j < end; j++) {
                storage_idx_t i2 = neighbors[j];
                if (i2 < 0) break;
                FAISS_ASSERT(i2 != i);
                size_t begin2, end2;
                neighbor_range(i2, level, &begin2, &end2);
                for (size_t j2 = begin2; j2 < end2; j2++) {
                    storage_idx_t i3 = neighbors[j2];
                    if (i3 < 0) break;
                    if (i3 == i) {
                        n_reciprocal++;
                        continue;
                    }
                    if (neighset.count(i3)) {
                        neighset.erase(i3);
                        n_common++;
                    }
                }
            }
            tot_neigh += n_neigh;
            tot_common += n_common;
            tot_reciprocal += n_reciprocal;
        }
    }
    float normalizer = n_node;
    printf("   nb of nodes at that level %ld\n", n_node);
    printf("   neighbors per node: %.2f (%ld)\n", tot_neigh / normalizer, tot_neigh);
    printf("   nb of reciprocal neighbors: %.2f\n", tot_reciprocal / normalizer);
    printf("   nb of neighbors that are also neighbor-of-neighbors: %.2f (%ld)\n",
           tot_common / normalizer, tot_common);



}

HNSWStats hnsw_stats;

void HNSWStats::reset ()
{
    memset(this, 0, sizeof(*this));
}



/**************************************************************
 * Building, parallel
 **************************************************************/

void HNSW::add_with_locks(
      DistanceComputer & ptdis, int pt_level, int pt_id,
      std::vector<omp_lock_t> & locks,
      VisitedTable &vt)
{
    //  greedy search on upper levels

    storage_idx_t nearest;
#pragma omp critical
    {
        nearest = entry_point;

        if (nearest == -1) {
            max_level = pt_level;
            entry_point = pt_id;
        }
    }

    if (nearest < 0) {
        return;
    }

    omp_set_lock(&locks[pt_id]);

    int level = max_level; // level at which we start adding neighbors
    float d_nearest = ptdis(nearest);

    for(; level > pt_level; level--) {
        greedy_update_nearest(*this, ptdis, level, nearest, d_nearest);
    }

    for(; level >= 0; level--) {
        add_links_starting_from(*this, ptdis, pt_id, nearest, d_nearest,
                                level, locks.data(), vt);
    }

    omp_unset_lock(&locks[pt_id]);

    if (pt_level > max_level) {
        max_level = pt_level;
        entry_point = pt_id;
    }

}



/**************************************************************
 * Searching
 **************************************************************/




void HNSW::search(DistanceComputer & qdis,
                  int k, idx_t *I, float * D,
                  VisitedTable &vt) const
{

    if (upper_beam == 1) {

        //  greedy search on upper levels
        storage_idx_t nearest = entry_point;
        float d_nearest = qdis(nearest);

        for(int level = max_level; level >= 1; level--) {
            greedy_update_nearest(*this, qdis, level, nearest, d_nearest);
        }

        int candidates_size = std::max(efSearch, k);
        MinimaxHeap candidates(candidates_size);

        candidates.push(nearest, d_nearest);

        search_from_candidates (
                  *this, qdis, k, I, D, candidates, vt, 0);
        vt.advance();

    } else {

        int candidates_size = upper_beam;
        MinimaxHeap candidates(candidates_size);

        std::vector<idx_t> I_to_next(candidates_size);
        std::vector<float> D_to_next(candidates_size);

        int nres = 1;
        I_to_next[0] = entry_point;
        D_to_next[0] = qdis(entry_point);

        for(int level = max_level; level >= 0; level--) {

            // copy I, D -> candidates

            candidates.clear();

            for (int i = 0; i < nres; i++) {
                candidates.push(I_to_next[i], D_to_next[i]);
            }

            if (level == 0) {
                nres = search_from_candidates (
                   *this, qdis, k, I, D, candidates, vt, 0);
            } else  {
                nres = search_from_candidates (
                   *this, qdis, candidates_size,
                   I_to_next.data(), D_to_next.data(),
                   candidates, vt, level);
            }
            vt.advance();
        }

    }
}

/**************************************************************
 * add / search blocks of descriptors
 **************************************************************/

namespace {

int prepare_level_tab (HNSW & hnsw, size_t n, bool preset_levels = false)
{
    size_t n0 = hnsw.offsets.size() - 1;

    if (preset_levels) {
        FAISS_ASSERT (n0 + n == hnsw.levels.size());
    } else {
        FAISS_ASSERT (n0 == hnsw.levels.size());
        for (int i = 0; i < n; i++) {
            int pt_level = hnsw.random_level();
            hnsw.levels.push_back(pt_level + 1);
        }
    }

    int max_level = 0;
    for (int i = 0; i < n; i++) {
        int pt_level = hnsw.levels[i + n0] - 1;
        if (pt_level > max_level) max_level = pt_level;
        hnsw.offsets.push_back(hnsw.offsets.back() +
                               hnsw.cum_nb_neighbors(pt_level + 1));
        hnsw.neighbors.resize(hnsw.offsets.back(), -1);
    }
    return max_level;
}

void hnsw_add_vertices(IndexHNSW &index_hnsw,
                       size_t n0,
                       size_t n, const float *x,
                       bool verbose,
                       bool preset_levels = false) {
    HNSW & hnsw = index_hnsw.hnsw;
    size_t ntotal = n0 + n;
    double t0 = getmillisecs();
    if (verbose) {
        printf("hnsw_add_vertices: adding %ld elements on top of %ld "
               "(preset_levels=%d)\n",
               n, n0, int(preset_levels));
    }

    int max_level = prepare_level_tab (index_hnsw.hnsw, n, preset_levels);

    if (verbose) {
        printf("  max_level = %d\n", max_level);
    }

    std::vector<omp_lock_t> locks(ntotal);
    for(int i = 0; i < ntotal; i++)
        omp_init_lock(&locks[i]);

    // add vectors from highest to lowest level
    std::vector<int> hist;
    std::vector<int> order(n);

    { // make buckets with vectors of the same level

        // build histogram
        for (int i = 0; i < n; i++) {
            storage_idx_t pt_id = i + n0;
            int pt_level = hnsw.levels[pt_id] - 1;
            while (pt_level >= hist.size())
                hist.push_back(0);
            hist[pt_level] ++;
        }

        // accumulate
        std::vector<int> offsets(hist.size() + 1, 0);
        for (int i = 0; i < hist.size() - 1; i++) {
            offsets[i + 1] = offsets[i] + hist[i];
        }

        // bucket sort
        for (int i = 0; i < n; i++) {
            storage_idx_t pt_id = i + n0;
            int pt_level = hnsw.levels[pt_id] - 1;
            order[offsets[pt_level]++] = pt_id;
        }
    }

    { // perform add
        RandomGenerator rng2(789);

        int i1 = n;

        for (int pt_level = hist.size() - 1; pt_level >= 0; pt_level--) {
            int i0 = i1 - hist[pt_level];

            if (verbose) {
                printf("Adding %d elements at level %d\n",
                       i1 - i0, pt_level);
            }

            // random permutation to get rid of dataset order bias
            for (int j = i0; j < i1; j++)
                std::swap(order[j], order[j + rng2.rand_int(i1 - j)]);

#pragma omp parallel
            {
                VisitedTable vt (ntotal);

                DistanceComputer *dis = index_hnsw.get_distance_computer();
                ScopeDeleter1<DistanceComputer> del(dis);
                int prev_display = verbose && omp_get_thread_num() == 0 ? 0 : -1;

#pragma omp  for schedule(dynamic)
                for (int i = i0; i < i1; i++) {
                    storage_idx_t pt_id = order[i];
                    dis->set_query (x + (pt_id - n0) * dis->d);

                    hnsw.add_with_locks (
                           *dis, pt_level, pt_id, locks,
                           vt);

                    if (prev_display >= 0 && i - i0 > prev_display + 10000) {
                        prev_display = i - i0;
                        printf("  %d / %d\r", i - i0, i1 - i0);
                        fflush(stdout);
                    }
                }
            }
            i1 = i0;
        }
        FAISS_ASSERT(i1 == 0);
    }
    if (verbose)
        printf("Done in %.3f ms\n", getmillisecs() - t0);

    for(int i = 0; i < ntotal; i++)
        omp_destroy_lock(&locks[i]);

}


} // anonymous namespace


void HNSW::fill_with_random_links(size_t n)
{
    int max_level = prepare_level_tab (*this, n);
    RandomGenerator rng2(456);

    for (int level = max_level - 1; level >= 0; level++) {
        std::vector<int> elts;
        for (int i = 0; i < n; i++) {
            if (levels[i] > level) {
                elts.push_back(i);
            }
        }
        printf ("linking %ld elements in level %d\n",
                elts.size(), level);

        if (elts.size() == 1) continue;

        for (int ii = 0; ii < elts.size(); ii++) {
            int i = elts[ii];
            size_t begin, end;
            neighbor_range(i, 0, &begin, &end);
            for (size_t j = begin; j < end; j++) {
                int other = 0;
                do {
                    other = elts[rng2.rand_int(elts.size())];
                } while(other == i);

                neighbors[j] = other;
            }

        }

    }

}




/**************************************************************
 * IndexHNSW implementation
 **************************************************************/

IndexHNSW::IndexHNSW(int d, int M):
    Index(d, METRIC_L2),
    hnsw(M),
    own_fields(false),
    storage(nullptr),
    reconstruct_from_neighbors(nullptr)
{}

IndexHNSW::IndexHNSW(Index *storage, int M):
    Index(storage->d, METRIC_L2),
    hnsw(M),
    own_fields(false),
    storage(storage),
    reconstruct_from_neighbors(nullptr)
{}

IndexHNSW::~IndexHNSW() {
    if (own_fields) {
        delete storage;
    }
}

void IndexHNSW::train(idx_t n, const float* x)
{
    // hnsw structure does not require training
    storage->train (n, x);
    is_trained = true;
}

void IndexHNSW::search (idx_t n, const float *x, idx_t k,
                            float *distances, idx_t *labels) const

{

#pragma omp parallel
    {
        VisitedTable vt (ntotal);
        DistanceComputer *dis = get_distance_computer();
        ScopeDeleter1<DistanceComputer> del(dis);
        size_t nreorder = 0;

#pragma omp for
        for(int i = 0; i < n; i++) {
            idx_t * idxi = labels + i * k;
            float * simi = distances + i * k;
            dis->set_query(x + i * d);

            maxheap_heapify (k, simi, idxi);
            hnsw.search (*dis, k, idxi, simi, vt);

            maxheap_reorder (k, simi, idxi);

            if (reconstruct_from_neighbors &&
                reconstruct_from_neighbors->k_reorder != 0) {
                int k_reorder = reconstruct_from_neighbors->k_reorder;
                if (k_reorder == -1 || k_reorder > k) k_reorder = k;

                nreorder += reconstruct_from_neighbors->compute_distances(
                       k_reorder, idxi, x + i * d, simi);

                // sort top k_reorder
                maxheap_heapify (k_reorder, simi, idxi, simi, idxi, k_reorder);
                maxheap_reorder (k_reorder, simi, idxi);
            }
        }
#pragma omp critical
        {
            hnsw_stats.nreorder += nreorder;
        }
    }


}


void IndexHNSW::add(idx_t n, const float *x)
{
    FAISS_THROW_IF_NOT(is_trained);
    int n0 = ntotal;
    storage->add(n, x);
    ntotal = storage->ntotal;

    hnsw_add_vertices (*this, n0, n, x, verbose,
                       hnsw.levels.size() == ntotal);

}

void IndexHNSW::reset()
{
    hnsw.reset();
    storage->reset();
    ntotal = 0;
}

void IndexHNSW::reconstruct (idx_t key, float* recons) const
{
    storage->reconstruct(key, recons);
}

void IndexHNSW::shrink_level_0_neighbors(int new_size)
{
#pragma omp parallel
    {
        DistanceComputer *dis = get_distance_computer();
        ScopeDeleter1<DistanceComputer> del(dis);

#pragma omp for
        for (idx_t i = 0; i < ntotal; i++) {

            size_t begin, end;
            hnsw.neighbor_range(i, 0, &begin, &end);

            std::priority_queue<NodeDistFarther> initial_list;

            for (size_t j = begin; j < end; j++) {
                int v1 = hnsw.neighbors[j];
                if (v1 < 0) break;
                initial_list.emplace(dis->symmetric_dis(i, v1), v1);

                // initial_list.emplace(qdis(v1), v1);
            }

            std::vector<NodeDistFarther> shrunk_list;
            shrink_neighbor_list (*dis, initial_list, shrunk_list, new_size);

            for (size_t j = begin; j < end; j++) {
                if (j - begin < shrunk_list.size())
                    hnsw.neighbors[j] = shrunk_list[j - begin].id;
                else
                    hnsw.neighbors[j] = -1;
            }
        }
    }

}

void IndexHNSW::search_level_0(
                        idx_t n, const float *x, idx_t k,
                        const storage_idx_t *nearest, const float *nearest_d,
                        float *distances, idx_t *labels, int nprobe,
                        int search_type) const
{

    storage_idx_t ntotal = hnsw.levels.size();
#pragma omp parallel
    {
        DistanceComputer *qdis = get_distance_computer();
        ScopeDeleter1<DistanceComputer> del(qdis);

        VisitedTable vt (ntotal);

#pragma omp for
        for(idx_t i = 0; i < n; i++) {
            idx_t * idxi = labels + i * k;
            float * simi = distances + i * k;

            qdis->set_query(x + i * d);
            maxheap_heapify (k, simi, idxi);

            if (search_type == 1) {

                int nres = 0;

                for(int j = 0; j < nprobe; j++) {
                    storage_idx_t cj = nearest[i * nprobe + j];

                    if (cj < 0) break;

                    if (vt.get(cj)) continue;

                    int candidates_size = std::max(hnsw.efSearch, int(k));
                    MinimaxHeap candidates(candidates_size);

                    candidates.push(cj, nearest_d[i * nprobe + j]);

                    nres = search_from_candidates (
                      hnsw, *qdis, k, idxi, simi,
                      candidates, vt, 0, nres);
                }
            } else if (search_type == 2) {

                int candidates_size = std::max(hnsw.efSearch, int(k));
                candidates_size = std::max(candidates_size, nprobe);

                MinimaxHeap candidates(candidates_size);
                for(int j = 0; j < nprobe; j++) {
                    storage_idx_t cj = nearest[i * nprobe + j];

                    if (cj < 0) break;
                    candidates.push(cj, nearest_d[i * nprobe + j]);
                }
                search_from_candidates (
                      hnsw, *qdis, k, idxi, simi,
                      candidates, vt, 0);

            }
            vt.advance();

            maxheap_reorder (k, simi, idxi);

        }
    }


}

void IndexHNSW::init_level_0_from_knngraph(
       int k, const float *D, const idx_t *I)
{
    int dest_size = hnsw.nb_neighbors (0);

#pragma omp parallel for
    for (idx_t i = 0; i < ntotal; i++) {
        DistanceComputer *qdis = get_distance_computer();
        float vec[d];
        storage->reconstruct(i, vec);
        qdis->set_query(vec);

        std::priority_queue<NodeDistFarther> initial_list;

        for (size_t j = 0; j < k; j++) {
            int v1 = I[i * k + j];
            if (v1 == i) continue;
            if (v1 < 0) break;
            initial_list.emplace(D[i * k + j], v1);
        }

        std::vector<NodeDistFarther> shrunk_list;
        shrink_neighbor_list (*qdis, initial_list, shrunk_list, dest_size);

        size_t begin, end;
        hnsw.neighbor_range(i, 0, &begin, &end);

        for (size_t j = begin; j < end; j++) {
            if (j - begin < shrunk_list.size())
                hnsw.neighbors[j] = shrunk_list[j - begin].id;
            else
                hnsw.neighbors[j] = -1;
        }
    }
}



void IndexHNSW::init_level_0_from_entry_points(
          int n, const storage_idx_t *points,
          const storage_idx_t *nearests)
{

    std::vector<omp_lock_t> locks(ntotal);
    for(int i = 0; i < ntotal; i++)
        omp_init_lock(&locks[i]);

#pragma omp parallel
    {
        VisitedTable vt (ntotal);

        DistanceComputer *dis = get_distance_computer();
        ScopeDeleter1<DistanceComputer> del(dis);
        float vec[storage->d];

#pragma omp  for schedule(dynamic)
        for (int i = 0; i < n; i++) {
            storage_idx_t pt_id = points[i];
            storage_idx_t nearest = nearests[i];
            storage->reconstruct (pt_id, vec);
            dis->set_query (vec);

            add_links_starting_from(hnsw, *dis, pt_id, nearest, (*dis)(nearest),
                                    0, locks.data(), vt);

            if (verbose && i % 10000 == 0) {
                printf("  %d / %d\r", i, n);
                fflush(stdout);
            }
        }
    }
    if (verbose) {
        printf("\n");
    }

    for(int i = 0; i < ntotal; i++)
        omp_destroy_lock(&locks[i]);
}

void IndexHNSW::reorder_links()
{
    int M = hnsw.nb_neighbors(0);

#pragma omp parallel
    {
        std::vector<float> distances (M);
        std::vector<size_t> order (M);
        std::vector<storage_idx_t> tmp (M);
        DistanceComputer *dis = get_distance_computer();
        ScopeDeleter1<DistanceComputer> del(dis);

#pragma omp for
        for(storage_idx_t i = 0; i < ntotal; i++) {

            size_t begin, end;
            hnsw.neighbor_range(i, 0, &begin, &end);

            for (size_t j = begin; j < end; j++) {
                storage_idx_t nj = hnsw.neighbors[j];
                if (nj < 0) {
                    end = j;
                    break;
                }
                distances[j - begin] = dis->symmetric_dis(i, nj);
                tmp [j - begin] = nj;
            }

            fvec_argsort (end - begin, distances.data(), order.data());
            for (size_t j = begin; j < end; j++) {
                hnsw.neighbors[j] = tmp[order[j - begin]];
            }
        }

    }
}


void IndexHNSW::link_singletons()
{
    printf("search for singletons\n");

    std::vector<bool> seen(ntotal);

    for (size_t i = 0; i < ntotal; i++) {
        size_t begin, end;
        hnsw.neighbor_range(i, 0, &begin, &end);
        for (size_t j = begin; j < end; j++) {
            storage_idx_t ni = hnsw.neighbors[j];
            if (ni >= 0) seen[ni] = true;
        }
    }

    int n_sing = 0, n_sing_l1 = 0;
    std::vector<storage_idx_t> singletons;
    for (storage_idx_t i = 0; i < ntotal; i++) {
        if (!seen[i]) {
            singletons.push_back(i);
            n_sing++;
            if (hnsw.levels[i] > 1)
                n_sing_l1++;
        }
    }

    printf("  Found %d / %ld singletons (%d appear in a level above)\n",
           n_sing, ntotal, n_sing_l1);

    std::vector<float>recons(singletons.size() * d);
    for (int i = 0; i < singletons.size(); i++) {

        FAISS_ASSERT(!"not implemented");

    }


}




// storage that explicitly reconstructs vectors before computing distances
struct GenericDistanceComputer: HNSW::DistanceComputer {

    const Index & storage;
    std::vector<float> buf;
    const float *q;

    GenericDistanceComputer(const Index & storage): storage(storage)
    {
        d = storage.d;
        buf.resize(d * 2);
    }

    float operator () (storage_idx_t i) override
    {
        storage.reconstruct(i, buf.data());
        return fvec_L2sqr(q, buf.data(), d);
    }

    float symmetric_dis(storage_idx_t i, storage_idx_t j) override
    {
        storage.reconstruct(i, buf.data());
        storage.reconstruct(j, buf.data() + d);
        return fvec_L2sqr(buf.data() + d, buf.data(), d);
    }

    void set_query(const float *x) override {
        q = x;
    }


};

HNSW::DistanceComputer * IndexHNSW::get_distance_computer () const
{
    return new GenericDistanceComputer (*storage);
}


/**************************************************************
 * ReconstructFromNeighbors implementation
 **************************************************************/


ReconstructFromNeighbors::ReconstructFromNeighbors(
             const IndexHNSW & index, size_t k, size_t nsq):
    index(index), k(k), nsq(nsq) {
    M = index.hnsw.nb_neighbors(0);
    FAISS_ASSERT(k <= 256);
    code_size = k == 1 ? 0 : nsq;
    ntotal = 0;
    d = index.d;
    FAISS_ASSERT(d % nsq == 0);
    dsub = d / nsq;
    k_reorder = -1;
}

void ReconstructFromNeighbors::reconstruct(storage_idx_t i, float *x, float *tmp) const
{


    const HNSW & hnsw = index.hnsw;
    size_t begin, end;
    hnsw.neighbor_range(i, 0, &begin, &end);

    if (k == 1 || nsq == 1) {
        const float * beta;
        if (k == 1) {
            beta = codebook.data();
        } else {
            int idx = codes[i];
            beta = codebook.data() + idx * (M + 1);
        }

        float w0 = beta[0]; // weight of image itself
        index.storage->reconstruct(i, tmp);

        for (int l = 0; l < d; l++)
            x[l] = w0 * tmp[l];

        for (size_t j = begin; j < end; j++) {

            storage_idx_t ji = hnsw.neighbors[j];
            if (ji < 0) ji = i;
            float w = beta[j - begin + 1];
            index.storage->reconstruct(ji, tmp);
            for (int l = 0; l < d; l++)
                x[l] += w * tmp[l];
        }
    } else if (nsq == 2) {
        int idx0 = codes[2 * i];
        int idx1 = codes[2 * i + 1];

        const float *beta0 = codebook.data() +  idx0 * (M + 1);
        const float *beta1 = codebook.data() + (idx1 + k) * (M + 1);

        index.storage->reconstruct(i, tmp);

        float w0;

        w0 = beta0[0];
        for (int l = 0; l < dsub; l++)
            x[l] = w0 * tmp[l];

        w0 = beta1[0];
        for (int l = dsub; l < d; l++)
            x[l] = w0 * tmp[l];

        for (size_t j = begin; j < end; j++) {
            storage_idx_t ji = hnsw.neighbors[j];
            if (ji < 0) ji = i;
            index.storage->reconstruct(ji, tmp);
            float w;
            w = beta0[j - begin + 1];
            for (int l = 0; l < dsub; l++)
                x[l] += w * tmp[l];

            w = beta1[j - begin + 1];
            for (int l = dsub; l < d; l++)
                x[l] += w * tmp[l];
        }
    } else {
        const float *betas[nsq];
        {
            const float *b = codebook.data();
            const uint8_t *c = &codes[i * code_size];
            for (int sq = 0; sq < nsq; sq++) {
                betas[sq] = b + (*c++) * (M + 1);
                b += (M + 1) * k;
            }
        }

        index.storage->reconstruct(i, tmp);
        {
            int d0 = 0;
            for (int sq = 0; sq < nsq; sq++) {
                float w = *(betas[sq]++);
                int d1 = d0 + dsub;
                for (int l = d0; l < d1; l++) {
                    x[l] = w * tmp[l];
                }
                d0 = d1;
            }
        }

        for (size_t j = begin; j < end; j++) {
            storage_idx_t ji = hnsw.neighbors[j];
            if (ji < 0) ji = i;

            index.storage->reconstruct(ji, tmp);
            int d0 = 0;
            for (int sq = 0; sq < nsq; sq++) {
                float w = *(betas[sq]++);
                int d1 = d0 + dsub;
                for (int l = d0; l < d1; l++) {
                    x[l] += w * tmp[l];
                }
                d0 = d1;
            }
        }
    }
}

void ReconstructFromNeighbors::reconstruct_n(storage_idx_t n0,
                                             storage_idx_t ni,
                                             float *x) const
{
#pragma omp parallel
    {
        std::vector<float> tmp(index.d);
#pragma omp for
        for (storage_idx_t i = 0; i < ni; i++) {
            reconstruct(n0 + i, x + i * index.d, tmp.data());
        }
    }
}

size_t ReconstructFromNeighbors::compute_distances(size_t n, const idx_t *shortlist,
                                                 const float *query, float *distances) const
{
    std::vector<float> tmp(2 * index.d);
    size_t ncomp = 0;
    for (int i = 0; i < n; i++) {
        if (shortlist[i] < 0) break;
        reconstruct(shortlist[i], tmp.data(), tmp.data() + index.d);
        distances[i] = fvec_L2sqr(query, tmp.data(), index.d);
        ncomp++;
    }
    return ncomp;
}

void ReconstructFromNeighbors::get_neighbor_table(storage_idx_t i, float *tmp1) const
{
    const HNSW & hnsw = index.hnsw;
    size_t begin, end;
    hnsw.neighbor_range(i, 0, &begin, &end);
    size_t d = index.d;

    index.storage->reconstruct(i, tmp1);

    for (size_t j = begin; j < end; j++) {
        storage_idx_t ji = hnsw.neighbors[j];
        if (ji < 0) ji = i;
        index.storage->reconstruct(ji, tmp1 + (j - begin + 1) * d);
    }

}


/// called by add_codes
void ReconstructFromNeighbors::estimate_code(
       const float *x, storage_idx_t i, uint8_t *code) const
{

    // fill in tmp table with the neighbor values
    float *tmp1 = new float[d * (M + 1) + (d * k)];
    float *tmp2 = tmp1 + d * (M + 1);
    ScopeDeleter<float> del(tmp1);

    // collect coordinates of base
    get_neighbor_table (i, tmp1);

    for (int sq = 0; sq < nsq; sq++) {
        int d0 = sq * dsub;
        int d1 = d0 + dsub;

        {
            FINTEGER ki = k, di = d, m1 = M + 1;
            FINTEGER dsubi = dsub;
            float zero = 0, one = 1;

            sgemm_ ("N", "N", &dsubi, &ki, &m1, &one,
                    tmp1 + d0, &di,
                    codebook.data() + sq * (m1 * k), &m1,
                    &zero, tmp2, &dsubi);
        }

        float min = HUGE_VAL;
        int argmin = -1;
        for (int j = 0; j < k; j++) {
            float dis = fvec_L2sqr(x + d0, tmp2 + j * dsub, dsub);
            if (dis < min) {
                min = dis;
                argmin = j;
            }
        }
        code[sq] = argmin;
    }

}

void ReconstructFromNeighbors::add_codes(size_t n, const float *x)
{
    if (k == 1) { // nothing to encode
        ntotal += n;
        return;
    }
    codes.resize(codes.size() + code_size * n);
#pragma omp parallel for
    for (int i = 0; i < n; i++) {
        estimate_code(x + i * index.d, ntotal + i,
                      codes.data() + (ntotal + i) * code_size);
    }
    ntotal += n;
    FAISS_ASSERT (codes.size() == ntotal * code_size);
}


/**************************************************************
 * IndexHNSWFlat implementation
 **************************************************************/


struct FlatL2Dis: HNSW::DistanceComputer {
    Index::idx_t nb;
    const float *q;
    const float *b;
    size_t ndis;

    float operator () (storage_idx_t i) override
    {
        ndis++;
        return (fvec_L2sqr(q, b + i * d, d));
    }

    float symmetric_dis(storage_idx_t i, storage_idx_t j) override
    {
        return (fvec_L2sqr(b + j * d, b + i * d, d));
    }


    FlatL2Dis(const IndexFlatL2 & storage, const float *q = nullptr):
        q(q)
    {
        nb = storage.ntotal;
        d = storage.d;
        b = storage.xb.data();
        ndis = 0;
    }

    void set_query(const float *x) override {
        q = x;
    }

    virtual ~FlatL2Dis () {
#pragma omp critical
        {
            hnsw_stats.ndis += ndis;
        }
    }
};


IndexHNSWFlat::IndexHNSWFlat()
{
    is_trained = true;
}


IndexHNSWFlat::IndexHNSWFlat(int d, int M):
    IndexHNSW(new IndexFlatL2(d), M)
{
    own_fields = true;
    is_trained = true;
}


HNSW::DistanceComputer * IndexHNSWFlat::get_distance_computer () const
{
    return new FlatL2Dis (*dynamic_cast<IndexFlatL2*> (storage));
}




/**************************************************************
 * IndexHNSWPQ implementation
 **************************************************************/


struct PQDis: HNSW::DistanceComputer {
    Index::idx_t nb;
    const uint8_t *codes;
    size_t code_size;
    const ProductQuantizer & pq;
    const float *sdc;
    std::vector<float> precomputed_table;
    size_t ndis;

    float operator () (storage_idx_t i) override
    {
        const uint8_t *code = codes + i * code_size;
        const float *dt = precomputed_table.data();
        float accu = 0;
        for (int j = 0; j < pq.M; j++) {
            accu += dt[*code++];
            dt += 256;
        }
        ndis++;
        return accu;
    }

    float symmetric_dis(storage_idx_t i, storage_idx_t j) override
    {
        const float * sdci = sdc;
        float accu = 0;
        const uint8_t *codei = codes + i * code_size;
        const uint8_t *codej = codes + j * code_size;

        for (int l = 0; l < pq.M; l++) {
            accu += sdci[(*codei++) + (*codej++) * 256];
            sdci += 256 * 256;
        }
        return accu;
    }


    PQDis(const IndexPQ & storage, const float *q = nullptr):
        pq(storage.pq)
    {
        precomputed_table.resize(pq.M * pq.ksub);
        nb = storage.ntotal;
        d = storage.d;
        codes = storage.codes.data();
        code_size = pq.code_size;
        FAISS_ASSERT(pq.ksub == 256);
        FAISS_ASSERT(pq.sdc_table.size() == pq.ksub * pq.ksub * pq.M);
        sdc = pq.sdc_table.data();
        ndis = 0;
    }

    void set_query(const float *x) override {
        pq.compute_distance_table(x, precomputed_table.data());
    }

    virtual ~PQDis () {
#pragma omp critical
        {
            hnsw_stats.ndis += ndis;
        }
    }
};

IndexHNSWPQ::IndexHNSWPQ() {}

IndexHNSWPQ::IndexHNSWPQ(int d, int pq_m, int M):
    IndexHNSW(new IndexPQ(d, pq_m, 8), M)
{
    own_fields = true;
    is_trained = false;
}

void IndexHNSWPQ::train(idx_t n, const float* x)
{
    IndexHNSW::train (n, x);
    (dynamic_cast<IndexPQ*> (storage))->pq.compute_sdc_table();
}



HNSW::DistanceComputer * IndexHNSWPQ::get_distance_computer () const
{
    return new PQDis (*dynamic_cast<IndexPQ*> (storage));
}


/**************************************************************
 * IndexHNSWSQ implementation
 **************************************************************/


struct SQDis: HNSW::DistanceComputer {
    Index::idx_t nb;
    const uint8_t *codes;
    size_t code_size;
    const ScalarQuantizer & sq;
    const float *q;
    ScalarQuantizer::DistanceComputer * dc;

    float operator () (storage_idx_t i) override
    {
        const uint8_t *code = codes + i * code_size;

        return dc->compute_distance (q, code);
    }

    float symmetric_dis(storage_idx_t i, storage_idx_t j) override
    {
        const uint8_t *codei = codes + i * code_size;
        const uint8_t *codej = codes + j * code_size;
        return dc->compute_code_distance (codei, codej);
    }


    SQDis(const IndexScalarQuantizer & storage, const float *q = nullptr):
        sq(storage.sq)
    {
        nb = storage.ntotal;
        d = storage.d;
        codes = storage.codes.data();
        code_size = sq.code_size;
        dc = sq.get_distance_computer();
    }

    void set_query(const float *x) override {
        q = x;
    }

    virtual ~SQDis () {
        delete dc;
    }
};

IndexHNSWSQ::IndexHNSWSQ(int d, ScalarQuantizer::QuantizerType qtype, int M):
    IndexHNSW (new IndexScalarQuantizer (d, qtype), M)
{
    own_fields = true;
}

IndexHNSWSQ::IndexHNSWSQ() {}

HNSW::DistanceComputer * IndexHNSWSQ::get_distance_computer () const
{
    return new SQDis (*dynamic_cast<IndexScalarQuantizer*> (storage));
}




/**************************************************************
 * IndexHNSW2Level implementation
 **************************************************************/



IndexHNSW2Level::IndexHNSW2Level(Index *quantizer, size_t nlist, int m_pq, int M):
    IndexHNSW (new Index2Layer (quantizer, nlist, m_pq), M)
{
    own_fields = true;
    is_trained = false;
}

IndexHNSW2Level::IndexHNSW2Level() {}

struct Distance2Level: HNSW::DistanceComputer {

    const Index2Layer & storage;
    std::vector<float> buf;
    const float *q;

    const float *pq_l1_tab, *pq_l2_tab;

    Distance2Level(const Index2Layer & storage): storage(storage)
    {
        d = storage.d;
        FAISS_ASSERT(storage.pq.dsub == 4);
        pq_l2_tab = storage.pq.centroids.data();
        buf.resize(2 * d);
    }

    float symmetric_dis(storage_idx_t i, storage_idx_t j) override
    {
        storage.reconstruct(i, buf.data());
        storage.reconstruct(j, buf.data() + d);
        return fvec_L2sqr(buf.data() + d, buf.data(), d);
    }

    void set_query(const float *x) override {
        q = x;
    }


};


// well optimized for xNN+PQNN
struct DistanceXPQ4: Distance2Level {

    int M, k;

    DistanceXPQ4(const Index2Layer & storage):
        Distance2Level (storage)
    {
        const IndexFlat *quantizer =
            dynamic_cast<IndexFlat*> (storage.q1.quantizer);

        FAISS_ASSERT(quantizer);
        M = storage.pq.M;
        pq_l1_tab = quantizer->xb.data();
    }

    float operator () (storage_idx_t i) override
    {
        const uint8_t *code = storage.codes.data() + i * storage.code_size;
        long key = 0;
        memcpy (&key, code, storage.code_size_1);
        code += storage.code_size_1;

        // walking pointers
        const float *qa = q;
        const __m128 *l1_t = (const __m128 *)(pq_l1_tab + d * key);
        const __m128 *pq_l2_t = (const __m128 *)pq_l2_tab;
        __m128 accu = _mm_setzero_ps();

        for (int m = 0; m < M; m++) {
            __m128 qi = _mm_loadu_ps(qa);
            __m128 recons = l1_t[m] + pq_l2_t[*code++];
            __m128 diff = qi - recons;
            accu += diff * diff;
            pq_l2_t += 256;
            qa += 4;
        }

        accu = _mm_hadd_ps (accu, accu);
        accu = _mm_hadd_ps (accu, accu);
        return  _mm_cvtss_f32 (accu);
    }

};

// well optimized for 2xNN+PQNN
struct Distance2xXPQ4: Distance2Level {

    int M_2, mi_nbits;

    Distance2xXPQ4(const Index2Layer & storage):
        Distance2Level (storage)
    {
        const MultiIndexQuantizer *mi =
            dynamic_cast<MultiIndexQuantizer*> (storage.q1.quantizer);

        FAISS_ASSERT(mi);
        FAISS_ASSERT(storage.pq.M % 2 == 0);
        M_2 = storage.pq.M / 2;
        mi_nbits = mi->pq.nbits;
        pq_l1_tab = mi->pq.centroids.data();
    }

    float operator () (storage_idx_t i) override
    {
        const uint8_t *code = storage.codes.data() + i * storage.code_size;
        long key01 = 0;
        memcpy (&key01, code, storage.code_size_1);
        code += storage.code_size_1;

        // walking pointers
        const float *qa = q;
        const __m128 *pq_l1_t = (const __m128 *)pq_l1_tab;
        const __m128 *pq_l2_t = (const __m128 *)pq_l2_tab;
        __m128 accu = _mm_setzero_ps();

        for (int mi_m = 0; mi_m < 2; mi_m++) {
            long l1_idx = key01 & ((1L << mi_nbits) - 1);
            const __m128 * pq_l1 = pq_l1_t + M_2 * l1_idx;

            for (int m = 0; m < M_2; m++) {
                __m128 qi = _mm_loadu_ps(qa);
                __m128 recons = pq_l1[m] + pq_l2_t[*code++];
                __m128 diff = qi - recons;
                accu += diff * diff;
                pq_l2_t += 256;
                qa += 4;
            }
            pq_l1_t += M_2 << mi_nbits;
            key01 >>= mi_nbits;
        }
        accu = _mm_hadd_ps (accu, accu);
        accu = _mm_hadd_ps (accu, accu);
        return  _mm_cvtss_f32 (accu);
    }

};



HNSW::DistanceComputer * IndexHNSW2Level::get_distance_computer () const
{
    const Index2Layer *storage2l =
        dynamic_cast<Index2Layer*>(storage);

    if (storage2l) {

        const MultiIndexQuantizer *mi =
            dynamic_cast<MultiIndexQuantizer*> (storage2l->q1.quantizer);

        if (mi && storage2l->pq.M % 2 == 0 && storage2l->pq.dsub == 4) {
            return new Distance2xXPQ4(*storage2l);
        }

        const IndexFlat *fl =
            dynamic_cast<IndexFlat*> (storage2l->q1.quantizer);

        if (fl && storage2l->pq.dsub == 4) {
            return new DistanceXPQ4(*storage2l);
        }
    }

    // IVFPQ and cases not handled above
    return new GenericDistanceComputer (*storage);

}


namespace {


// same as search_from_candidates but uses v
// visno -> is in result list
// visno + 1 -> in result list + in candidates
int search_from_candidates_2(const HNSW & hnsw,
                             DistanceComputer & qdis, int k,
                             idx_t *I, float * D,
                             MinimaxHeap &candidates,
                             VisitedTable &vt,
                             int level, int nres_in = 0)
{
    int nres = nres_in;
    int ndis = 0;
    for (int i = 0; i < candidates.size(); i++) {
        idx_t v1 = candidates.ids[i];
        float d = candidates.dis[i];
        FAISS_ASSERT(v1 >= 0);
        vt.visited[v1] = vt.visno + 1;
    }

    bool do_dis_check = hnsw.check_relative_distance;
    int nstep = 0;

    while (candidates.size() > 0) {
        float d0 = 0;
        int v0 = candidates.pop_min(&d0);

        if (do_dis_check) {
            // tricky stopping condition: there are more that ef
            // distances that are processed already that are smaller
            // than d0

            int n_dis_below = candidates.count_below(d0);
            if(n_dis_below >= hnsw.efSearch) {
                break;
            }
        }
        size_t begin, end;
        hnsw.neighbor_range(v0, level, &begin, &end);

        for (size_t j = begin; j < end; j++) {
            int v1 = hnsw.neighbors[j];
            if (v1 < 0) break;
            if (vt.visited[v1] == vt.visno + 1) {
                // nothing to do
            } else {
                ndis++;
                float d = qdis(v1);
                candidates.push(v1, d);

                // never seen before --> add to heap
                if (vt.visited[v1] < vt.visno) {
                    if (nres < k) {
                        faiss::maxheap_push (++nres, D, I, d, v1);
                    } else if (d < D[0]) {
                        faiss::maxheap_pop (nres--, D, I);
                        faiss::maxheap_push (++nres, D, I, d, v1);
                    }
                }
                vt.visited[v1] = vt.visno + 1;
            }
        }

        nstep++;
        if (!do_dis_check && nstep > hnsw.efSearch) {
            break;
        }
    }

    if (level == 0) {
#pragma omp critical
        {
            hnsw_stats.n1 ++;
            if (candidates.size() == 0)
                hnsw_stats.n2 ++;
        }
    }


    return nres;
}


} // anonymous namespace

void IndexHNSW2Level::search (idx_t n, const float *x, idx_t k,
                              float *distances, idx_t *labels) const
{
    if (dynamic_cast<const Index2Layer*>(storage)) {
        IndexHNSW::search (n, x, k, distances, labels);

    } else { // "mixed" search

        const IndexIVFPQ *index_ivfpq =
            dynamic_cast<const IndexIVFPQ*>(storage);

        int nprobe = index_ivfpq->nprobe;

        long * coarse_assign = new long [n * nprobe];
        ScopeDeleter<long> del (coarse_assign);
        float * coarse_dis = new float [n * nprobe];
        ScopeDeleter<float> del2 (coarse_dis);

        index_ivfpq->quantizer->search (n, x, nprobe, coarse_dis, coarse_assign);

        index_ivfpq->search_preassigned (
            n, x, k, coarse_assign, coarse_dis, distances, labels, false);

#pragma omp parallel
        {
            VisitedTable vt (ntotal);
            DistanceComputer *dis = get_distance_computer();
            ScopeDeleter1<DistanceComputer> del(dis);

            int candidates_size = hnsw.upper_beam;
            MinimaxHeap candidates(candidates_size);

#pragma omp for
            for(int i = 0; i < n; i++) {
                idx_t * idxi = labels + i * k;
                float * simi = distances + i * k;
                dis->set_query(x + i * d);

                // mark all inverted list elements as visited

                for (int j = 0; j < nprobe; j++) {
                    idx_t key = coarse_assign[j + i * nprobe];
                    if (key < 0) break;
                    size_t list_length = index_ivfpq->get_list_size (key);
                    const idx_t * ids = index_ivfpq->invlists->get_ids (key);

                    for (int jj = 0; jj < list_length; jj++) {
                        vt.set (ids[jj]);
                    }
                }

                candidates.clear();
                // copy the upper_beam elements to candidates list

                int search_policy = 2;

                if (search_policy == 1) {

                    for (int j = 0 ; j < hnsw.upper_beam && j < k; j++) {
                        if (idxi[j] < 0) break;
                        candidates.push (idxi[j], simi[j]);
                        // search_from_candidates adds them back
                        idxi[j] = -1;
                        simi[j] = HUGE_VAL;
                    }

                    // reorder from sorted to heap
                    maxheap_heapify (k, simi, idxi, simi, idxi, k);

                    search_from_candidates (
                        hnsw, *dis, k, idxi, simi,
                        candidates, vt, 0, k);

                    vt.advance();

                } else if (search_policy == 2) {

                    for (int j = 0 ; j < hnsw.upper_beam && j < k; j++) {
                        if (idxi[j] < 0) break;
                        candidates.push (idxi[j], simi[j]);
                    }

                    // reorder from sorted to heap
                    maxheap_heapify (k, simi, idxi, simi, idxi, k);

                    search_from_candidates_2 (
                        hnsw, *dis, k, idxi, simi,
                        candidates, vt, 0, k);
                    vt.advance ();
                    vt.advance ();

                }

                maxheap_reorder (k, simi, idxi);
            }
        }
    }


}


void IndexHNSW2Level::flip_to_ivf ()
{
    Index2Layer *storage2l =
        dynamic_cast<Index2Layer*>(storage);

    FAISS_THROW_IF_NOT (storage2l);

    IndexIVFPQ * index_ivfpq =
        new IndexIVFPQ (storage2l->q1.quantizer,
                        d, storage2l->q1.nlist,
                        storage2l->pq.M, 8);
    index_ivfpq->pq = storage2l->pq;
    index_ivfpq->is_trained = storage2l->is_trained;
    index_ivfpq->precompute_table();
    index_ivfpq->own_fields = storage2l->q1.own_fields;
    storage2l->transfer_to_IVFPQ(*index_ivfpq);
    index_ivfpq->make_direct_map (true);

    storage = index_ivfpq;
    delete storage2l;

}


} // namespace faiss
