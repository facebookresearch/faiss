/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/impl/pipnn/Partition.h>

#include <omp.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <exception>
#include <iterator>
#include <limits>
#include <memory>
#include <utility>

#include <faiss/Index.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissAssert.h>
#include <faiss/impl/FaissException.h>
#include <faiss/impl/pipnn/kernels.h>
#include <faiss/utils/AlignedTable.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/random.h>

namespace faiss {
namespace pipnn {

int PartitionParams::fanout_at(int depth) const {
    FAISS_ASSERT(depth >= 0);
    return static_cast<size_t>(depth) < fanout.size() ? fanout[depth] : 1;
}

uint64_t mix_seed(uint64_t a, uint64_t b) {
    return SplitMix64RandomGenerator(int64_t(a ^ (b + (a << 6) + (a >> 2))))
            .next();
}

namespace detail {

size_t leader_count(size_t size, const PartitionParams& p, int depth) {
    size_t by_fraction = static_cast<size_t>(std::ceil(
            static_cast<double>(p.leader_fraction) *
            static_cast<double>(size)));
    size_t floor4 = 4 * static_cast<size_t>(p.fanout_at(depth));
    size_t count = std::max(by_fraction, floor4);
    count = std::min(count, static_cast<size_t>(p.leader_cap));
    return std::min(count, size);
}

void sample_positions(
        size_t N,
        size_t k,
        uint64_t seed,
        std::vector<size_t>& out) {
    FAISS_ASSERT(k <= N);
    out.clear();
    out.reserve(k);
    SplitMix64RandomGenerator rng(static_cast<int64_t>(seed));
    for (size_t j = N - k; j < N; j++) {
        size_t t = static_cast<size_t>(
                rng.next() % (static_cast<uint64_t>(j) + 1));
        auto it = std::lower_bound(out.begin(), out.end(), t);
        if (it != out.end() && *it == t) {
            out.push_back(j); // every sampled position is < j
        } else {
            out.insert(it, t);
        }
    }
}

Ranking ranking_for(const PartitionParams& p, bool force_l2) {
    if (p.metric == METRIC_L2 || force_l2) {
        return Ranking::L2;
    }
    return p.ip_partition_by_angle ? Ranking::Angle : Ranking::IP;
}

} // namespace detail

namespace {

/// Positions per counting-sort block; unlike kStripe, it does not affect
/// leaves.
constexpr size_t kScatterBlock = 65536;
/// Above any leader index, so the merge-order seed never equals a child's.
constexpr uint64_t kMergeShuffleIndex = uint64_t(1) << 33;

constexpr int kMaxCMax = 4096;

constexpr size_t kNoSplit = std::numeric_limits<size_t>::max();

/// `ids` is sorted and owned by a LevelStorage (or the root list).
struct Node {
    const int32_t* ids;
    size_t size;
    uint64_t seed;
    int depth;
    bool force_l2; // L2 ranking: this or an ancestor was a > 90% IP child
};

/// Owns the id lists of every node of one level.
struct LevelStorage {
    std::unique_ptr<int32_t[]> child_ids;
    std::vector<std::vector<int32_t>> merge_pools;
};

/// Inputs and stats shared by every level of one partition.
struct LevelContext {
    const Index& storage;
    const PartitionParams& p;
    const LeafVisitor& on_leaf;
    PartitionStats& stats;
};

/// A leaf (split == kNoSplit) or kStripe points of a split node from `begin`.
struct WorkItem {
    size_t node;
    size_t split;
    size_t begin;
};

struct SplitInfo {
    size_t node = 0;
    std::vector<int32_t> leaders; // ascending ids
    int fanout = 0;
    Ranking ranking = Ranking::L2;
    size_t offset = 0; // start in the level assignment / child-id buffers
    size_t n_blocks = 0;
    size_t count_offset = 0;       // start in the level count buffer
    std::vector<size_t> child_off; // |L| + 1 offsets into the child-id buffer
};

struct LevelPlan {
    std::vector<WorkItem> items;
    std::vector<SplitInfo> splits;
    size_t n_instances = 0; // (point, chosen leader) pairs of all splits
    size_t n_counts = 0;    // entries of the count buffer
};

/// Next-level nodes of one split. Merged groups have ids == nullptr until
/// collect_children points them into their pool.
struct SplitOutput {
    std::vector<Node> nodes;
    std::vector<size_t> pool_offsets;
    std::vector<int32_t> pool;
    size_t n_slices = 0;
    size_t n_l2 = 0;
};

void check_params(const Index& storage, const PartitionParams& p) {
    FAISS_THROW_IF_NOT_MSG(
            p.c_min >= 1 && p.c_min <= p.c_max,
            "PiPNN partition: need 1 <= c_min <= c_max");
    FAISS_THROW_IF_NOT_FMT(
            p.c_max <= kMaxCMax,
            "PiPNN partition: need c_max <= %d (got %d): each thread holds a "
            "c_max x c_max float matrix",
            kMaxCMax,
            p.c_max);
    FAISS_THROW_IF_NOT_MSG(
            p.leader_cap >= 1 && p.leader_cap <= 65535,
            "PiPNN partition: need 1 <= leader_cap <= 65535");
    FAISS_THROW_IF_NOT_MSG(
            p.leader_fraction >= 0 && p.leader_fraction <= 1,
            "PiPNN partition: need 0 <= leader_fraction <= 1");
    for (int f : p.fanout) {
        FAISS_THROW_IF_NOT_MSG(
                f >= 1, "PiPNN partition: every fanout must be >= 1");
    }
    FAISS_THROW_IF_NOT_MSG(
            p.metric == METRIC_L2 || p.metric == METRIC_INNER_PRODUCT,
            "PiPNN partition: metric must be L2 or inner product");
    FAISS_THROW_IF_NOT_MSG(
            storage.ntotal >= 0 &&
                    storage.ntotal <= std::numeric_limits<int32_t>::max(),
            "PiPNN partition: need 0 <= n <= INT32_MAX");
}

void record_leaf(PartitionStats& stats, size_t size, int depth) {
    stats.n_leaves++;
    stats.n_point_instances += size;
    stats.max_leaf = std::max(stats.max_leaf, size);
    stats.max_depth_seen = std::max(stats.max_depth_seen, depth);
    const size_t bucket = (size - 1) / 64;
    if (stats.leaf_size_histogram.size() <= bucket) {
        stats.leaf_size_histogram.resize(bucket + 1, 0);
    }
    stats.leaf_size_histogram[bucket]++;
}

void ensure_size(AlignedTable<float, 64>& table, size_t n) {
    if (table.size() < n) {
        table.resize(n);
    }
}

void push_slices(const Node& child, size_t c_max, SplitOutput& out) {
    for (size_t o = 0; o < child.size; o += c_max) {
        out.nodes.push_back(
                Node{child.ids + o,
                     std::min(c_max, child.size - o),
                     child.seed,
                     child.depth,
                     child.force_l2});
    }
}

LevelPlan plan_level(const LevelContext& ctx, const std::vector<Node>& nodes) {
    const PartitionParams& p = ctx.p;
    const size_t c_max = static_cast<size_t>(p.c_max);
    size_t n_items = 0;
    size_t n_splits = 0;
    for (const Node& nd : nodes) {
        if (nd.size <= c_max) {
            n_items++;
        } else {
            n_items += (nd.size + kStripe - 1) / kStripe;
            n_splits++;
        }
    }
    LevelPlan plan;
    plan.splits.reserve(n_splits);
    plan.items.reserve(n_items);
    for (size_t i = 0; i < nodes.size(); i++) {
        const Node& nd = nodes[i];
        if (nd.size <= c_max) {
            record_leaf(ctx.stats, nd.size, nd.depth);
            plan.items.push_back(WorkItem{i, kNoSplit, 0});
            continue;
        }
        SplitInfo sp;
        sp.node = i;
        const size_t n_leaders = detail::leader_count(nd.size, p, nd.depth);
        std::vector<size_t> positions;
        detail::sample_positions(nd.size, n_leaders, nd.seed, positions);
        sp.leaders.resize(n_leaders);
        for (size_t l = 0; l < n_leaders; l++) {
            sp.leaders[l] = nd.ids[positions[l]];
        }
        sp.fanout =
                std::min(p.fanout_at(nd.depth), static_cast<int>(n_leaders));
        sp.ranking = detail::ranking_for(p, nd.force_l2);
        sp.offset = plan.n_instances;
        plan.n_instances += nd.size * sp.fanout;
        sp.n_blocks = (nd.size + kScatterBlock - 1) / kScatterBlock;
        sp.count_offset = plan.n_counts;
        plan.n_counts += sp.n_blocks * n_leaders;
        for (size_t b = 0; b < nd.size; b += kStripe) {
            plan.items.push_back(WorkItem{i, plan.splits.size(), b});
        }
        plan.splits.push_back(std::move(sp));
    }
    return plan;
}

/// Runs the leaf and stripe items; returns each point's `fanout` nearest
/// leaders.
std::unique_ptr<uint16_t[]> assign_leaders(
        const LevelContext& ctx,
        const std::vector<Node>& nodes,
        const LevelPlan& plan) {
    const Index& storage = ctx.storage;
    const size_t d = static_cast<size_t>(storage.d);
    // new T[n], not std::vector: every entry is written first; zeroing is
    // serial.
    std::unique_ptr<uint16_t[]> assign(new uint16_t[plan.n_instances]);
    std::exception_ptr ex;
    std::atomic<bool> interrupt{false};
#pragma omp parallel
    {
        AlignedTable<float, 64> stripe_vecs, leader_vecs, dist;
        std::vector<float> leader_norms;
        size_t n_done = 0;
#pragma omp for schedule(dynamic)
        for (int64_t i = 0; i < static_cast<int64_t>(plan.items.size()); i++) {
            if (interrupt.load(std::memory_order_relaxed)) {
                continue;
            }
            try {
                const WorkItem& item = plan.items[i];
                const Node& nd = nodes[item.node];
                if (item.split == kNoSplit) {
                    ctx.on_leaf(nd.ids, nd.size);
                } else {
                    const SplitInfo& sp = plan.splits[item.split];
                    const size_t S = std::min(kStripe, nd.size - item.begin);
                    const size_t L = sp.leaders.size();
                    ensure_size(stripe_vecs, S * d);
                    ensure_size(leader_vecs, L * d);
                    ensure_size(dist, S * L);
                    leader_norms.resize(L);
                    gather_rows(
                            storage,
                            nd.ids + item.begin,
                            S,
                            stripe_vecs.data());
                    gather_rows(
                            storage, sp.leaders.data(), L, leader_vecs.data());
                    for (size_t l = 0; l < L; l++) {
                        leader_norms[l] =
                                fvec_norm_L2sqr(leader_vecs.data() + l * d, d);
                    }
                    stripe_assign(
                            stripe_vecs.data(),
                            S,
                            leader_vecs.data(),
                            leader_norms.data(),
                            L,
                            d,
                            sp.ranking,
                            sp.fanout,
                            dist.data(),
                            assign.get() + sp.offset + item.begin * sp.fanout);
                }
                if (omp_get_thread_num() == 0 && n_done++ % 64 == 0 &&
                    InterruptCallback::is_interrupted()) {
                    FAISS_THROW_MSG("computation interrupted");
                }
            } catch (...) {
                omp_capture_exception(ex, [&] { interrupt = true; });
            }
        }
    }
    omp_rethrow_if_exception(ex);
    return assign;
}

/// Counting sort per (split, block): count, prefix sums in (leader, block)
/// order, scatter. Every child comes out in parent order. Fills child_off and
/// returns the child ids, owned by next_storage.
const int32_t* scatter_children(
        const std::vector<Node>& nodes,
        LevelPlan& plan,
        const uint16_t* assign,
        LevelStorage& next_storage) {
    std::vector<SplitInfo>& splits = plan.splits;
    std::vector<std::pair<size_t, size_t>> blocks;
    for (size_t s = 0; s < splits.size(); s++) {
        for (size_t b = 0; b < splits[s].n_blocks; b++) {
            blocks.emplace_back(s, b);
        }
    }
    std::unique_ptr<size_t[]> counts(new size_t[plan.n_counts]);
#pragma omp parallel for schedule(dynamic)
    for (int64_t i = 0; i < static_cast<int64_t>(blocks.size()); i++) {
        const SplitInfo& sp = splits[blocks[i].first];
        const size_t b = blocks[i].second;
        const Node& nd = nodes[sp.node];
        const size_t L = sp.leaders.size();
        size_t* cnt = counts.get() + sp.count_offset + b * L;
        std::fill(cnt, cnt + L, 0);
        const uint16_t* a = assign + sp.offset;
        const size_t end = std::min(nd.size, (b + 1) * kScatterBlock);
        for (size_t pos = b * kScatterBlock; pos < end; pos++) {
            for (int j = 0; j < sp.fanout; j++) {
                cnt[a[pos * sp.fanout + j]]++;
            }
        }
    }
    for (SplitInfo& sp : splits) {
        const size_t L = sp.leaders.size();
        sp.child_off.resize(L + 1);
        size_t running = sp.offset;
        for (size_t l = 0; l < L; l++) {
            sp.child_off[l] = running;
            for (size_t b = 0; b < sp.n_blocks; b++) {
                size_t& c = counts[sp.count_offset + b * L + l];
                const size_t block_count = c;
                c = running;
                running += block_count;
            }
        }
        sp.child_off[L] = running;
    }
    next_storage.child_ids.reset(new int32_t[plan.n_instances]);
    int32_t* child_ids = next_storage.child_ids.get();
#pragma omp parallel for schedule(dynamic)
    for (int64_t i = 0; i < static_cast<int64_t>(blocks.size()); i++) {
        const SplitInfo& sp = splits[blocks[i].first];
        const size_t b = blocks[i].second;
        const Node& nd = nodes[sp.node];
        size_t* cursor = counts.get() + sp.count_offset + b * sp.leaders.size();
        const uint16_t* a = assign + sp.offset;
        const size_t end = std::min(nd.size, (b + 1) * kScatterBlock);
        for (size_t pos = b * kScatterBlock; pos < end; pos++) {
            for (int j = 0; j < sp.fanout; j++) {
                child_ids[cursor[a[pos * sp.fanout + j]]++] = nd.ids[pos];
            }
        }
    }
    return child_ids;
}

/// Child rules, in order: below c_min it is merged; up to c_max it is a leaf;
/// at max_depth it is sliced; an IP-ranked child above 90% of its parent is
/// re-split by L2; a child equal to its parent is sliced; else it is split.
void classify_children(
        const PartitionParams& p,
        const Node& nd,
        const SplitInfo& sp,
        const int32_t* child_ids,
        SplitOutput& out) {
    const size_t c_min = static_cast<size_t>(p.c_min);
    const size_t c_max = static_cast<size_t>(p.c_max);
    const int child_depth = nd.depth + 1;
    std::vector<uint32_t> small_children;
    for (size_t l = 0; l < sp.leaders.size(); l++) {
        Node child{
                child_ids + sp.child_off[l],
                sp.child_off[l + 1] - sp.child_off[l],
                mix_seed(nd.seed, l),
                child_depth,
                nd.force_l2};
        if (child.size < c_min) {
            small_children.push_back(static_cast<uint32_t>(l));
        } else if (child.size <= c_max) {
            out.nodes.push_back(child);
        } else if (child_depth >= p.max_depth) {
            push_slices(child, c_max, out);
            out.n_slices++;
        } else if (sp.ranking != Ranking::L2 && 10 * child.size > 9 * nd.size) {
            child.force_l2 = true;
            out.nodes.push_back(child);
            out.n_l2++;
        } else if (child.size == nd.size) {
            push_slices(child, c_max, out);
            out.n_slices++;
        } else {
            out.nodes.push_back(child);
        }
    }

    std::vector<int> merge_order(small_children.size());
    rand_perm_splitmix64(
            merge_order.data(),
            merge_order.size(),
            static_cast<int64_t>(mix_seed(nd.seed, kMergeShuffleIndex)));
    std::vector<int32_t> group;
    auto close_group = [&]() {
        std::sort(group.begin(), group.end());
        group.erase(std::unique(group.begin(), group.end()), group.end());
        out.pool_offsets.push_back(out.pool.size());
        out.pool.insert(out.pool.end(), group.begin(), group.end());
        // At most c_max points, so it is a leaf and its seed is never used.
        out.nodes.push_back(
                Node{nullptr, group.size(), 0, child_depth, nd.force_l2});
        group.clear();
    };
    for (int i : merge_order) {
        const uint32_t l = small_children[i];
        const int32_t* c = child_ids + sp.child_off[l];
        const size_t size = sp.child_off[l + 1] - sp.child_off[l];
        if (!group.empty() && group.size() + size > c_max) {
            close_group();
        }
        group.insert(group.end(), c, c + size);
    }
    if (!group.empty()) {
        close_group();
    }
}

void collect_children(
        const LevelContext& ctx,
        const std::vector<Node>& nodes,
        const std::vector<SplitInfo>& splits,
        const int32_t* child_ids,
        std::vector<Node>& next,
        LevelStorage& next_storage) {
    std::vector<SplitOutput> outputs(splits.size());
    std::exception_ptr ex;
#pragma omp parallel for schedule(dynamic)
    for (int64_t s = 0; s < static_cast<int64_t>(splits.size()); s++) {
        try {
            classify_children(
                    ctx.p,
                    nodes[splits[s].node],
                    splits[s],
                    child_ids,
                    outputs[s]);
        } catch (...) {
            omp_capture_exception(ex);
        }
    }
    omp_rethrow_if_exception(ex);
    size_t n_next = next.size();
    for (const SplitOutput& out : outputs) {
        n_next += out.nodes.size();
    }
    next.reserve(n_next);
    next_storage.merge_pools.reserve(outputs.size());
    for (SplitOutput& out : outputs) {
        ctx.stats.n_slice_fallbacks += out.n_slices;
        ctx.stats.n_l2_repartitions += out.n_l2;
        next_storage.merge_pools.push_back(std::move(out.pool));
        const int32_t* pool = next_storage.merge_pools.back().data();
        size_t k = 0;
        for (Node& child : out.nodes) {
            if (child.ids == nullptr) {
                child.ids = pool + out.pool_offsets[k++];
            }
        }
        next.insert(
                next.end(),
                std::make_move_iterator(out.nodes.begin()),
                std::make_move_iterator(out.nodes.end()));
        std::vector<Node>().swap(out.nodes);
        std::vector<size_t>().swap(out.pool_offsets);
    }
}

void run_level(
        const LevelContext& ctx,
        const std::vector<Node>& nodes,
        std::vector<Node>& next,
        LevelStorage& next_storage) {
    LevelPlan plan = plan_level(ctx, nodes);
    const std::unique_ptr<uint16_t[]> assign = assign_leaders(ctx, nodes, plan);
    if (plan.splits.empty()) {
        return;
    }
    const int32_t* child_ids =
            scatter_children(nodes, plan, assign.get(), next_storage);
    collect_children(ctx, nodes, plan.splits, child_ids, next, next_storage);
}

} // namespace

void partition_and_visit(
        const Index& storage,
        const PartitionParams& p,
        uint64_t root_seed,
        const LeafVisitor& on_leaf,
        PartitionStats* stats) {
    check_params(storage, p);
    PartitionStats st;
    const LevelContext ctx{storage, p, on_leaf, st};
    const size_t n = static_cast<size_t>(storage.ntotal);
    if (n > 0) {
        InterruptCallback::check();
        std::unique_ptr<int32_t[]> root_ids(new int32_t[n]);
        for (size_t i = 0; i < n; i++) {
            root_ids[i] = static_cast<int32_t>(i);
        }
        std::vector<Node> root_children;
        LevelStorage root_storage;
        run_level(
                ctx,
                {Node{root_ids.get(), n, root_seed, 0, false}},
                root_children,
                root_storage);
        root_ids.reset();

        size_t w0 = 0;
        while (w0 < root_children.size()) {
            size_t w1 = w0;
            size_t load = 0;
            while (w1 < root_children.size() &&
                   (w1 == w0 ||
                    load + root_children[w1].size <= p.wave_budget)) {
                load += root_children[w1].size;
                w1++;
            }
            st.n_waves++;
            std::vector<Node> nodes(
                    root_children.begin() + w0, root_children.begin() + w1);
            std::unique_ptr<LevelStorage> level_storage;
            while (!nodes.empty()) {
                InterruptCallback::check();
                std::vector<Node> next;
                auto next_storage = std::make_unique<LevelStorage>();
                run_level(ctx, nodes, next, *next_storage);
                nodes = std::move(next);
                level_storage =
                        std::move(next_storage); // frees the parent lists
            }
            w0 = w1;
        }
    }
    if (stats) {
        *stats = std::move(st);
    }
}

} // namespace pipnn
} // namespace faiss
