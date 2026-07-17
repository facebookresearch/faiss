/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Generic index_factory train / add / search Benchmarks
//
// Builds an arbitrary index_factory index, times train and add, then sweeps
// search parameters (nprobe, ...) reporting recall@1/10/100, ndis and
// ms/query — with an optional ParameterSpace autotune exploration.
//
// This benchmark fits the Google Benchmark model:
//   * `factory/train/*` and `factory/add/*` cases time building an index from
//     any index_factory key (default sweep: a few representative IVF keys).
//   * `factory/search/*` cases time search over an nprobe sweep and report
//     recall@1/10/100 against exact ground truth (a --db dataset when
//     available, otherwise synthetic data with a computed brute-force GT).
//   * `factory/autotune/*` runs ParameterSpace::explore and reports
//     the best time to reach a recall target.
//
// CLI-overridable options:
//   * --metric=L2|IP selects the metric, threaded through index_factory and
//     ground-truth computation.
//   * search reports recall_at_1 / recall_at_10 / recall_at_100 counters
//     (respecting the available gt_k).
//   * --inter switches autotune to IntersectionCriterion(nq, k) instead of
//     the default OneRecallAtRCriterion.
//   * autotune defaults: --n_autotune=500, --min_test_duration=3.
//   * construction knobs applied to the built index where the type supports
//     them (each guarded by a dynamic_cast so unsupported keys are untouched):
//     --by_residual, --no_precomputed_tables (use_precomputed_table=0),
//     --rq_beam_size (RQ/AQ max_beam_size), --lsq_encode_ils_iters
//     (LSQ encode_ils_iters), and --add_bs (batched add loop).
//   * --db=<name> selects a dataset loaded from --data_dir (sift1M, bigann,
//     deep*, gist, ...); missing files fall back to synthetic data.
//
// Design notes:
//   * defaults: --indexkey sweeps four IVF keys, --db sift1M when present
//     else synthetic, --threads=1.
//   * knobs not included: --RQ_use_beam_LUT, --RQ_train_default,
//     --clustering_niter, --autotune_max / --autotune_range, arbitrary
//     --searchparams lists, and add-time quantizer tuning (IndexRefine
//     k_factor and quantizer nprobe/efSearch boosts).

#include <map>
#include <memory>
#include <sstream>
#include <string>
#include <vector>

#include <gflags/gflags.h>
#include <omp.h>

#include <benchmark/benchmark.h>
#include <faiss/AutoTune.h>
#include <faiss/IVFlib.h>
#include <faiss/IndexAdditiveQuantizer.h>
#include <faiss/IndexAdditiveQuantizerFastScan.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFAdditiveQuantizer.h>
#include <faiss/IndexIVFAdditiveQuantizerFastScan.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/clone_index.h>
#include <faiss/impl/AdditiveQuantizer.h>
#include <faiss/index_factory.h>
#include <faiss/utils/random.h>

#include "bench_cli_utils.h"
#include "bench_dataset_utils.h"

using namespace faiss;

DEFINE_uint32(iterations, 0, "iterations (0 = auto)");
DEFINE_uint32(threads, 1, "number of threads");
BENCH_DEFINE_DATASET_FILE_FLAGS();
DEFINE_uint32(d, 64, "dimension (synthetic data)");
DEFINE_uint32(nb, 100000, "database size (synthetic data)");
DEFINE_uint32(
        nt,
        65536,
        "training set size (synthetic data; default "
        "256*256)");
DEFINE_uint32(nq, 1000, "number of queries (synthetic data)");
DEFINE_int32(k, 100, "number of nearest neighbors");
DEFINE_string(
        indexkey,
        "",
        "semicolon-separated index_factory keys (';' not ',' because factory "
        "keys themselves contain commas) "
        "(default: 'IVF256,Flat;IVF256,PQ8;IVF256,SQ8;IVF1024,Flat')");
DEFINE_string(
        nprobe,
        "",
        "comma-separated nprobe values for the search sweep "
        "(default: 1,4,16,64)");
DEFINE_int32(
        n_autotune,
        500,
        "max number of autotune experiments (0 = try all combinations)");
DEFINE_double(
        min_test_duration,
        3.0,
        "min test duration (s) per autotune operating point");
DEFINE_string(metric, "L2", "metric type: L2 or IP (inner product)");
DEFINE_string(
        db,
        "",
        "dataset name to load from --data_dir (e.g. sift1M, bigann, deep1M, "
        "gist); empty keeps the SIFT1M-if-present-else-synthetic default");
DEFINE_bool(
        inter,
        false,
        "use IntersectionCriterion(nq, k) instead of OneRecallAtRCriterion "
        "in autotune");
// Construction knobs (AQ/IVF options). Each is applied only where the
// built index type supports it; unsupported keys are left unchanged.
DEFINE_int32(
        by_residual,
        -1,
        "set IVF by_residual (0/1); -1 leaves the index default");
DEFINE_bool(
        no_precomputed_tables,
        false,
        "disable IVFPQ precomputed tables (use_precomputed_table=0)");
DEFINE_int32(
        rq_beam_size,
        -1,
        "RQ/AQ max_beam_size at add time; -1 leaves the default");
DEFINE_int32(
        lsq_encode_ils_iters,
        -1,
        "LSQ encode_ils_iters; -1 leaves the default");
DEFINE_int32(
        add_bs,
        -1,
        "add vectors in batches of this size; -1 adds all at once");

namespace {

// Parse the --metric flag once.
MetricType parse_metric() {
    if (FLAGS_metric == "IP" || FLAGS_metric == "ip") {
        return METRIC_INNER_PRODUCT;
    }
    return METRIC_L2;
}

// Apply the construction knobs to the built index before train/add.
// Each knob is guarded by a dynamic_cast (or field check) so index types
// that do not support it are simply left unchanged.
void apply_construction_options(Index* index) {
    IndexIVF* ivf = ivflib::try_extract_index_ivf(index);

    if (FLAGS_by_residual != -1 && ivf != nullptr) {
        ivf->by_residual = (FLAGS_by_residual == 1);
    }

    if (FLAGS_no_precomputed_tables) {
        if (auto* ivfpq = dynamic_cast<IndexIVFPQ*>(ivf)) {
            ivfpq->use_precomputed_table = 0;
        }
    }

    // Reach the AdditiveQuantizer subquantizer for RQ/LSQ knobs, covering both
    // the IVF-AQ and flat-AQ (and their FastScan) index families.
    AdditiveQuantizer* aq = nullptr;
    if (auto* ivfaq = dynamic_cast<IndexIVFAdditiveQuantizer*>(index)) {
        aq = ivfaq->aq;
    } else if (
            auto* ivfaqfs =
                    dynamic_cast<IndexIVFAdditiveQuantizerFastScan*>(index)) {
        aq = ivfaqfs->aq;
    } else if (auto* iaq = dynamic_cast<IndexAdditiveQuantizer*>(index)) {
        aq = iaq->aq;
    } else if (
            auto* iaqfs =
                    dynamic_cast<IndexAdditiveQuantizerFastScan*>(index)) {
        aq = iaqfs->aq;
    }

    if (FLAGS_rq_beam_size != -1) {
        if (auto* rq = dynamic_cast<ResidualQuantizer*>(aq)) {
            rq->max_beam_size = FLAGS_rq_beam_size;
        }
    }
    if (FLAGS_lsq_encode_ils_iters != -1) {
        if (auto* lsq = dynamic_cast<LocalSearchQuantizer*>(aq)) {
            lsq->encode_ils_iters = FLAGS_lsq_encode_ils_iters;
        }
    }
}

// Add `nb` vectors, optionally in batches of --add_bs.
void add_vectors(Index* index, int nb, const float* xb) {
    if (FLAGS_add_bs <= 0) {
        index->add(nb, xb);
        return;
    }
    for (int i0 = 0; i0 < nb; i0 += FLAGS_add_bs) {
        int i1 = std::min(nb, i0 + FLAGS_add_bs);
        index->add(i1 - i0, xb + (size_t)i0 * index->d);
    }
}

// A dataset for the benchmark: either SIFT1M or synthetic. Ground truth is
// the exact k-NN, needed to report recall.
struct BenchData {
    int d = 0;
    int nt = 0, nb = 0, nq = 0;
    int gt_k = 0;
    std::vector<float> xt, xb, xq;
    std::vector<int64_t> gt; // [nq x gt_k]
    std::string tag;         // "sift1m" or "synthetic"
};

// Compute exact k-NN ground truth with a flat index (metric-aware).
void compute_ground_truth(BenchData& data, int gt_k) {
    IndexFlat flat(data.d, parse_metric());
    flat.add(data.nb, data.xb.data());
    std::vector<float> gt_d(data.nq * gt_k);
    data.gt.resize((size_t)data.nq * gt_k);
    flat.search(data.nq, data.xq.data(), gt_k, gt_d.data(), data.gt.data());
    data.gt_k = gt_k;
}

// Recall@k against exact ground truth (int64 gt variant).
double recall_at(
        const int64_t* labels,
        const int64_t* gt,
        int nq,
        int k,
        int gt_k) {
    int eval_k = std::min(k, gt_k);
    size_t n_found = 0;
    for (int i = 0; i < nq; i++) {
        for (int j = 0; j < eval_k; j++) {
            int64_t gt_id = gt[(size_t)i * gt_k + j];
            for (int l = 0; l < k; l++) {
                if (labels[(size_t)i * k + l] == gt_id) {
                    n_found++;
                    break;
                }
            }
        }
    }
    return (double)n_found / ((size_t)nq * eval_k);
}

// R@rank: fraction of queries whose true top-1 neighbor (gt[:, 0]) is
// found within the first `rank` search results. Matches
// bench_index_ivf_factory.py's `(I[:, :rank] == gt[:, :1]).sum() / nq`.
double recall_at_rank(
        const int64_t* labels,
        const int64_t* gt,
        int nq,
        int k,
        int gt_k,
        int rank) {
    int eval_rank = std::min(rank, k);
    size_t n_found = 0;
    for (int i = 0; i < nq; i++) {
        int64_t gt0 = gt[(size_t)i * gt_k];
        for (int l = 0; l < eval_rank; l++) {
            if (labels[(size_t)i * k + l] == gt0) {
                n_found++;
                break;
            }
        }
    }
    return (double)n_found / nq;
}

// Build (train + add) an index for `key`. Cached across the search/autotune
// sweep so we don't rebuild per nprobe.
Index* build_index(const BenchData& data, const std::string& key) {
    static std::map<std::string, std::unique_ptr<Index>> cache;
    auto it = cache.find(key + "@" + data.tag);
    if (it == cache.end()) {
        std::unique_ptr<Index> index(
                index_factory(data.d, key.c_str(), parse_metric()));
        apply_construction_options(index.get());
        omp_set_num_threads(FLAGS_threads);
        index->train(data.nt, data.xt.data());
        add_vectors(index.get(), data.nb, data.xb.data());
        it = cache.emplace(key + "@" + data.tag, std::move(index)).first;
    }
    return it->second.get();
}

} // namespace

// factory/train — time index_factory + train (fresh index each iteration).
static void bench_factory_train(
        benchmark::State& state,
        const BenchData* data,
        std::string key) {
    omp_set_num_threads(FLAGS_threads);
    for (auto _ : state) {
        std::unique_ptr<Index> index(
                index_factory(data->d, key.c_str(), parse_metric()));
        apply_construction_options(index.get());
        index->train(data->nt, data->xt.data());
        benchmark::DoNotOptimize(index.get());
    }
    state.counters["d"] = data->d;
    state.counters["nt"] = data->nt;
    state.counters["threads"] = FLAGS_threads;
}

// factory/add — time add() into a freshly trained index.
static void bench_factory_add(
        benchmark::State& state,
        const BenchData* data,
        std::string key) {
    omp_set_num_threads(FLAGS_threads);
    // Train once (untimed); clone/re-add per iteration to measure add only.
    std::unique_ptr<Index> trained(
            index_factory(data->d, key.c_str(), parse_metric()));
    apply_construction_options(trained.get());
    trained->train(data->nt, data->xt.data());

    for (auto _ : state) {
        state.PauseTiming();
        std::unique_ptr<Index> index(clone_index(trained.get()));
        state.ResumeTiming();
        add_vectors(index.get(), data->nb, data->xb.data());
        benchmark::DoNotOptimize(index.get());
    }
    state.SetItemsProcessed(state.iterations() * data->nb);
    state.counters["d"] = data->d;
    state.counters["nb"] = data->nb;
    state.counters["threads"] = FLAGS_threads;
}

// factory/search — time search at a fixed nprobe and report recall/ndis.
static void bench_factory_search(
        benchmark::State& state,
        const BenchData* data,
        std::string key,
        int nprobe,
        int k) {
    Index* index = build_index(*data, key);
    omp_set_num_threads(FLAGS_threads);
    if (IndexIVF* ivf = ivflib::try_extract_index_ivf(index)) {
        ivf->nprobe = nprobe;
    }

    std::vector<float> distances((size_t)data->nq * k);
    std::vector<int64_t> labels((size_t)data->nq * k);

    // Warmup
    index->search(
            data->nq, data->xq.data(), k, distances.data(), labels.data());

    indexIVF_stats.reset();
    for (auto _ : state) {
        index->search(
                data->nq, data->xq.data(), k, distances.data(), labels.data());
        benchmark::DoNotOptimize(distances[0]);
    }
    state.SetItemsProcessed(state.iterations() * data->nq);
    state.counters["d"] = data->d;
    state.counters["nb"] = data->nb;
    state.counters["nq"] = data->nq;
    state.counters["k"] = k;
    state.counters["nprobe"] = nprobe;
    state.counters["ndis"] = indexIVF_stats.ndis;
    state.counters["threads"] = FLAGS_threads;
    state.counters["recall"] =
            recall_at(labels.data(), data->gt.data(), data->nq, k, data->gt_k);
    // R@1 / R@10 / R@100 (respecting gt_k and k).
    state.counters["recall_at_1"] = recall_at_rank(
            labels.data(), data->gt.data(), data->nq, k, data->gt_k, 1);
    state.counters["recall_at_10"] = recall_at_rank(
            labels.data(), data->gt.data(), data->nq, k, data->gt_k, 10);
    state.counters["recall_at_100"] = recall_at_rank(
            labels.data(), data->gt.data(), data->nq, k, data->gt_k, 100);
}

// factory/autotune — explore operating points (ParameterSpace) and report
// the fastest one meeting a recall target. This mirrors the
// --searchparams=autotune path (1-recall@1 criterion).
static void bench_factory_autotune(
        benchmark::State& state,
        const BenchData* data,
        std::string key,
        int k) {
    Index* index = build_index(*data, key);
    omp_set_num_threads(FLAGS_threads);

    ParameterSpace ps;
    ps.initialize(index);
    ps.verbose = 0;
    ps.n_experiments = FLAGS_n_autotune;
    ps.min_test_duration = FLAGS_min_test_duration;

    // Default optimizes 1-recall@1; --inter switches to intersection@k.
    std::unique_ptr<AutoTuneCriterion> crit_holder;
    if (FLAGS_inter) {
        crit_holder = std::make_unique<IntersectionCriterion>(data->nq, k);
    } else {
        crit_holder = std::make_unique<OneRecallAtRCriterion>(data->nq, 1);
    }
    AutoTuneCriterion& crit = *crit_holder;
    crit.nnn = k;
    crit.set_groundtruth(data->gt_k, nullptr, data->gt.data());

    // The exploration itself is the timed work (it runs many searches).
    double best_perf = 0.0;
    double best_t = 0.0;
    for (auto _ : state) {
        OperatingPoints ops;
        ps.explore(index, data->nq, data->xq.data(), crit, &ops);
        // Best operating point at the highest achieved recall.
        best_perf = 0.0;
        best_t = 0.0;
        for (const auto& op : ops.optimal_pts) {
            if (op.perf > best_perf) {
                best_perf = op.perf;
                best_t = op.t;
            }
        }
        benchmark::DoNotOptimize(best_perf);
    }
    state.counters["d"] = data->d;
    state.counters["nq"] = data->nq;
    state.counters["k"] = k;
    state.counters["n_experiments"] = ps.n_experiments;
    state.counters[FLAGS_inter ? "best_inter" : "best_recall_at_1"] = best_perf;
    state.counters["best_op_ms"] = best_t;
    state.counters["threads"] = FLAGS_threads;
}

int main(int argc, char** argv) {
    benchmarks::bench_init(
            &argc,
            &argv,
            "generic index_factory train/add/search with recall + ndis and "
            "ParameterSpace autotune exploration",
            "--indexkey='IVF1024,Flat' --nprobe=1,16,64 "
            "--benchmark_filter='factory/search/.*'");

    int k = FLAGS_k;
    // Factory keys contain commas, so they are split on ';' rather than
    // routed through the comma-based str_list helper.
    std::vector<std::string> keys;
    if (FLAGS_indexkey.empty()) {
        keys = {"IVF256,Flat", "IVF256,PQ8", "IVF256,SQ8", "IVF1024,Flat"};
    } else {
        std::stringstream ss(FLAGS_indexkey);
        std::string token;
        while (std::getline(ss, token, ';')) {
            if (!token.empty())
                keys.push_back(token);
        }
    }
    std::vector<int> nprobes =
            benchmarks::int_list(FLAGS_nprobe, {1, 4, 16, 64});

    // Choose the dataset once. Precedence:
    //   1. --db=<name>: load {train,base,query,gt} from --data_dir (generic
    //      .fvecs/.bvecs loader). Falls through to synthetic if files absent.
    //   2. otherwise SIFT1M if present in --data_dir.
    //   3. otherwise synthetic data with a computed brute-force ground truth.
    static BenchData data;
    static benchmarks::GenericDataset generic;
    static benchmarks::DatasetSIFT1M sift;
    bool loaded = false;
    if (!FLAGS_db.empty() && benchmarks::dataset_available(FLAGS_data_dir) &&
        generic.load(FLAGS_data_dir, FLAGS_db)) {
        data.d = (int)generic.d;
        data.nt = (int)generic.nt;
        data.nb = (int)generic.nb;
        data.nq = (int)generic.nq;
        data.xt = std::move(generic.xt);
        data.xb = std::move(generic.xb);
        data.xq = std::move(generic.xq);
        data.tag = FLAGS_db;
        data.gt_k = (int)generic.gt_k;
        data.gt.assign(generic.gt.begin(), generic.gt.end());
        loaded = true;
    } else if (
            benchmarks::dataset_available(FLAGS_data_dir) &&
            sift.load(
                    FLAGS_data_dir,
                    FLAGS_train_file,
                    FLAGS_base_file,
                    FLAGS_query_file,
                    FLAGS_gt_file)) {
        data.d = (int)sift.d;
        data.nt = (int)sift.nt;
        data.nb = (int)sift.nb;
        data.nq = (int)sift.nq;
        data.xt = std::move(sift.xt);
        data.xb = std::move(sift.xb);
        data.xq = std::move(sift.xq);
        // Keep the historical "sift1m" label for the default SIFT files;
        // otherwise derive a tag from the user-specified base filename.
        data.tag = benchmarks::dataset_label(FLAGS_base_file);
        // sift ground truth is int32 with gt_k neighbors; widen to int64.
        data.gt_k = (int)sift.gt_k;
        data.gt.assign(sift.gt.begin(), sift.gt.end());
        loaded = true;
    }
    if (!loaded) {
        data.d = FLAGS_d;
        data.nt = FLAGS_nt;
        data.nb = FLAGS_nb;
        data.nq = FLAGS_nq;
        data.tag = "synthetic";
        data.xt.resize((size_t)data.d * data.nt);
        data.xb.resize((size_t)data.d * data.nb);
        data.xq.resize((size_t)data.d * data.nq);
        rand_smooth_vectors(data.nt, data.d, data.xt.data(), 1234);
        rand_smooth_vectors(data.nb, data.d, data.xb.data(), 4567);
        rand_smooth_vectors(data.nq, data.d, data.xq.data(), 7890);
        compute_ground_truth(data, std::max(k, 100));
    }
    // Cap k at the number of ground-truth neighbors available.
    if (k > data.gt_k)
        k = data.gt_k;

    auto register_case = [&](const std::string& name, auto fn, auto... a) {
        auto* b = benchmark::RegisterBenchmark(name.c_str(), fn, a...);
        if (FLAGS_iterations > 0)
            b->Iterations(FLAGS_iterations);
    };

    for (const std::string& key : keys) {
        // Sanitize the key for use in a benchmark name (commas confuse the
        // --benchmark_filter regex less than they confuse humans, but keep
        // the original for the factory call).
        std::string safe = key;
        for (char& c : safe) {
            if (c == ',')
                c = '_';
        }

        register_case("factory/train/" + safe, bench_factory_train, &data, key);
        register_case("factory/add/" + safe, bench_factory_add, &data, key);
        for (int nprobe : nprobes) {
            register_case(
                    "factory/search/" + safe +
                            "/nprobe:" + std::to_string(nprobe),
                    bench_factory_search,
                    &data,
                    key,
                    nprobe,
                    k);
        }
        register_case(
                "factory/autotune/" + safe,
                bench_factory_autotune,
                &data,
                key,
                k);
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
