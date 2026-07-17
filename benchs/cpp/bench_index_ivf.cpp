/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// IVF Family Benchmarks (IVFFlat, IVFPQ, IVFScalarQuantizer)
//
// Design notes:
//   * The default ivfsq_rangestat group uses QT_8bit with one representative
//     arg per rangestat on synthetic d=128/nb=100k data. The full grids
//     remain available via --rangestat_sq_type, --rangestat and
//     --rangestat_arg.

#include <map>
#include <memory>
#include <tuple>

#include <gflags/gflags.h>
#include <omp.h>

#include <benchmark/benchmark.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexIVF.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/IndexScalarQuantizer.h>
#include <faiss/utils/AlignedTable.h>
#include <faiss/utils/random.h>

#include "bench_cli_utils.h"
#include "bench_dataset_utils.h"

using namespace faiss;

DEFINE_uint32(iterations, 0, "iterations (0 = auto)");
DEFINE_uint32(threads, 1, "number of threads");
BENCH_DEFINE_DATASET_FILE_FLAGS();
DEFINE_uint32(d, 128, "dimension");
DEFINE_uint32(nb, 100000, "database size");
DEFINE_string(nq, "", "comma-separated query batch sizes (default: 1,10,100)");
DEFINE_string(nlist, "", "comma-separated IVF list counts (default: 256,1024)");
DEFINE_string(
        nprobe,
        "",
        "comma-separated nprobe values (default: 1,8,32; "
        "sift1m ivfflat group: 1,16,64; sift1m ivfpq group: 1,16)");
DEFINE_string(
        M,
        "",
        "comma-separated PQ subquantizer counts "
        "(default: 8,16,32; sift1m ivfpq group: 8,16)");
DEFINE_string(
        precomp,
        "",
        "comma-separated 0/1 values for IVFPQ precomputed tables "
        "(default: 0,1)");
DEFINE_string(
        sq_type,
        "",
        "comma-separated scalar quantizer types "
        "(default: QT_8bit,QT_fp16,QT_4bit,QT_bf16)");
DEFINE_string(
        rangestat,
        "",
        "comma-separated scalar quantizer range statistics for the "
        "ivfsq_rangestat group "
        "(default: RS_minmax,RS_meanstd,RS_quantiles,RS_optim)");
DEFINE_string(
        rangestat_sq_type,
        "",
        "comma-separated scalar quantizer types for the ivfsq_rangestat group "
        "(default: QT_8bit)");
DEFINE_string(
        rangestat_arg,
        "",
        "comma-separated rangestat_arg override for the ivfsq_rangestat group "
        "(default: representative arg per rangestat)");

static void bench_ivfflat_search(
        benchmark::State& state,
        int d,
        int nb,
        int nlist,
        int nprobe,
        int nq,
        int k) {
    int nt = std::min(nb, 50000);
    AlignedTable<float> xt(d * nt);
    AlignedTable<float> xb((size_t)d * nb);
    AlignedTable<float> xq(d * nq);
    float_rand(xt.data(), d * nt, 12345);
    float_rand(xb.data(), (size_t)d * nb, 54321);
    float_rand(xq.data(), d * nq, 67890);

    IndexFlatL2 quantizer(d);
    IndexIVFFlat index(&quantizer, d, nlist);
    index.verbose = false;
    omp_set_num_threads(FLAGS_threads);
    index.train(nt, xt.data());
    index.add(nb, xb.data());
    index.nprobe = nprobe;

    std::vector<float> distances(nq * k);
    std::vector<int64_t> labels(nq * k);

    // Warmup
    index.search(nq, xq.data(), k, distances.data(), labels.data());

    indexIVF_stats.reset();
    for (auto _ : state) {
        index.search(nq, xq.data(), k, distances.data(), labels.data());
        benchmark::DoNotOptimize(distances[0]);
    }
    state.SetItemsProcessed(state.iterations() * nq);
    state.counters["d"] = d;
    state.counters["nb"] = nb;
    state.counters["nlist"] = nlist;
    state.counters["nprobe"] = nprobe;
    state.counters["nq"] = nq;
    state.counters["k"] = k;
    state.counters["ndis"] = indexIVF_stats.ndis;
    state.counters["nlist_visited"] = indexIVF_stats.nlist;
    state.counters["threads"] = FLAGS_threads;
}

// SIFT1M variant: accepts external data pointers, builds index once per call
static void bench_ivfflat_search_dataset(
        benchmark::State& state,
        const float* xt,
        int nt,
        const float* xb,
        int nb,
        const float* xq,
        int nq,
        const int32_t* gt,
        size_t gt_k,
        int d,
        int nlist,
        int nprobe,
        int k) {
    IndexFlatL2 quantizer(d);
    IndexIVFFlat index(&quantizer, d, nlist);
    index.verbose = false;
    omp_set_num_threads(FLAGS_threads);
    index.train(nt, xt);
    index.add(nb, xb);
    index.nprobe = nprobe;

    std::vector<float> distances(nq * k);
    std::vector<int64_t> labels(nq * k);

    // Warmup
    index.search(nq, xq, k, distances.data(), labels.data());

    indexIVF_stats.reset();
    for (auto _ : state) {
        index.search(nq, xq, k, distances.data(), labels.data());
        benchmark::DoNotOptimize(distances[0]);
    }
    state.SetItemsProcessed(state.iterations() * nq);
    state.counters["d"] = d;
    state.counters["nb"] = nb;
    state.counters["nlist"] = nlist;
    state.counters["nprobe"] = nprobe;
    state.counters["nq"] = nq;
    state.counters["k"] = k;
    state.counters["dataset"] = 1;
    state.counters["ndis"] = indexIVF_stats.ndis;
    state.counters["threads"] = FLAGS_threads;

    // Compute recall from last iteration's results
    double recall =
            benchmarks::compute_recall_at(labels.data(), gt, nq, k, gt_k);
    state.counters["recall"] = recall;
}

// SIFT1M variant for IVFPQ
static void bench_ivfpq_search_dataset(
        benchmark::State& state,
        const float* xt,
        int nt,
        const float* xb,
        int nb,
        const float* xq,
        int nq,
        const int32_t* gt,
        size_t gt_k,
        int d,
        int nlist,
        int M,
        int nprobe,
        int k) {
    IndexFlatL2 quantizer(d);
    IndexIVFPQ index(&quantizer, d, nlist, M, 8);
    index.verbose = false;
    omp_set_num_threads(FLAGS_threads);
    index.train(nt, xt);
    index.add(nb, xb);
    index.nprobe = nprobe;

    std::vector<float> distances(nq * k);
    std::vector<int64_t> labels(nq * k);

    // Warmup
    index.search(nq, xq, k, distances.data(), labels.data());

    indexIVF_stats.reset();
    for (auto _ : state) {
        index.search(nq, xq, k, distances.data(), labels.data());
        benchmark::DoNotOptimize(distances[0]);
    }
    state.SetItemsProcessed(state.iterations() * nq);
    state.counters["d"] = d;
    state.counters["nb"] = nb;
    state.counters["nlist"] = nlist;
    state.counters["M"] = M;
    state.counters["nprobe"] = nprobe;
    state.counters["nq"] = nq;
    state.counters["k"] = k;
    state.counters["dataset"] = 1;
    state.counters["ndis"] = indexIVF_stats.ndis;
    state.counters["threads"] = FLAGS_threads;

    // Compute recall from last iteration's results
    double recall =
            benchmarks::compute_recall_at(labels.data(), gt, nq, k, gt_k);
    state.counters["recall"] = recall;
}

static void bench_ivfpq_search(
        benchmark::State& state,
        int d,
        int nb,
        int nlist,
        int M,
        int nprobe,
        int nq,
        int k,
        bool use_precomputed) {
    int nt = std::min(nb, 50000);
    AlignedTable<float> xt(d * nt);
    AlignedTable<float> xb((size_t)d * nb);
    AlignedTable<float> xq(d * nq);
    float_rand(xt.data(), d * nt, 12345);
    float_rand(xb.data(), (size_t)d * nb, 54321);
    float_rand(xq.data(), d * nq, 67890);

    IndexFlatL2 quantizer(d);
    IndexIVFPQ index(&quantizer, d, nlist, M, 8);
    index.verbose = false;
    omp_set_num_threads(FLAGS_threads);
    index.train(nt, xt.data());
    index.add(nb, xb.data());
    index.nprobe = nprobe;
    index.use_precomputed_table = use_precomputed ? 1 : 0;

    std::vector<float> distances(nq * k);
    std::vector<int64_t> labels(nq * k);

    // Warmup
    index.search(nq, xq.data(), k, distances.data(), labels.data());

    indexIVF_stats.reset();
    for (auto _ : state) {
        index.search(nq, xq.data(), k, distances.data(), labels.data());
        benchmark::DoNotOptimize(distances[0]);
    }
    state.SetItemsProcessed(state.iterations() * nq);
    state.counters["d"] = d;
    state.counters["nb"] = nb;
    state.counters["nlist"] = nlist;
    state.counters["M"] = M;
    state.counters["nprobe"] = nprobe;
    state.counters["nq"] = nq;
    state.counters["k"] = k;
    state.counters["precomputed"] = use_precomputed;
    state.counters["ndis"] = indexIVF_stats.ndis;
    state.counters["threads"] = FLAGS_threads;
}

static void bench_ivfsq_search(
        benchmark::State& state,
        int d,
        int nb,
        int nlist,
        ScalarQuantizer::QuantizerType sq_type,
        int nprobe,
        int nq,
        int k) {
    int nt = std::min(nb, 50000);
    AlignedTable<float> xt(d * nt);
    AlignedTable<float> xb((size_t)d * nb);
    AlignedTable<float> xq(d * nq);
    float_rand(xt.data(), d * nt, 12345);
    float_rand(xb.data(), (size_t)d * nb, 54321);
    float_rand(xq.data(), d * nq, 67890);

    IndexFlatL2 quantizer(d);
    IndexIVFScalarQuantizer index(&quantizer, d, nlist, sq_type);
    index.verbose = false;
    omp_set_num_threads(FLAGS_threads);
    index.train(nt, xt.data());
    index.add(nb, xb.data());
    index.nprobe = nprobe;

    std::vector<float> distances(nq * k);
    std::vector<int64_t> labels(nq * k);

    // Warmup
    index.search(nq, xq.data(), k, distances.data(), labels.data());

    indexIVF_stats.reset();
    for (auto _ : state) {
        index.search(nq, xq.data(), k, distances.data(), labels.data());
        benchmark::DoNotOptimize(distances[0]);
    }
    state.SetItemsProcessed(state.iterations() * nq);
    state.counters["d"] = d;
    state.counters["nb"] = nb;
    state.counters["nlist"] = nlist;
    state.counters["nprobe"] = nprobe;
    state.counters["nq"] = nq;
    state.counters["k"] = k;
    state.counters["ndis"] = indexIVF_stats.ndis;
    state.counters["threads"] = FLAGS_threads;
}

// Mirror of benchs/python/bench_index_ivf.py::bench_ivfsq_rangestat_search —
// the rangestat calibration sweep for IVFScalarQuantizer. Reproduces the
// sweep: every QT_* quantizer type crossed
// with a full rangestat_arg grid per rangestat, ncent=256, nprobe=16, k=100.
// rangestat affects training, so it is set BEFORE train and the trained
// index is cached once per (nlist, sq_type, rangestat, arg) and reused across
// the nq sweep.
static IndexIVFScalarQuantizer* ivfsq_rangestat_index(
        int d,
        int nb,
        int nlist,
        ScalarQuantizer::QuantizerType qtype,
        ScalarQuantizer::RangeStat rangestat,
        float rangestat_arg) {
    struct Key {
        int nlist;
        int qtype;
        int rangestat;
        int arg_bits;
        bool operator<(const Key& o) const {
            return std::tie(nlist, qtype, rangestat, arg_bits) <
                    std::tie(o.nlist, o.qtype, o.rangestat, o.arg_bits);
        }
    };
    static std::map<Key, std::unique_ptr<IndexIVFScalarQuantizer>> cache;
    // Fold the float arg into the key by its bit pattern so distinct args
    // train distinct cached indexes.
    int arg_bits;
    memcpy(&arg_bits, &rangestat_arg, sizeof(arg_bits));
    Key key{nlist, (int)qtype, (int)rangestat, arg_bits};
    auto it = cache.find(key);
    if (it == cache.end()) {
        int nt = std::min(nb, 50000);
        AlignedTable<float> xt(d * nt);
        AlignedTable<float> xb((size_t)d * nb);
        float_rand(xt.data(), d * nt, 12345);
        float_rand(xb.data(), (size_t)d * nb, 54321);

        auto quantizer = new IndexFlatL2(d);
        auto index = std::make_unique<IndexIVFScalarQuantizer>(
                quantizer, d, nlist, qtype);
        index->own_fields = true;
        index->verbose = false;
        // rangestat affects training: set before train
        index->sq.rangestat = rangestat;
        index->sq.rangestat_arg = rangestat_arg;
        omp_set_num_threads(FLAGS_threads);
        index->train(nt, xt.data());
        index->add(nb, xb.data());
        it = cache.emplace(key, std::move(index)).first;
    }
    return it->second.get();
}

static void bench_ivfsq_rangestat_search(
        benchmark::State& state,
        int d,
        int nb,
        int nlist,
        ScalarQuantizer::QuantizerType qtype,
        ScalarQuantizer::RangeStat rangestat,
        float rangestat_arg,
        int nprobe,
        int nq,
        int k) {
    IndexIVFScalarQuantizer* index = ivfsq_rangestat_index(
            d, nb, nlist, qtype, rangestat, rangestat_arg);
    omp_set_num_threads(FLAGS_threads);
    index->nprobe = nprobe;

    AlignedTable<float> xq(d * nq);
    float_rand(xq.data(), d * nq, 67890);

    std::vector<float> distances(nq * k);
    std::vector<int64_t> labels(nq * k);

    // Warmup
    index->search(nq, xq.data(), k, distances.data(), labels.data());

    indexIVF_stats.reset();
    for (auto _ : state) {
        index->search(nq, xq.data(), k, distances.data(), labels.data());
        benchmark::DoNotOptimize(distances[0]);
    }
    state.SetItemsProcessed(state.iterations() * nq);
    state.counters["d"] = d;
    state.counters["nb"] = nb;
    state.counters["nlist"] = nlist;
    state.counters["rangestat_arg"] = rangestat_arg;
    state.counters["nprobe"] = nprobe;
    state.counters["nq"] = nq;
    state.counters["k"] = k;
    state.counters["ndis"] = indexIVF_stats.ndis;
    state.counters["threads"] = FLAGS_threads;
}

int main(int argc, char** argv) {
    benchmarks::bench_init(
            &argc,
            &argv,
            "IVF-family index search: IVFFlat, IVFPQ and IVFScalarQuantizer "
            "on synthetic data and SIFT1M",
            "--nlist=1024 --nprobe=16 --benchmark_filter='ivfflat/search/.*'");

    int d = FLAGS_d;
    int nb = FLAGS_nb;
    int k = 10;
    std::vector<int> nqs = benchmarks::int_list(FLAGS_nq, {1, 10, 100});
    std::vector<int> nlists = benchmarks::int_list(FLAGS_nlist, {256, 1024});
    std::vector<int> nprobes = benchmarks::int_list(FLAGS_nprobe, {1, 8, 32});

    // IVFFlat
    for (int nlist : nlists) {
        for (int nprobe : nprobes) {
            if (nprobe > nlist)
                continue;
            for (int nq : nqs) {
                std::string name =
                        "ivfflat/search/nlist:" + std::to_string(nlist) +
                        "/nprobe:" + std::to_string(nprobe) +
                        "/nq:" + std::to_string(nq);
                auto* b = benchmark::RegisterBenchmark(
                        name.c_str(),
                        bench_ivfflat_search,
                        d,
                        nb,
                        nlist,
                        nprobe,
                        nq,
                        k);
                if (FLAGS_iterations > 0)
                    b->Iterations(FLAGS_iterations);
            }
        }
    }

    // IVFPQ
    std::vector<int> Ms = benchmarks::int_list(FLAGS_M, {8, 16, 32});
    std::vector<int> precomps = benchmarks::int_list(FLAGS_precomp, {0, 1});
    for (int nlist : nlists) {
        for (int M : Ms) {
            for (int nprobe : nprobes) {
                if (nprobe > nlist)
                    continue;
                for (int nq : nqs) {
                    for (int precomp_int : precomps) {
                        bool precomp = (bool)precomp_int;
                        std::string name =
                                "ivfpq/search/nlist:" + std::to_string(nlist) +
                                "/M:" + std::to_string(M) +
                                "/nprobe:" + std::to_string(nprobe) +
                                "/nq:" + std::to_string(nq) +
                                "/precomp:" + std::to_string(precomp);
                        auto* b = benchmark::RegisterBenchmark(
                                name.c_str(),
                                bench_ivfpq_search,
                                d,
                                nb,
                                nlist,
                                M,
                                nprobe,
                                nq,
                                k,
                                precomp);
                        if (FLAGS_iterations > 0)
                            b->Iterations(FLAGS_iterations);
                    }
                }
            }
        }
    }

    // IVFScalarQuantizer
    struct SQDef {
        const char* name;
        ScalarQuantizer::QuantizerType type;
    };
    std::vector<SQDef> sq_defs = {
            {"QT_8bit", ScalarQuantizer::QT_8bit},
            {"QT_fp16", ScalarQuantizer::QT_fp16},
            {"QT_4bit", ScalarQuantizer::QT_4bit},
            {"QT_bf16", ScalarQuantizer::QT_bf16},
    };
    std::vector<std::string> sq_names = benchmarks::str_list(
            FLAGS_sq_type, {"QT_8bit", "QT_fp16", "QT_4bit", "QT_bf16"});
    std::vector<SQDef> sq_types;
    for (const std::string& sq_name : sq_names) {
        bool found = false;
        for (auto& def : sq_defs) {
            if (sq_name == def.name) {
                sq_types.push_back(def);
                found = true;
                break;
            }
        }
        if (!found) {
            fprintf(stderr,
                    "unknown scalar quantizer type '%s' "
                    "(expected QT_8bit, QT_fp16, QT_4bit or QT_bf16)\n",
                    sq_name.c_str());
            return 1;
        }
    }

    for (int nlist : nlists) {
        for (auto& sq : sq_types) {
            for (int nprobe : nprobes) {
                if (nprobe > nlist)
                    continue;
                for (int nq : nqs) {
                    std::string name = std::string("ivfsq/search/") + sq.name +
                            "/nlist:" + std::to_string(nlist) +
                            "/nprobe:" + std::to_string(nprobe) +
                            "/nq:" + std::to_string(nq);
                    auto* b = benchmark::RegisterBenchmark(
                            name.c_str(),
                            bench_ivfsq_search,
                            d,
                            nb,
                            nlist,
                            sq.type,
                            nprobe,
                            nq,
                            k);
                    if (FLAGS_iterations > 0)
                        b->Iterations(FLAGS_iterations);
                }
            }
        }
    }

    // IVFScalarQuantizer rangestat sweep (mirrors
    // bench_index_ivf.py::bench_ivfsq_rangestat_search). Crosses every QT_*
    // type with a full rangestat_arg grid per rangestat at ncent=256,
    // nprobe=16, k=100. The default here fixes QT_8bit and one representative
    // arg per rangestat to stay tractable; --rangestat_sq_type walks all types
    // and --rangestat_arg walks the full calibration curve.
    const int rs_k = 100; // searches k=100
    struct RangeStatDef {
        const char* name;
        ScalarQuantizer::RangeStat rangestat;
        float default_arg; // representative arg
    };
    std::vector<RangeStatDef> rangestat_defs = {
            {"RS_minmax", ScalarQuantizer::RS_minmax, 0.0f},
            {"RS_meanstd", ScalarQuantizer::RS_meanstd, 2.0f},
            {"RS_quantiles", ScalarQuantizer::RS_quantiles, 0.1f},
            {"RS_optim", ScalarQuantizer::RS_optim, 0.0f},
    };
    std::vector<std::string> rangestat_names = benchmarks::str_list(
            FLAGS_rangestat,
            {"RS_minmax", "RS_meanstd", "RS_quantiles", "RS_optim"});
    std::vector<RangeStatDef> rangestats;
    for (const std::string& rs_name : rangestat_names) {
        bool found = false;
        for (auto& def : rangestat_defs) {
            if (rs_name == def.name) {
                rangestats.push_back(def);
                found = true;
                break;
            }
        }
        if (!found) {
            fprintf(stderr,
                    "unknown rangestat '%s' (expected RS_minmax, RS_meanstd, "
                    "RS_quantiles or RS_optim)\n",
                    rs_name.c_str());
            return 1;
        }
    }

    // Scalar quantizer types swept by the rangestat group. Default QT_8bit;
    // the full list is available via --rangestat_sq_type.
    struct RSQDef {
        const char* name;
        ScalarQuantizer::QuantizerType type;
    };
    std::vector<RSQDef> rs_sq_defs = {
            {"QT_8bit", ScalarQuantizer::QT_8bit},
            {"QT_4bit", ScalarQuantizer::QT_4bit},
            {"QT_6bit", ScalarQuantizer::QT_6bit},
            {"QT_fp16", ScalarQuantizer::QT_fp16},
            {"QT_bf16", ScalarQuantizer::QT_bf16},
            {"QT_8bit_uniform", ScalarQuantizer::QT_8bit_uniform},
            {"QT_4bit_uniform", ScalarQuantizer::QT_4bit_uniform},
            {"QT_8bit_direct", ScalarQuantizer::QT_8bit_direct},
            {"QT_8bit_direct_signed", ScalarQuantizer::QT_8bit_direct_signed},
    };
    std::vector<std::string> rs_sq_names =
            benchmarks::str_list(FLAGS_rangestat_sq_type, {"QT_8bit"});
    std::vector<RSQDef> rs_sq_types;
    for (const std::string& sq_name : rs_sq_names) {
        bool found = false;
        for (auto& def : rs_sq_defs) {
            if (sq_name == def.name) {
                rs_sq_types.push_back(def);
                found = true;
                break;
            }
        }
        if (!found) {
            fprintf(stderr,
                    "unknown rangestat_sq_type '%s'\n",
                    sq_name.c_str());
            return 1;
        }
    }

    // Optional explicit rangestat_arg override; empty -> per-rangestat default.
    std::vector<float> rs_arg_override;
    if (!FLAGS_rangestat_arg.empty()) {
        std::stringstream ss(FLAGS_rangestat_arg);
        std::string tok;
        while (std::getline(ss, tok, ',')) {
            if (!tok.empty())
                rs_arg_override.push_back(std::stof(tok));
        }
    }

    std::vector<int> rs_nlists = benchmarks::int_list(FLAGS_nlist, {256});
    std::vector<int> rs_nprobes = benchmarks::int_list(FLAGS_nprobe, {16});
    for (auto& rs : rangestats) {
        // Args: explicit override, else the single representative default.
        std::vector<float> args = rs_arg_override.empty()
                ? std::vector<float>{rs.default_arg}
                : rs_arg_override;
        for (auto& sq : rs_sq_types) {
            for (float arg : args) {
                for (int nlist : rs_nlists) {
                    for (int nprobe : rs_nprobes) {
                        if (nprobe > nlist)
                            continue;
                        for (int nq : nqs) {
                            std::string name =
                                    std::string("ivfsq_rangestat/search/") +
                                    sq.name + "/" + rs.name +
                                    "/arg:" + std::to_string(arg).substr(0, 5) +
                                    "/nlist:" + std::to_string(nlist) +
                                    "/nprobe:" + std::to_string(nprobe) +
                                    "/nq:" + std::to_string(nq);
                            auto* b = benchmark::RegisterBenchmark(
                                    name.c_str(),
                                    bench_ivfsq_rangestat_search,
                                    d,
                                    nb,
                                    nlist,
                                    sq.type,
                                    rs.rangestat,
                                    arg,
                                    nprobe,
                                    nq,
                                    rs_k);
                            if (FLAGS_iterations > 0)
                                b->Iterations(FLAGS_iterations);
                        }
                    }
                }
            }
        }
    }

    // SIFT1M-based benchmarks (if dataset available)
    static benchmarks::DatasetSIFT1M sift;
    if (benchmarks::dataset_available(FLAGS_data_dir) &&
        sift.load(
                FLAGS_data_dir,
                FLAGS_train_file,
                FLAGS_base_file,
                FLAGS_query_file,
                FLAGS_gt_file)) {
        const std::string ds = benchmarks::dataset_label(FLAGS_base_file);
        int sd = (int)sift.d;
        int snb = (int)sift.nb;
        int snq = (int)sift.nq;
        int snt = (int)sift.nt;

        // IVFFlat on SIFT1M
        std::vector<int> sift_nprobes =
                benchmarks::int_list(FLAGS_nprobe, {1, 16, 64});
        for (int nlist : nlists) {
            for (int nprobe : sift_nprobes) {
                std::string name = ds +
                        "/ivfflat/search/nlist:" + std::to_string(nlist) +
                        "/nprobe:" + std::to_string(nprobe);
                auto* b = benchmark::RegisterBenchmark(
                        name.c_str(),
                        bench_ivfflat_search_dataset,
                        sift.xt.data(),
                        snt,
                        sift.xb.data(),
                        snb,
                        sift.xq.data(),
                        snq,
                        sift.gt.data(),
                        sift.gt_k,
                        sd,
                        nlist,
                        nprobe,
                        k);
                if (FLAGS_iterations > 0)
                    b->Iterations(FLAGS_iterations);
            }
        }

        // IVFPQ on SIFT1M
        std::vector<int> sift_Ms = benchmarks::int_list(FLAGS_M, {8, 16});
        std::vector<int> sift_pq_nprobes =
                benchmarks::int_list(FLAGS_nprobe, {1, 16});
        for (int nlist : nlists) {
            for (int M : sift_Ms) {
                for (int nprobe : sift_pq_nprobes) {
                    std::string name = ds +
                            "/ivfpq/search/nlist:" + std::to_string(nlist) +
                            "/M:" + std::to_string(M) +
                            "/nprobe:" + std::to_string(nprobe);
                    auto* b = benchmark::RegisterBenchmark(
                            name.c_str(),
                            bench_ivfpq_search_dataset,
                            sift.xt.data(),
                            snt,
                            sift.xb.data(),
                            snb,
                            sift.xq.data(),
                            snq,
                            sift.gt.data(),
                            sift.gt_k,
                            sd,
                            nlist,
                            M,
                            nprobe,
                            k);
                    if (FLAGS_iterations > 0)
                        b->Iterations(FLAGS_iterations);
                }
            }
        }
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
