/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/AutoTune.h>
#include <faiss/Index.h>
#include <faiss/index_factory.h>
#include <faiss/utils/random.h>
#include <faiss/utils/utils.h>
#include <omp.h>

#include <iomanip>
#include <iostream>
#include <map>
#include <memory>
#include <vector>

namespace faiss {

namespace {

constexpr int nb = 100000;
constexpr int nq = 1000;
constexpr int nrun = 5;

struct IndexData {
    std::unique_ptr<Index> index;
    std::vector<float> xq;
};

struct BenchmarkResult {
    std::string index_factory;
    int d;
    int k;
    int nprobe;
    double time_per_query_us;
};

double run_search(
        IndexData& data,
        int d,
        int k,
        int nprobe,
        const char* factory_string) {
    ParameterSpace().set_index_parameter(data.index.get(), "nprobe", nprobe);

    omp_set_num_threads(1);

    std::vector<float> distances(nq * k);
    std::vector<idx_t> labels(nq * k);

    // Warmup
    data.index->search(nq, data.xq.data(), k, distances.data(), labels.data());

    // Timed runs - stop if total time exceeds 2 seconds
    double t0 = getmillisecs();
    int effective_runs = 0;
    for (int run = 0; run < nrun; run++) {
        data.index->search(
                nq, data.xq.data(), k, distances.data(), labels.data());
        effective_runs++;
        if (getmillisecs() - t0 > 2000.0) {
            break;
        }
    }
    double t1 = getmillisecs();

    double time_per_query_us = (t1 - t0) / effective_runs / nq * 1000.0;
    return time_per_query_us;
}

IndexData build_index(int d, const char* factory_string) {
    omp_set_num_threads(32);

    int nt = std::max(nb, 1024);

    std::vector<float> xt(nt * d);
    std::vector<float> xb(nb * d);

    rand_smooth_vectors(nt, d, xt.data(), 12345);
    rand_smooth_vectors(nb, d, xb.data(), 23456);

    IndexData data;
    data.index.reset(index_factory(d, factory_string));
    data.index->train(nt, xt.data());
    data.index->add(nb, xb.data());

    data.xq.resize(nq * d);
    rand_smooth_vectors(nq, d, data.xq.data(), 34567);

    return data;
}

void print_results_table(
        const std::string& index_factory,
        int d,
        const std::vector<BenchmarkResult>& results) {
    std::vector<int> ks_list = {1, 4, 16, 64};
    std::vector<int> nprobes_list = {1, 4, 16, 64};

    std::map<std::pair<int, int>, double> result_map;
    for (const auto& r : results) {
        result_map[{r.k, r.nprobe}] = r.time_per_query_us;
    }

    std::cout << "\n" << index_factory << " d=" << d << " (time in us/query)\n";
    std::cout << std::string(60, '-') << "\n";

    std::cout << std::setw(8) << "k \\ np"
              << " |";
    for (int np : nprobes_list) {
        std::cout << std::setw(10) << np << " |";
    }
    std::cout << "\n";
    std::cout << std::string(60, '-') << "\n";

    for (int k : ks_list) {
        std::cout << std::setw(8) << k << " |";
        for (int np : nprobes_list) {
            auto it = result_map.find({k, np});
            if (it != result_map.end()) {
                std::cout << std::setw(10) << std::fixed << std::setprecision(1)
                          << it->second << " |";
            } else {
                std::cout << std::setw(10) << "N/A"
                          << " |";
            }
        }
        std::cout << "\n";
    }
}

} // namespace

} // namespace faiss

int main() {
    /*
    std::vector<std::string> indexes = {
            "IVF256,SQ4", "IVF256,RaBitQ", "IVF256,SQfp16"};
    std::vector<int> dims = {32, 64, 128};
    std::vector<int> ks = {1, 4, 16, 64};
    std::vector<int> nprobes = {1, 4, 16, 64};
*/
    std::vector<std::string> indexes = {
            "IVF256,SQ4",
    };
    std::vector<int> dims = {32};
    std::vector<int> ks = {1, 4, 16};
    std::vector<int> nprobes = {1, 4, 16};

    for (const auto& index_factory : indexes) {
        for (int d : dims) {
            std::cout << "Building " << index_factory << " d=" << d << "..."
                      << std::flush;
            faiss::IndexData data =
                    faiss::build_index(d, index_factory.c_str());
            std::cout << " done\n";

            std::vector<faiss::BenchmarkResult> results;
            for (int k : ks) {
                for (int nprobe : nprobes) {
                    double time_us = faiss::run_search(
                            data, d, k, nprobe, index_factory.c_str());
                    results.push_back({index_factory, d, k, nprobe, time_us});
                }
            }

            faiss::print_results_table(index_factory, d, results);
        }
    }

    return 0;
}
