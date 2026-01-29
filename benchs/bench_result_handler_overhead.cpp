/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/AutoTune.h>
#include <faiss/Index.h>
#include <faiss/IndexIVF.h>
#include <faiss/index_factory.h>
#include <faiss/utils/random.h>
#include <faiss/utils/utils.h>
#include <omp.h>

#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
#include <map>

namespace faiss {

namespace {

constexpr int nb = 100000;
constexpr int nq = 1000;
constexpr int nrun = 100;
constexpr float min_run_len_ms = 2000.0;

struct IndexData {
    std::unique_ptr<Index> index;
    std::vector<float> xq;
};

struct BenchmarkResult {
    std::string index_factory;
    int d;
    int k;
    int nprobe;
    double mean_time;
    double std_time;
};

std::pair<double, double> run_search(
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
    std::vector<double> search_times; 
    for (int run = 0; run < nrun; run++) {
        indexIVF_stats.reset();
        data.index->search(
                nq, data.xq.data(), k, distances.data(), labels.data());
        search_times.push_back(indexIVF_stats.search_time);
        if (getmillisecs() - t0 > min_run_len_ms) {
            break;
        }
    }

    // Compute mean and std (in us/query)
    double sum = 0.0;
    for (double t : search_times) {
        sum += t;
    }
    double mean = sum / search_times.size() / nq * 1000.0;

    double sq_sum = 0.0;
    for (double t : search_times) {
        double t_us = t / nq * 1000.0;
        sq_sum += (t_us - mean) * (t_us - mean);
    }
    double std = search_times.size() > 1 ? std::sqrt(sq_sum / (search_times.size() - 1)) : 0.0;

    return {mean, std};
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
    std::vector<int> ks_list = {1, 4, 16};
    std::vector<int> nprobes_list = {1, 4, 16};

    std::map<std::pair<int, int>, std::pair<double, double>> result_map;
    for (const auto& r : results) {        
        result_map[{r.k, r.nprobe}] = {r.mean_time, r.std_time};
    }

    std::cout << "\n" << index_factory << " d=" << d << " (time in us/query, mean ± stddev)\n";
    std::cout << std::string(76, '-') << "\n";

    std::cout << std::setw(8) << "k \\ np"
              << " |";
    for (int np : nprobes_list) {
        std::cout << std::setw(16) << np << " |";
    }
    std::cout << "\n";
    std::cout << std::string(76, '-') << "\n";

    for (int k : ks_list) {
        std::cout << std::setw(8) << k << " |";
        for (int np : nprobes_list) {
            auto it = result_map.find({k, np});
            if (it != result_map.end()) {
                std::ostringstream oss;
                oss << std::fixed << std::setprecision(1)
                    << it->second.first << " ± " << it->second.second;
                std::cout << std::setw(16) << oss.str() << " |";
            } else {
                std::cout << std::setw(16) << "N/A"
                          << " |";
            }
        }
        std::cout << "\n";
    }
}

} // namespace

} // namespace faiss

int main() {

    std::vector<std::pair<int, std::string> > indexes = {        
            {64, "IVF256,SQ4"}, {256, "IVF256,RaBitQ"}, {16, "IVF256,SQfp16"}, // 256 bit types 
            {128, "IVF256,SQ4"}, {512, "IVF256,RaBitQ"}, {32, "IVF256,SQfp16"}, // 512 bit types 
    };
    std::vector<int> ks = {1, 4, 16};
    std::vector<int> nprobes = {1, 4, 16};

    for (const auto p : indexes) {
        std::string index_factory = p.second; 
        int d = p.first; 
        std::cout << "Building " << index_factory << " d=" << d << "..."
                    << std::flush;
        faiss::IndexData data =
                faiss::build_index(d, index_factory.c_str());
        std::cout << " done\n";

        std::vector<faiss::BenchmarkResult> results;
        for (int k : ks) {
            for (int nprobe : nprobes) {
                auto [mean, std] = faiss::run_search(
                        data, d, k, nprobe, index_factory.c_str());
                results.push_back({index_factory, d, k, nprobe, mean, std});
            }

        }
        faiss::print_results_table(index_factory, d, results);
    }

    return 0;
}
