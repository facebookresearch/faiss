#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/impl/IDSelector.h>
#include <faiss/index_factory.h>

#include <faiss/cppcontrib/knowhere/utils/Bitset.h>
#include <faiss/cppcontrib/knowhere/IndexHNSWWrapper.cpp>

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <numeric>
#include <random>
#include <unordered_set>
#include <vector>

std::vector<float> generate_dataset(
        const size_t n,
        const size_t d,
        uint64_t seed) {
    std::default_random_engine rng(seed);
    std::uniform_real_distribution<float> u(-1, 1);

    std::vector<float> data(n * d);
    for (size_t i = 0; i < data.size(); i++) {
        data[i] = u(rng);
    }

    return data;
}

float get_recall_rate(
        const size_t nq,
        const size_t k,
        const std::vector<faiss::idx_t>& baseline,
        const std::vector<faiss::idx_t>& candidate) {
    size_t n = 0;
    for (size_t i = 0; i < nq; i++) {
        std::unordered_set<faiss::idx_t> a_set(k * 4);

        for (size_t j = 0; j < k; j++) {
            a_set.insert(baseline[i * k + j]);
        }

        for (size_t j = 0; j < k; j++) {
            auto itr = a_set.find(candidate[i * k + j]);
            if (itr != a_set.cend()) {
                n += 1;
            }
        }
    }

    return (float)n / candidate.size();
}

struct StopWatch {
    using timepoint_t = std::chrono::time_point<std::chrono::steady_clock>;
    timepoint_t Start;

    StopWatch() {
        Start = std::chrono::steady_clock::now();
    }

    double elapsed() const {
        const auto now = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = now - Start;
        return elapsed.count();
    }
};

void test(const size_t nt, const size_t d, const size_t nq, const size_t k) {
    // generate a dataset for train
    std::vector<float> xt = generate_dataset(nt, d, 123);

    // create an baseline
    std::unique_ptr<faiss::Index> baseline_index(
            faiss::index_factory(d, "Flat", faiss::MetricType::METRIC_L2));
    baseline_index->train(nt, xt.data());
    baseline_index->add(nt, xt.data());

    // create an hnsw index
    std::unique_ptr<faiss::Index> hnsw_index(faiss::index_factory(
            d, "HNSW32,Flat", faiss::MetricType::METRIC_L2));
    hnsw_index->train(nt, xt.data());
    hnsw_index->add(nt, xt.data());

    // generate a query dataset
    std::vector<float> xq = generate_dataset(nq, d, 123);

    // a seed
    std::default_random_engine rng(789);

    // print header
    printf("d=%zd, nt=%zd, nq=%zd\n", d, nt, nq);

    // perform evaluation with a different level of filtering
    for (const size_t percent :
         {0, 1, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99}) {
        // generate a bitset with a given percentage
        std::vector<size_t> ids_to_use(nt);
        std::iota(ids_to_use.begin(), ids_to_use.end(), 0);

        std::shuffle(ids_to_use.begin(), ids_to_use.end(), rng);

        // number of points to use
        const size_t nt_real =
                size_t(std::max(1.0, nt - (nt * percent / 100.0)));

        // create a bitset
        faiss::cppcontrib::knowhere::Bitset bitset =
                faiss::cppcontrib::knowhere::Bitset::create_cleared(nt);
        for (size_t i = 0; i < nt_real; i++) {
            bitset.set(ids_to_use[i]);
        }

        // create an IDSelector
        faiss::IDSelectorBitmap sel(nt, bitset.bits.get());

        // the quant of a search
        const size_t nbatch = nq;

        // perform a baseline search
        std::vector<float> baseline_dis(k * nq, -1);
        std::vector<faiss::idx_t> baseline_ids(k * nq, -1);

        faiss::SearchParameters baseline_params;
        baseline_params.sel = &sel;

        StopWatch sw_baseline;
        for (size_t p = 0; p < nq; p += nbatch) {
            size_t p0 = std::min(nq, p + nbatch);
            size_t np = p0 - p;

            baseline_index->search(
                    np,
                    xq.data() + p * d,
                    k,
                    baseline_dis.data() + k * p,
                    baseline_ids.data() + k * p,
                    &baseline_params);
        }
        double baseline_elapsed = sw_baseline.elapsed();

        // perform an hnsw search
        std::vector<float> hnsw_dis(k * nq, -1);
        std::vector<faiss::idx_t> hnsw_ids(k * nq, -1);

        faiss::SearchParametersHNSW hnsw_params;
        hnsw_params.sel = &sel;
        hnsw_params.efSearch = 64;

        StopWatch sw_hnsw;
        for (size_t p = 0; p < nq; p += nbatch) {
            size_t p0 = std::min(nq, p + nbatch);
            size_t np = p0 - p;

            // hnsw_index->search(nq, xq.data(), k, hnsw_dis.data(),
            // hnsw_ids.data(), &hnsw_params);
            hnsw_index->search(
                    np,
                    xq.data() + p * d,
                    k,
                    hnsw_dis.data() + k * p,
                    hnsw_ids.data() + k * p,
                    &hnsw_params);
        }
        double hnsw_elapsed = sw_hnsw.elapsed();

        // perform a cppcontrib/knowhere search
        std::vector<float> hnsw_candidate_dis(k * nq, -1);
        std::vector<faiss::idx_t> hnsw_candidate_ids(k * nq, -1);

        faiss::cppcontrib::knowhere::SearchParametersHNSWWrapper
                hnsw_candidate_params;
        hnsw_candidate_params.sel = &sel;
        hnsw_candidate_params.kAlpha = ((float)nt_real / nt) * 0.7f;
        hnsw_candidate_params.efSearch = 64;

        faiss::cppcontrib::knowhere::IndexHNSWWrapper wrapper(
                dynamic_cast<faiss::IndexHNSW*>(hnsw_index.get()));

        StopWatch sw_hnsw_candidate;
        for (size_t p = 0; p < nq; p += nbatch) {
            size_t p0 = std::min(nq, p + nbatch);
            size_t np = p0 - p;

            // wrapper.search(nq, xq.data(), k, hnsw_candidate_dis.data(),
            // hnsw_candidate_ids.data(), &hnsw_candidate_params);
            wrapper.search(
                    np,
                    xq.data() + p * d,
                    k,
                    hnsw_candidate_dis.data() + k * p,
                    hnsw_candidate_ids.data() + k * p,
                    &hnsw_candidate_params);
        }
        double hnsw_candidate_elapsed = sw_hnsw_candidate.elapsed();

        // compute the recall rate
        const float recall_hnsw =
                get_recall_rate(nq, k, baseline_ids, hnsw_ids);
        const float recall_hnsw_candidate =
                get_recall_rate(nq, k, baseline_ids, hnsw_candidate_ids);

        printf("perc=%zd, R_baseline=%f, R_candidate=%f, t_baseline=%f ms, t_candidate=%f ms\n",
               percent,
               recall_hnsw,
               recall_hnsw_candidate,
               hnsw_elapsed,
               hnsw_candidate_elapsed);
    }
}

int main() {
    // this takes time to eval
    test(65536, 256, 1024, 64);

    return 0;
}