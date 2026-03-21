#include <gflags/gflags.h>

#include <benchmark/benchmark.h>
#include <faiss/IndexFlat.h>
#include <faiss/utils/distances.h>
#include <faiss/utils/random.h>

using namespace faiss;

DEFINE_uint32(d, 64, "vector dimension");
DEFINE_uint32(nb, 10000, "database size");
DEFINE_uint32(nq, 32, "number of queries");
DEFINE_uint32(k, 10, "k");
DEFINE_uint32(iterations, 20, "iterations");
DEFINE_uint32(threshold, 20, "distance_compute_blas_threshold");
DEFINE_uint32(blas_query_bs, 0, "distance_compute_blas_query_bs");
DEFINE_uint32(blas_database_bs, 0, "distance_compute_blas_database_bs");
DEFINE_bool(ip, false, "use inner product instead of L2");

static void bench_flat_search(benchmark::State& state) {
    int d = static_cast<int>(FLAGS_d);
    idx_t nb = static_cast<idx_t>(FLAGS_nb);
    idx_t nq = static_cast<idx_t>(FLAGS_nq);
    idx_t k = static_cast<idx_t>(FLAGS_k);

    std::vector<float> xb(d * nb);
    std::vector<float> xq(d * nq);
    float_rand(xb.data(), xb.size(), 123);
    float_rand(xq.data(), xq.size(), 456);

    std::unique_ptr<Index> index;
    if (FLAGS_ip) {
        index.reset(new IndexFlatIP(d));
    } else {
        index.reset(new IndexFlatL2(d));
    }
    index->add(nb, xb.data());

    distance_compute_blas_threshold = static_cast<int>(FLAGS_threshold);
    if (FLAGS_blas_query_bs > 0) {
        distance_compute_blas_query_bs =
                static_cast<int>(FLAGS_blas_query_bs);
    }
    if (FLAGS_blas_database_bs > 0) {
        distance_compute_blas_database_bs =
                static_cast<int>(FLAGS_blas_database_bs);
    }

    std::vector<float> distances(nq * k);
    std::vector<idx_t> labels(nq * k);

    for (auto _ : state) {
        index->search(nq, xq.data(), k, distances.data(), labels.data());
    }
}

int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);
    gflags::AllowCommandLineReparsing();
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    benchmark::RegisterBenchmark("flat_search", bench_flat_search)
            ->Iterations(FLAGS_iterations);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
