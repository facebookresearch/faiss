/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <gflags/gflags.h>

#include <benchmark/benchmark.h>
#include <faiss/IndexAdditiveQuantizer.h> // @manual=//faiss:faiss_no_multithreading
#include <faiss/utils/random.h> // @manual=//faiss:faiss_no_multithreading

using namespace faiss;
DEFINE_uint32(iterations, 20, "iterations");
DEFINE_uint32(nprobe, 1, "nprobe");
DEFINE_uint32(batch_size, 1, "batch_size");
DEFINE_double(beam_factor, 4.0, "beam factor");

static void bench_search(
        benchmark::State& state,
        int batch_size,
        int nprobe,
        float beam_factor) {
    int d = 512;
    int nt = 2 << 15;
    std::vector<float> xt(d * nt);

    float_rand(xt.data(), d * nt, 12345);
    ResidualCoarseQuantizer rq(d, {16, 8});
    rq.verbose = false;
    rq.train(nt, xt.data());

    std::vector<float> xq(d * batch_size);
    float_rand(xq.data(), d * batch_size, 12345);

    std::vector<float> distances(nprobe * batch_size);
    std::vector<int64_t> clusterIndices(nprobe * batch_size);
    SearchParametersResidualCoarseQuantizer param;
    param.beam_factor = beam_factor;
    for (auto _ : state) {
        rq.search(
                batch_size,
                xq.data(),
                nprobe,
                distances.data(),
                clusterIndices.data(),
                &param);
    }
}

int main(int argc, char** argv) {
    benchmark::Initialize(&argc, argv);
    gflags::AllowCommandLineReparsing();
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    int iterations = FLAGS_iterations;
    int nprobe = FLAGS_nprobe;
    float beam_factor = FLAGS_beam_factor;
    int batch_size = FLAGS_batch_size;
    benchmark::RegisterBenchmark(
            "search", bench_search, batch_size, nprobe, beam_factor)
            ->Iterations(iterations);
    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
