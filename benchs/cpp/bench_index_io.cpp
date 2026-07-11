/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Index I/O Benchmarks

#include <gflags/gflags.h>
#include <omp.h>
#include <algorithm>
#include <cstdio>
#include <filesystem>

#include <benchmark/benchmark.h>
#include <faiss/IndexFlat.h>
#include <faiss/IndexHNSW.h>
#include <faiss/IndexIVFPQ.h>
#include <faiss/index_io.h>
#include <faiss/utils/AlignedTable.h>
#include <faiss/utils/random.h>

#include "bench_cli_utils.h"
#include "bench_dataset_utils.h"

using namespace faiss;

DEFINE_uint32(iterations, 0, "iterations (0 = auto)");
DEFINE_string(tmpdir, "/tmp", "directory for temporary index files");
BENCH_DEFINE_DATASET_FILE_FLAGS();

static void bench_write_index(
        benchmark::State& state,
        const char* index_type,
        Index* index) {
    std::string fname =
            FLAGS_tmpdir + "/bench_io_" + index_type + ".faissindex";

    for (auto _ : state) {
        write_index(index, fname.c_str());
        benchmark::ClobberMemory();
    }

    // Report file size
    auto fsize = std::filesystem::file_size(fname);
    state.counters["file_size_MB"] = (double)fsize / (1024 * 1024);
    state.SetBytesProcessed(state.iterations() * fsize);

    // Cleanup
    std::filesystem::remove(fname);
}

static void bench_read_index(
        benchmark::State& state,
        const char* index_type,
        Index* index,
        int io_flags) {
    std::string fname =
            FLAGS_tmpdir + "/bench_io_" + index_type + ".faissindex";
    write_index(index, fname.c_str());

    auto fsize = std::filesystem::file_size(fname);

    for (auto _ : state) {
        Index* loaded = read_index(fname.c_str(), io_flags);
        benchmark::DoNotOptimize(loaded->ntotal);
        delete loaded;
    }

    state.counters["file_size_MB"] = (double)fsize / (1024 * 1024);
    state.SetBytesProcessed(state.iterations() * fsize);
    state.counters["mmap"] = (io_flags & IO_FLAG_MMAP) ? 1 : 0;

    std::filesystem::remove(fname);
}

// Dataset variants: same timed write/read, tagged with the real-data
// counter. index_type is a std::string so the registered copy owns the
// temp-file tag.
static void bench_write_index_dataset(
        benchmark::State& state,
        const std::string& index_type,
        Index* index) {
    bench_write_index(state, index_type.c_str(), index);
    state.counters["dataset"] = 1;
}

static void bench_read_index_dataset(
        benchmark::State& state,
        const std::string& index_type,
        Index* index,
        int io_flags) {
    bench_read_index(state, index_type.c_str(), index, io_flags);
    state.counters["dataset"] = 1;
}

int main(int argc, char** argv) {
    benchmarks::bench_init(
            &argc,
            &argv,
            "index serialization benchmarks: write_index and read_index "
            "(standard and mmap) for IVFPQ and HNSW indexes",
            "--tmpdir=/dev/shm --benchmark_filter='read_index/.*'");

    int d = 128;
    int nb = 100000;
    int nt = 50000;

    AlignedTable<float> xt(d * nt);
    AlignedTable<float> xb((size_t)d * nb);
    float_rand(xt.data(), d * nt, 12345);
    float_rand(xb.data(), (size_t)d * nb, 54321);

    omp_set_num_threads(1);

    // Build indexes
    // IVF+PQ
    IndexFlatL2 quantizer(d);
    IndexIVFPQ ivfpq(&quantizer, d, 256, 16, 8);
    ivfpq.verbose = false;
    ivfpq.train(nt, xt.data());
    ivfpq.add(nb, xb.data());

    // HNSW
    IndexHNSWFlat hnsw(d, 32);
    hnsw.verbose = false;
    hnsw.add(nb, xb.data());

    // Register benchmarks
    {
        auto* b = benchmark::RegisterBenchmark(
                "write_index/ivfpq",
                bench_write_index,
                "ivfpq",
                (Index*)&ivfpq);
        if (FLAGS_iterations > 0)
            b->Iterations(FLAGS_iterations);
    }
    {
        auto* b = benchmark::RegisterBenchmark(
                "write_index/hnsw", bench_write_index, "hnsw", (Index*)&hnsw);
        if (FLAGS_iterations > 0)
            b->Iterations(FLAGS_iterations);
    }

    // Read benchmarks (standard)
    {
        auto* b = benchmark::RegisterBenchmark(
                "read_index/ivfpq/standard",
                bench_read_index,
                "ivfpq",
                (Index*)&ivfpq,
                0);
        if (FLAGS_iterations > 0)
            b->Iterations(FLAGS_iterations);
    }
    {
        auto* b = benchmark::RegisterBenchmark(
                "read_index/hnsw/standard",
                bench_read_index,
                "hnsw",
                (Index*)&hnsw,
                0);
        if (FLAGS_iterations > 0)
            b->Iterations(FLAGS_iterations);
    }

    // Read benchmarks (mmap)
    {
        auto* b = benchmark::RegisterBenchmark(
                "read_index/ivfpq/mmap",
                bench_read_index,
                "ivfpq",
                (Index*)&ivfpq,
                IO_FLAG_MMAP);
        if (FLAGS_iterations > 0)
            b->Iterations(FLAGS_iterations);
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
        // Subsample the real train/base sets to the synthetic sizes so the
        // one-time index builds (HNSW in particular) stay cheap; the timed
        // write/read still serializes indexes built from real vectors. The
        // indexes are static so they outlive registration.
        int snt = std::min((int)sift.nt, nt);
        int snb = std::min((int)sift.nb, nb);

        static IndexFlatL2 sift_quantizer(sd);
        static IndexIVFPQ sift_ivfpq(&sift_quantizer, sd, 256, 16, 8);
        sift_ivfpq.verbose = false;
        sift_ivfpq.train(snt, sift.xt.data());
        sift_ivfpq.add(snb, sift.xb.data());

        static IndexHNSWFlat sift_hnsw(sd, 32);
        sift_hnsw.verbose = false;
        sift_hnsw.add(snb, sift.xb.data());

        {
            auto* b = benchmark::RegisterBenchmark(
                    (ds + "/write_index/ivfpq").c_str(),
                    bench_write_index_dataset,
                    ds + "_ivfpq",
                    (Index*)&sift_ivfpq);
            if (FLAGS_iterations > 0)
                b->Iterations(FLAGS_iterations);
        }
        {
            auto* b = benchmark::RegisterBenchmark(
                    (ds + "/write_index/hnsw").c_str(),
                    bench_write_index_dataset,
                    ds + "_hnsw",
                    (Index*)&sift_hnsw);
            if (FLAGS_iterations > 0)
                b->Iterations(FLAGS_iterations);
        }
        {
            auto* b = benchmark::RegisterBenchmark(
                    (ds + "/read_index/ivfpq/standard").c_str(),
                    bench_read_index_dataset,
                    ds + "_ivfpq",
                    (Index*)&sift_ivfpq,
                    0);
            if (FLAGS_iterations > 0)
                b->Iterations(FLAGS_iterations);
        }
        {
            auto* b = benchmark::RegisterBenchmark(
                    (ds + "/read_index/hnsw/standard").c_str(),
                    bench_read_index_dataset,
                    ds + "_hnsw",
                    (Index*)&sift_hnsw,
                    0);
            if (FLAGS_iterations > 0)
                b->Iterations(FLAGS_iterations);
        }
        {
            auto* b = benchmark::RegisterBenchmark(
                    (ds + "/read_index/ivfpq/mmap").c_str(),
                    bench_read_index_dataset,
                    ds + "_ivfpq",
                    (Index*)&sift_ivfpq,
                    IO_FLAG_MMAP);
            if (FLAGS_iterations > 0)
                b->Iterations(FLAGS_iterations);
        }
    }

    benchmark::RunSpecifiedBenchmarks();
    benchmark::Shutdown();
}
