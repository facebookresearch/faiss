/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include "faiss/IndexSVS.h"
#include "faiss/Index.h"
#include "faiss/MetricType.h"

#include "svs/core/data.h"
#include "svs/orchestrators/dynamic_vamana.h"

namespace faiss {

namespace detail {
SVSTempDirectory::SVSTempDirectory() {
    root = std::filesystem::temp_directory_path() /
            ("faiss_svs_" + std::to_string(std::rand()));
    config = root / "config";
    graph = root / "graph";
    data = root / "data";

    std::filesystem::create_directories(config);
    std::filesystem::create_directories(graph);
    std::filesystem::create_directories(data);
}

SVSTempDirectory::~SVSTempDirectory() {
    std::error_code ec;
    std::filesystem::remove_all(root, ec); // best-effort cleanup
}

void SVSTempDirectory::write_files_to_stream(std::ostream& out) const {
    for (const auto& dir : {config, graph, data}) {
        const std::string dir_name = dir.filename().string();
        for (const auto& entry : std::filesystem::directory_iterator(dir)) {
            const std::string filename = entry.path().filename().string();

            const uint64_t dir_len = dir_name.size();
            const uint64_t file_len = filename.size();
            const uint64_t file_size = std::filesystem::file_size(entry.path());

            out.write(reinterpret_cast<const char*>(&dir_len), sizeof(dir_len));
            out.write(dir_name.data(), dir_len);

            out.write(
                    reinterpret_cast<const char*>(&file_len), sizeof(file_len));
            out.write(filename.data(), file_len);

            out.write(
                    reinterpret_cast<const char*>(&file_size),
                    sizeof(file_size));

            std::ifstream in(entry.path(), std::ios::binary);
            FAISS_THROW_IF_NOT_MSG(
                    in, "Failed to open temp SVS file for reading");

            out << in.rdbuf();
        }
    }
}

void SVSTempDirectory::write_stream_to_files(std::istream& in) const {
    while (in && in.peek() != EOF) {
        uint64_t dir_len, file_len, file_size;

        in.read(reinterpret_cast<char*>(&dir_len), sizeof(dir_len));
        std::string dir_name(dir_len, '\0');
        in.read(dir_name.data(), dir_len);

        in.read(reinterpret_cast<char*>(&file_len), sizeof(file_len));
        std::string filename(file_len, '\0');
        in.read(filename.data(), file_len);

        in.read(reinterpret_cast<char*>(&file_size), sizeof(file_size));
        std::vector<char> buffer(file_size);
        in.read(buffer.data(), file_size);

        std::filesystem::path base;
        if (dir_name == "config") {
            base = config;
        } else if (dir_name == "graph") {
            base = graph;
        } else if (dir_name == "data") {
            base = data;
        } else {
            FAISS_THROW_IF_NOT_MSG(false, "Unknown SVS subdirectory name");
        }

        std::filesystem::path full_path = base / filename;
        std::ofstream out(full_path, std::ios::binary);
        FAISS_THROW_IF_NOT_MSG(out, "Failed to open temp SVS file for writing");
        out.write(buffer.data(), buffer.size());
    }
}

} // namespace detail

IndexSVS::IndexSVS() : Index{} {}

IndexSVS::IndexSVS(idx_t d, MetricType metric) : Index(d, metric) {
    alpha = metric == METRIC_L2 ? 1.2f : 0.95f;
}

IndexSVS::~IndexSVS() {
    if (impl) {
        delete impl;
    }
}

void IndexSVS::add(idx_t n, const float* x) {
    if (!impl) {
        init_impl(n, x);
        return;
    }

    // construct sequential labels
    std::vector<size_t> labels(n);

    svs::threads::parallel_for(
            impl->get_threadpool_handle(),
            svs::threads::StaticPartition(n),
            [&](auto is, auto SVS_UNUSED(tid)) {
                for (auto i : is) {
                    labels[i] = ntotal + i;
                }
            });
    ntotal += n;

    auto data = svs::data::ConstSimpleDataView<float>(x, n, d);
    impl->add_points(data, labels);
}

void IndexSVS::reset() {
    if (impl) {
        delete impl;
        impl = nullptr;
    }
    ntotal = 0;
}

void IndexSVS::search(
        idx_t n,
        const float* x,
        idx_t k,
        float* distances,
        idx_t* labels,
        const SearchParameters* params) const {
    FAISS_THROW_IF_NOT(impl);
    FAISS_THROW_IF_NOT(k > 0);
    FAISS_THROW_IF_NOT(is_trained);

    auto queries = svs::data::ConstSimpleDataView<float>(x, n, d);

    // TODO: use params for SVS search parameters
    auto sp = impl->get_search_parameters();
    sp.buffer_config({search_window_size, search_buffer_capacity});
    // TODO: faiss use int64_t as label whereas SVS uses size_t?
    auto results = svs::QueryResultView<size_t>{
            svs::MatrixView<size_t>{
                    svs::make_dims(n, k),
                    static_cast<size_t*>(static_cast<void*>(labels))},
            svs::MatrixView<float>{svs::make_dims(n, k), distances}};
    impl->search(results, queries, sp);
}

void IndexSVS::init_impl(idx_t n, const float* x) {
    std::vector<size_t> labels(n);
    auto data = svs::data::SimpleData<float>(n, d);
    auto threadpool = svs::threads::as_threadpool(num_threads);

    svs::threads::parallel_for(
            threadpool,
            svs::threads::StaticPartition(n),
            [&](auto is, auto SVS_UNUSED(tid)) {
                for (auto i : is) {
                    data.set_datum(i, std::span<const float>(x + i * d, d));
                    labels[i] = ntotal + i;
                }
            });
    ntotal += n;

    svs::index::vamana::VamanaBuildParameters build_parameters{
            alpha,
            graph_max_degree,
            construction_window_size,
            max_candidate_pool_size,
            prune_to,
            use_full_search_history};

    switch (metric_type) {
        case METRIC_INNER_PRODUCT:
            impl = new svs::DynamicVamana(svs::DynamicVamana::build<float>(
                    std::move(build_parameters),
                    std::move(data),
                    std::move(labels),
                    svs::DistanceIP(),
                    std::move(threadpool)));
            break;
        case METRIC_L2:
            impl = new svs::DynamicVamana(svs::DynamicVamana::build<float>(
                    std::move(build_parameters),
                    std::move(data),
                    std::move(labels),
                    svs::DistanceL2(),
                    std::move(threadpool)));
            break;
        default:
            FAISS_ASSERT(!"not supported SVS distance");
    }
}

void IndexSVS::serialize_impl(std::ostream& out) const {
    FAISS_THROW_IF_NOT_MSG(
            impl, "Cannot serialize: SVS index not initialized.");

    // Write index to temporary files and concatenate the contents
    detail::SVSTempDirectory tmp;
    impl->save(tmp.config, tmp.graph, tmp.data);
    tmp.write_files_to_stream(out);
}

void IndexSVS::deserialize_impl(std::istream& in) {
    FAISS_THROW_IF_MSG(
            impl, "Cannot deserialize: SVS index already initialized.");

    // Write stream to files that can be read by DynamicVamana::assemble()
    detail::SVSTempDirectory tmp;
    tmp.write_stream_to_files(in);

    switch (metric_type) {
        case METRIC_INNER_PRODUCT:
            impl = new svs::DynamicVamana(svs::DynamicVamana::assemble<float>(
                    tmp.config.string(),
                    svs::GraphLoader(tmp.graph.string()),
                    svs::VectorDataLoader<float>(tmp.data.string()),
                    svs::distance::DistanceIP(),
                    num_threads));
            break;
        case METRIC_L2:
            impl = new svs::DynamicVamana(svs::DynamicVamana::assemble<float>(
                    tmp.config.string(),
                    svs::GraphLoader(tmp.graph.string()),
                    svs::VectorDataLoader<float>(tmp.data.string()),
                    svs::distance::DistanceL2(),
                    num_threads));
            break;
        default:
            FAISS_ASSERT(!"not supported SVS distance");
    }   
}

} // namespace faiss
