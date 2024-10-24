/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#include <exception>
#include <iostream>
#include <memory>

#include "RocksDBInvertedLists.h"

#include <faiss/IndexFlat.h>
#include <faiss/IndexIVFFlat.h>
#include <faiss/impl/AuxIndexStructures.h>
#include <faiss/impl/FaissException.h>
#include <faiss/utils/random.h>

using namespace faiss;

int main(int argc, char* argv[]) {
    try {
        if (argc != 2) {
            std::cerr << "missing db directory argument" << std::endl;
            return -1;
        }
        size_t d = 128;
        size_t nlist = 100;
        IndexFlatL2 quantizer(d);
        IndexIVFFlat index(&quantizer, d, nlist);
        faiss_rocksdb::RocksDBInvertedLists ril(
                argv[1], nlist, index.code_size);
        index.replace_invlists(&ril, false);

        idx_t nb = 10000;
        std::vector<float> xb(d * nb);
        float_rand(xb.data(), d * nb, 12345);
        std::vector<idx_t> xids(nb);
        std::iota(xids.begin(), xids.end(), 0);

        index.train(nb, xb.data());
        index.add_with_ids(nb, xb.data(), xids.data());

        idx_t nq = 20; // nb;
        index.nprobe = 2;

        std::cout << "search" << std::endl;
        idx_t k = 5;
        std::vector<float> distances(nq * k);
        std::vector<idx_t> labels(nq * k, -1);
        index.search(
                nq, xb.data(), k, distances.data(), labels.data(), nullptr);

        for (idx_t iq = 0; iq < nq; iq++) {
            std::cout << iq << ": ";
            for (auto j = 0; j < k; j++) {
                std::cout << labels[iq * k + j] << " " << distances[iq * k + j]
                          << " | ";
            }
            std::cout << std::endl;
        }

        std::cout << std::endl << "range search" << std::endl;
        float range = 15.0f;
        RangeSearchResult result(nq);
        index.range_search(nq, xb.data(), range, &result);

        for (idx_t iq = 0; iq < nq; iq++) {
            std::cout << iq << ": ";
            for (auto j = result.lims[iq]; j < result.lims[iq + 1]; j++) {
                std::cout << result.labels[j] << " " << result.distances[j]
                          << " | ";
            }
            std::cout << std::endl;
        }

    } catch (FaissException& e) {
        std::cerr << e.what() << '\n';
    } catch (std::exception& e) {
        std::cerr << e.what() << '\n';
    } catch (...) {
        std::cerr << "Unrecognized exception!\n";
    }
    return 0;
}
