/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#ifndef FAISS_INVERTED_LIST_SCANNER_STATS_H
#define FAISS_INVERTED_LIST_SCANNER_STATS_H

#include <cstddef>

namespace faiss {

/** Per-list statistics returned by inverted-list scanners. */
struct InvertedListScannerStats {
    /// Number of distances computed after IDSelector filtering.
    size_t scan_cnt = 0;

    /// Number of heap updates.
    size_t nheap_updates = 0;
};

} // namespace faiss

#endif
