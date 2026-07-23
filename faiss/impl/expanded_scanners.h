/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once

#include <cstdio>

#include <faiss/IndexIVF.h>
#include <faiss/impl/InvertedListScannerStats.h>
#include <faiss/impl/ResultHandler.h>

/* This is the inner loop of the inverted list scanners. The default version
 * that is defined in IndexIVF.cpp works fine but it cannot inline the distance
 * computation code by calling one or another of the run_scan_codes_* variants
 * with the exact ScannerType and by setting distance_to_code to be a final
 * function, the code can be inlined. The speed difference matters for very
 * small distance computations (eg. SQ or Flat) */

namespace faiss {

namespace {

template <class ScannerType, typename C, bool store_pairs, bool use_sel>
size_t run_scan_codes1(
        const ScannerType& scanner,
        size_t list_size,
        const uint8_t* codes,
        const idx_t* ids,
        ResultHandler& handler) {
    size_t nup = 0;
    size_t list_no = scanner.list_no;
    size_t code_size = scanner.code_size;
    const IDSelector* sel = scanner.sel;
    float threshold = handler.threshold;
    for (size_t j = 0; j < list_size; j++) {
        if (use_sel) {
            int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
            // skip code without computing distance
            if (!sel->is_member(id)) {
                codes += code_size;
                continue;
            }
        }

        // post-IDSelector: distance is about to be computed for this code.
        handler.stats.scan_cnt++;
        float dis = scanner.distance_to_code(codes); // will be inlined if final
        if (C::cmp(threshold, dis)) {
            int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
            if (handler.add_result(dis, id)) {
                handler.stats.nheap_updates++;
                nup++;
                threshold = handler.threshold;
            }
        }
        codes += code_size;
    }

    return nup;
}

// Batched variant of run_scan_codes1 for the SQ scanners: distances four codes
// per step via distance_to_codes_batch_4, then applies the threshold and heap
// updates in id order so results match run_scan_codes1. No-selector path only.
template <class ScannerType, typename C, bool store_pairs>
size_t run_scan_codes4(
        const ScannerType& scanner,
        size_t list_size,
        const uint8_t* codes,
        const idx_t* ids,
        ResultHandler& handler) {
    size_t nup = 0;
    size_t list_no = scanner.list_no;
    size_t code_size = scanner.code_size;
    float threshold = handler.threshold;

    size_t j = 0;
    for (; j + 4 <= list_size; j += 4) {
        float dis[4];
        scanner.distance_to_codes_batch_4(
                codes,
                codes + code_size,
                codes + 2 * code_size,
                codes + 3 * code_size,
                dis[0],
                dis[1],
                dis[2],
                dis[3]);
        handler.stats.scan_cnt += 4;
        for (size_t b = 0; b < 4; b++) {
            if (C::cmp(threshold, dis[b])) {
                int64_t id =
                        store_pairs ? lo_build(list_no, j + b) : ids[j + b];
                if (handler.add_result(dis[b], id)) {
                    handler.stats.nheap_updates++;
                    nup++;
                    threshold = handler.threshold;
                }
            }
        }
        codes += 4 * code_size;
    }

    // tail: the final < 4 codes, one at a time
    for (; j < list_size; j++) {
        handler.stats.scan_cnt++;
        float dis = scanner.distance_to_code(codes);
        if (C::cmp(threshold, dis)) {
            int64_t id = store_pairs ? lo_build(list_no, j) : ids[j];
            if (handler.add_result(dis, id)) {
                handler.stats.nheap_updates++;
                nup++;
                threshold = handler.threshold;
            }
        }
        codes += code_size;
    }

    return nup;
}

/*****************************************************************************
 * The following functions dispatch runtime parameters to templates, with
 * possibly some already-fixed templates.
 */

template <bool store_pairs, bool use_sel, class ScannerType>
size_t run_scan_codes_fix_store_pairs_fix_use_sel(
        const ScannerType& scanner,
        size_t list_size,
        const uint8_t* codes,
        const idx_t* ids,
        ResultHandler& handler) {
    if (scanner.keep_max) {
        return run_scan_codes1<
                ScannerType,
                CMin<float, idx_t>,
                store_pairs,
                use_sel>(scanner, list_size, codes, ids, handler);
    } else {
        return run_scan_codes1<
                ScannerType,
                CMax<float, idx_t>,
                store_pairs,
                use_sel>(scanner, list_size, codes, ids, handler);
    }
}

template <class C, bool use_sel, class ScannerType>
size_t run_scan_codes_fix_C_fix_use_sel(
        const ScannerType& scanner,
        size_t list_size,
        const uint8_t* codes,
        const idx_t* ids,
        ResultHandler& handler) {
    if (scanner.store_pairs) {
        return run_scan_codes1<ScannerType, C, true, use_sel>(
                scanner, list_size, codes, ids, handler);
    } else {
        return run_scan_codes1<ScannerType, C, false, use_sel>(
                scanner, list_size, codes, ids, handler);
    }
}

template <class C, class ScannerType>
size_t run_scan_codes_fix_C(
        const ScannerType& scanner,
        size_t list_size,
        const uint8_t* codes,
        const idx_t* ids,
        ResultHandler& handler) {
    if (scanner.sel) {
        if (scanner.store_pairs) {
            return run_scan_codes1<ScannerType, C, true, true>(
                    scanner, list_size, codes, ids, handler);
        } else {
            return run_scan_codes1<ScannerType, C, false, true>(
                    scanner, list_size, codes, ids, handler);
        }
    } else {
        if (scanner.store_pairs) {
            return run_scan_codes1<ScannerType, C, true, false>(
                    scanner, list_size, codes, ids, handler);
        } else {
            return run_scan_codes1<ScannerType, C, false, false>(
                    scanner, list_size, codes, ids, handler);
        }
    }
}

// Routing wrapper for the SQ scanners: the no-selector path takes the batched
// run_scan_codes4, the selector path stays on run_scan_codes1.
template <class C, class ScannerType>
size_t run_scan_codes4_fix_C(
        const ScannerType& scanner,
        size_t list_size,
        const uint8_t* codes,
        const idx_t* ids,
        ResultHandler& handler) {
    if (scanner.sel) {
        if (scanner.store_pairs) {
            return run_scan_codes1<ScannerType, C, true, true>(
                    scanner, list_size, codes, ids, handler);
        } else {
            return run_scan_codes1<ScannerType, C, false, true>(
                    scanner, list_size, codes, ids, handler);
        }
    } else {
        if (scanner.store_pairs) {
            return run_scan_codes4<ScannerType, C, true>(
                    scanner, list_size, codes, ids, handler);
        } else {
            return run_scan_codes4<ScannerType, C, false>(
                    scanner, list_size, codes, ids, handler);
        }
    }
}

template <class ScannerType>
size_t run_scan_codes(
        const ScannerType& scanner,
        size_t list_size,
        const uint8_t* codes,
        const idx_t* ids,
        ResultHandler& handler) {
    if (scanner.sel == nullptr) {
        if (scanner.store_pairs) {
            return run_scan_codes_fix_store_pairs_fix_use_sel<true, false>(
                    scanner, list_size, codes, ids, handler);
        } else {
            return run_scan_codes_fix_store_pairs_fix_use_sel<false, false>(
                    scanner, list_size, codes, ids, handler);
        }
    } else {
        if (scanner.store_pairs) {
            return run_scan_codes_fix_store_pairs_fix_use_sel<true, true>(
                    scanner, list_size, codes, ids, handler);
        } else {
            return run_scan_codes_fix_store_pairs_fix_use_sel<false, true>(
                    scanner, list_size, codes, ids, handler);
        }
    }
}

} // anonymous namespace

} // namespace faiss
