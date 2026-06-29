/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c -*-

// Smoke test for the three new fields on FaissSearchParametersIVF:
//   - max_lists_num
//   - ensure_topk_full
//   - max_empty_result_buckets
// Verifies that the C API symbols link, that getters/setters round-trip,
// and that the default values match the C++ struct defaults.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>

#include "IndexIVF_c.h"
#include "error_c.h"

#define FAISS_TRY(C)                                       \
    {                                                      \
        if (C) {                                           \
            fprintf(stderr, "%s", faiss_get_last_error()); \
            exit(-1);                                      \
        }                                                  \
    }

int main(void) {
    FaissSearchParametersIVF* sp = NULL;
    FAISS_TRY(faiss_SearchParametersIVF_new(&sp));

    // Defaults: nprobe=1, everything else 0/false.
    assert(faiss_SearchParametersIVF_nprobe(sp) == 1);
    assert(faiss_SearchParametersIVF_max_codes(sp) == 0);
    assert(faiss_SearchParametersIVF_max_lists_num(sp) == 0);
    assert(faiss_SearchParametersIVF_ensure_topk_full(sp) == 0);
    assert(faiss_SearchParametersIVF_max_empty_result_buckets(sp) == 0);

    // Roundtrip setters.
    faiss_SearchParametersIVF_set_max_lists_num(sp, 4);
    assert(faiss_SearchParametersIVF_max_lists_num(sp) == 4);

    faiss_SearchParametersIVF_set_ensure_topk_full(sp, 1);
    assert(faiss_SearchParametersIVF_ensure_topk_full(sp) != 0);

    faiss_SearchParametersIVF_set_ensure_topk_full(sp, 0);
    assert(faiss_SearchParametersIVF_ensure_topk_full(sp) == 0);

    faiss_SearchParametersIVF_set_max_empty_result_buckets(sp, 3);
    assert(faiss_SearchParametersIVF_max_empty_result_buckets(sp) == 3);

    faiss_SearchParametersIVF_free(sp);

    printf("example_search_params_ivf_c: OK\n");
    return 0;
}
