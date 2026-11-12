/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c -*-

#ifndef FAISS_FACTORY_TOOLS_C_H
#define FAISS_FACTORY_TOOLS_C_H

#include <stddef.h>

#include "IndexBinary_c.h"
#include "Index_c.h"

#ifdef __cplusplus
extern "C" {
#endif

/** Get the factory string that describes the given index, by writing it to
 *  the given character buffer as a NUL-terminated string.
 *
 *  The string is empty if the index type is not supported.
 *
 * @param index   the index to describe
 * @param buffer  output buffer
 * @param size    size of the output buffer, including the terminating NUL
 * @return 0 on success, otherwise an error code if the buffer is too small to
 *         hold the full string, in which case the buffer is left untouched
 */
int faiss_reverse_index_factory(
        const FaissIndex* index,
        char* buffer,
        size_t size);

/** Get the factory string that describes the given binary index, by writing
 *  it to the given character buffer as a NUL-terminated string.
 *
 *  The string is empty if the index type is not supported.
 *
 * @param index   the binary index to describe
 * @param buffer  output buffer
 * @param size    size of the output buffer, including the terminating NUL
 * @return 0 on success, otherwise an error code if the buffer is too small to
 *         hold the full string, in which case the buffer is left untouched
 */
int faiss_reverse_index_binary_factory(
        const FaissIndexBinary* index,
        char* buffer,
        size_t size);

#ifdef __cplusplus
}
#endif

#endif
