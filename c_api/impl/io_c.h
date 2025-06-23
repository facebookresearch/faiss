/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c -*-

#ifndef FAISS_IO_C_H
#define FAISS_IO_C_H

#include <stddef.h>
#include "../faiss_c.h"

#ifdef __cplusplus
extern "C" {
#endif

FAISS_DECLARE_CLASS(IOReader)
FAISS_DECLARE_DESTRUCTOR(IOReader)

FAISS_DECLARE_CLASS(IOWriter)
FAISS_DECLARE_DESTRUCTOR(IOWriter)

/*******************************************************
 * Custom reader + writer
 *
 * Reader and writer which wraps a function pointer,
 * primarily for FFI use.
 *******************************************************/

FAISS_DECLARE_CLASS(CustomIOReader)
FAISS_DECLARE_DESTRUCTOR(CustomIOReader)

int faiss_CustomIOReader_new(
        FaissCustomIOReader** p_out,
        size_t (*func_in)(void* ptr, size_t size, size_t nitems));

FAISS_DECLARE_CLASS(CustomIOWriter)
FAISS_DECLARE_DESTRUCTOR(CustomIOWriter)

int faiss_CustomIOWriter_new(
        FaissCustomIOWriter** p_out,
        size_t (*func_in)(const void* ptr, size_t size, size_t nitems));

#ifdef __cplusplus
}
#endif
#endif
