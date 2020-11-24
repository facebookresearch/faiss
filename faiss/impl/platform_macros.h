/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#pragma once


#ifdef _MSC_VER
// Windows specific macros


#ifdef FAISS_MAIN_LIB
#define FAISS_API __declspec(dllexport)
#else // _FAISS_MAIN_LIB
#define FAISS_API __declspec(dllimport)
#endif // FAISS_MAIN_LIB

#define __PRETTY_FUNCTION__ __FUNCSIG__

#define posix_memalign(p, a, s) (((*(p)) = _aligned_malloc((s), (a))), *(p) ?0 :errno)
#define posix_memalign_free _aligned_free


#else
// Linux and OSX

#define FAISS_API
#define posix_memalign_free free

#endif // _MSC_VER
