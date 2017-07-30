
/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree. An additional grant
 * of patent rights can be found in the PATENTS file in the same directory.
 */

// Copyright 2004-present Facebook. All Rights Reserved
// -*- c++ -*-

// This is the main internal include file for Faiss. It defines
// macros and some machine-specific functions shared across .cpp files

#ifndef FAISS_h
#define FAISS_h

#include <stdint.h>
#include <stdlib.h>
#include <stddef.h>

#ifndef __SSE2__
    #error "SSE optimized distance computations not set"
#endif




#ifdef _OPENMP
  #include <omp.h>
  #define SET_NT(ntlim)                          \
      size_t nt = omp_get_max_threads();         \
      if (nt > ntlim) nt = ntlim;
#else
  #warning "OpenMP is NOT activated"
  #define SET_NT(ntlim) size_t nt = 0; nt++;
#endif

/* This is to prevent warning by the linter (FINTEGER is defined externally) */
#ifndef FINTEGER
  #define FINTEGER long
#endif

#endif
