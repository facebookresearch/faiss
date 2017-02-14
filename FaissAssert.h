
/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the CC-by-NC license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved.

#ifndef FAISS_ASSERT_INCLUDED
#define FAISS_ASSERT_INCLUDED

#include <cstdlib>
#include <cstdio>

/// Asserts that risk to be triggered by user input
#define FAISS_ASSERT(X) ({if (! (X)) {                             \
    fprintf (stderr, "Faiss assertion %s failed in %s at %s:%d", \
             #X, __PRETTY_FUNCTION__, __FILE__, __LINE__);       \
    abort(); }})



#endif
