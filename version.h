/**
 * Copyright (c) 2015-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#ifndef VERSION_H
#define VERSION_H

#define FAISS_VERSION_MAJOR 1
#define FAISS_VERSION_MINOR 3
#define FAISS_VERSION_PATCH 0
#define FAISS_VERSION_STRING "1.3.0"


namespace faiss {


constexpr int VERSION_MAJOR = FAISS_VERSION_MAJOR;
constexpr int VERSION_MINOR = FAISS_VERSION_MINOR;
constexpr int VERSION_PATCH = FAISS_VERSION_PATCH;

constexpr const char VERSION_STRING[] = FAISS_VERSION_STRING;


}  // namespace faiss

#endif  // VERSION_H
