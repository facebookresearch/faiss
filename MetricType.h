/**
 * Copyright (c) 2018-present, Facebook, Inc.
 * All rights reserved.
 *
 * This source code is licensed under the BSD+Patents license found in the
 * LICENSE file in the root directory of this source tree.
 */

// Copyright 2004-present Facebook. All Rights Reserved
// -*- c++ -*-

#ifndef FAISS_METRIC_TYPE_H
#define FAISS_METRIC_TYPE_H

namespace faiss {


/// Some algorithms support both an inner product version and a L2 search version.
enum MetricType {
  METRIC_INNER_PRODUCT = 0,
  METRIC_L2 = 1,
};


}  // namespace faiss

#endif  // FAISS_METRIC_TYPE_H
