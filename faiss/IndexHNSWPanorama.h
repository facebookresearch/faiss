/*
 * Copyright (c) Meta Platforms, Inc. and affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

// -*- c++ -*-

#pragma once

namespace faiss {

// MASTER PLAN
// 1. Add to the existing FlatCodesDistanceComputer class to support partial dot product search.
// NB: After discussion, add it at the top and make it throw by default. Only implement it for
// L2 Flat distance computer.
// 2. Create a IndexHNSWFlatPanorama which inherits from IndexHNSWFlat and sets the HNSW as HNSWPanorama.
// Also enforce the invariant of L2 distance.
// 3. Create HNSWPanorama which inherits from HNSW, and changes the search (or more specific), and alos contains
// the number of levels as class parameter Try to only change search from unbounded and bounded.
// 4. Test bounded & unbounded. Maybe use 1 level as a way to test it. Also check how HNSW is currently tested.
// Run vanilla HNSW and compare the recall to be within a margin.
// 5. Demo and bench PCA + HNSW with and without Panorama.

};