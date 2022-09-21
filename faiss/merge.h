/**
 * Copyright (c) Facebook, Inc. and its affiliates.
 *
 * This source code is licensed under the MIT license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <faiss/Index.h>

namespace faiss {
namespace merge {
/** check if two indexes have the same parameters and are trained in
 * the same way, otherwise throw. */
void check_compatible_for_merge(const Index* index1, const Index* index2);

/** get a base Index from an index. The index may be the required Index or
 * some wrapper class that encloses an Index
 *
 * throws an exception if this is not the case.
 */

const Index* extract_index(const Index* index);
Index* extract_index(Index* index);

/// same as above but returns nullptr instead of throwing on failure

const Index* try_extract_index(const Index* index);
Index* try_extract_index(Index* index);

/** Merge index1 into index0. Works on all Index Types that implement merge_from
 * or embedded in a IndexPreTransform / IndexIDMap / IndexIDMap2. On output,
 * the index1 is empty.
 *
 * @param shift_ids: translate the ids from index1 to index0->prev_ntotal
 */
void merge_into(Index* index0, Index* index1, bool shift_ids = false);
} // namespace merge
} // namespace faiss
