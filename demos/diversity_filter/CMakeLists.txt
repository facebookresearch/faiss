# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

add_executable(demo_diversity_result_handler EXCLUDE_FROM_ALL
  demo_diversity_result_handler.cpp)
target_link_libraries(demo_diversity_result_handler PRIVATE faiss)
faiss_add_warnings(demo_diversity_result_handler)
