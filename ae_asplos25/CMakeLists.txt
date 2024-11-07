# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

add_executable(Fig_3 EXCLUDE_FROM_ALL Fig_3.cpp)
target_compile_options(Fig_3 PRIVATE -fopenmp)
target_link_libraries(Fig_3 PRIVATE faiss)

