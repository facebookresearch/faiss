# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.17 FATAL_ERROR)
project (ROCKSDB_IVF)
set(CMAKE_BUILD_TYPE Debug)
find_package(faiss REQUIRED)
find_package(RocksDB REQUIRED)

add_executable(demo_rocksdb_ivf demo_rocksdb_ivf.cpp RocksDBInvertedLists.cpp)
target_link_libraries(demo_rocksdb_ivf faiss RocksDB::rocksdb)
