// Copyright (C) 2019-2024 Zilliz. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License"); you may not
// use this file except in compliance with the License. You may obtain a copy of
// the License at
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
// WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
// License for the specific language governing permissions and limitations under
// the License.

#pragma once

#include <faiss/impl/io.h>

namespace faiss {
namespace cppcontrib {
namespace knowhere {

// an IOWriter that just counts the number of bytes without writing anything.
struct CountSizeIOWriter : IOWriter {
    size_t total_size = 0;

    size_t operator()(const void*, size_t size, size_t nitems) override {
        total_size += size * nitems;
        return nitems;
    }
};

} // namespace knowhere
} // namespace cppcontrib
} // namespace faiss
