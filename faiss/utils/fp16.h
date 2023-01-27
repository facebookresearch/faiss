#pragma once

#include <cstdint>

#include <faiss/impl/platform_macros.h>

#if defined(__F16C__)
#include <faiss/utils/fp16-fp16c.h>
#else
#include <faiss/utils/fp16-inl.h>
#endif
