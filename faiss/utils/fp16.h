#pragma once

#include <cstdint>

#if defined(__SSE__) && defined(USE_F16C)
#include <faiss/utils/fp16-fp16c.h>
#else
#include <faiss/utils/fp16-inl.h>
#endif
