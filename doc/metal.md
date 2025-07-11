# Metal Backend for Faiss

This document describes how to build and use the Metal backend for Faiss on Apple Silicon.

## Prerequisites

-   macOS 14.0 or later
-   Xcode 15.0 or later
-   CMake 3.24.0 or later

## Building with Metal Support

To build Faiss with Metal support, use the `metal.cmake` toolchain file:

```bash
cmake -B build -DCMAKE_TOOLCHAIN_FILE=../cmake/toolchains/metal.cmake ..
make -C build -j
```

## Using the Metal Backend

To use the Metal backend, you can use the `faiss-metal` conda package:

```bash
conda install -c faiss faiss-metal
```

Then, in Python, you can create Metal-backed indices:

```python
import faiss

# Create a Metal-backed IndexFlat
index = faiss.MetalIndexFlat(128)

# Or, use the index_factory with the ",Metal" suffix
index = faiss.index_factory(128, "Flat,Metal")
```
