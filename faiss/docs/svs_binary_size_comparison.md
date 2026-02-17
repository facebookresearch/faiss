# Faiss SVS Binary Size Comparison

## Overview

This document describes the methodology for comparing binary sizes between the default `faiss` target (without SVS) and the `faiss_svs` target (with SVS support).

## Background

SVS (Scalable Vector Search) is an optional feature in Faiss that adds additional index implementations. Including SVS increases binary size, so it was separated into an opt-in target to reduce binary bloat for users who don't need SVS functionality.

## Targets

| Target | Description |
|--------|-------------|
| `//faiss:faiss` | Default Faiss library without SVS |
| `//faiss:faiss_svs` | Faiss library with SVS support |
| `//faiss:pyfaiss` | Python bindings without SVS |
| `//faiss:pyfaiss_svs` | Python bindings with SVS |

## Methodology

### Why `demo_sift1M`?

The `demo_sift1M` binary is used for comparison because it uses `index_factory`, which creates indexes dynamically from string descriptions at runtime. This means:

- The linker cannot determine at link time which index implementations will be used
- All index implementations must be included in the binary
- This provides a worst-case (maximum) binary size measurement

### Build Commands

```bash
# Build without SVS
buck2 build @mode/opt fbcode//faiss/demos:demo_sift1M --show-full-output

# Build with SVS
buck2 build @mode/opt fbcode//faiss/demos:demo_sift1M_svs --show-full-output
```

### Size Comparison

```bash
# Check sizes
ls -lh <path_to_demo_sift1M>
ls -lh <path_to_demo_sift1M_svs>

# Verify no SVS symbols in default build
nm <path_to_demo_sift1M> | grep -i svs
```

## Results

| Configuration | Binary Size |
|---------------|-------------|
| Without SVS (`faiss`) | 33 MB |
| With SVS (`faiss_svs`) | 41 MB |
| **Difference** | **8.24 MB (20% reduction)** |

## Files Changed

The following files were modified to separate SVS from the default Faiss build:

### `faiss/xplat.bzl`
- Separated SVS source and header files into dedicated functions:
  - `svs_header_files()` - SVS header files
  - `svs_source_files()` - SVS source files

### `faiss/BUCK`
- Default `faiss` target no longer includes SVS
- Added new `faiss_svs` target with SVS support
- Updated `faiss_no_multithreading` and `faiss_omp_mock` to exclude SVS

### `faiss/python/defs.bzl`
- Added `with_svs` parameter to `pyfaiss_binary()` macro

### `faiss/python/BUCK`
- Added `pyfaiss_svs` target

### `faiss/fbcode.bzl`
- Updated `pyfaiss_libraries()` to include SVS variant

### `faiss/tests/BUCK`
- Updated SVS tests to use `faiss_svs` and `pyfaiss_svs`

### `faiss/demos/BUCK`
- Added `demo_sift1M_svs` for binary size comparison

## Usage

### For users who need SVS

Replace dependencies on `//faiss:faiss` with `//faiss:faiss_svs`:

```python
# BUCK file
cpp_binary(
    name = "my_binary",
    srcs = ["main.cpp"],
    deps = ["//faiss:faiss_svs"],  # Use faiss_svs for SVS support
)
```

For Python:
```python
python_binary(
    name = "my_script",
    srcs = ["main.py"],
    deps = ["//faiss:pyfaiss_svs"],  # Use pyfaiss_svs for SVS support
)
```

### For users who don't need SVS

No changes needed - the default `//faiss:faiss` and `//faiss:pyfaiss` targets now exclude SVS automatically.
