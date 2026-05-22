# FAISS Wheel Naming Convention

_Last updated: 2026-03-25_

This document records the research and rationale behind the `faiss-gpu-cu132`
package name used in this branch.

---

## Ecosystem Survey

### PyPI

| Package | Status | Latest | Notes |
|---------|--------|--------|-------|
| `faiss-cpu` | **Active** | 1.13.2 | The only actively maintained PyPI FAISS package |
| `faiss-gpu` | **Archived** | 1.7.2 (Jan 2022) | Discontinued — GPU wheels exceed PyPI's 100 MB file size limit |
| `faiss-gpu-cu128` | Does not exist | — | No CUDA-versioned names on PyPI |
| `faiss-gpu-cu132` | Does not exist | — | Same |

GPU wheels were dropped from PyPI as of faiss 1.7.3 and will not return.
The root cause is binary size: a GPU wheel with multiple CUDA architectures
easily exceeds PyPI's upload limit.
See [pypackaging-native.github.io — GPU key issues](https://pypackaging-native.github.io/key-issues/gpus/).

The community-maintained [faiss-wheels](https://github.com/faiss-wheels/faiss-wheels)
project (kyamagu) is the source for `faiss-cpu` on PyPI. Its
`scripts/rename_project.sh faiss-gpu` helper renames the project to `faiss-gpu`
for custom GPU builds, confirming `faiss-gpu` as the de-facto community GPU name
— but **no CUDA-version-specific names** (`-cu128`, `-cu132`) exist or are used.

### Anaconda (pytorch channel)

| Package | Version | Notes |
|---------|---------|-------|
| `faiss-cpu` | 1.14.1 | Active |
| `faiss-gpu` | 1.14.1 | Active, currently built against CUDA 12.6 |
| `faiss-gpu-cuvs` | 1.14.1 | cuVS/RAPIDS variant |
| `faiss-gpu-raft` | 1.9.0 | Older RAPIDS/RAFT variant |

Anaconda uses **feature suffixes** (`-cuvs`, `-raft`) to distinguish build
variants, but **not CUDA version suffixes**. The CUDA version appears only in
the conda package *filename* metadata (e.g. `_cuda12.6_`), not in the package
name itself.

---

## Why `faiss-gpu-cu132`?

CUDA-version-specific wheel suffixes (`-cu128`, `-cu132`) originate from
**PyTorch's** distribution convention (e.g. `torch-2.x+cu132`). That convention
has **not** been adopted by the FAISS ecosystem as of 2026-03.

However, it is the right choice for this private branch for several reasons:

1. **No collision risk** — neither PyPI nor Anaconda uses this name, so there is
   no chance of pulling the wrong wheel from a package index.
2. **CUDA runtime is a hard dependency** — a wheel built against CUDA 13.2 will
   not load on a system with a different major CUDA version. Encoding this in the
   name makes the dependency explicit to consumers.
3. **Private / internal distribution** — GPU wheels cannot be published to PyPI
   anyway (size limit). On a private index (Azure Artifacts, Gemfury, DevPI, a
   shared file server) the CUDA version in the name is genuinely useful for
   administrators managing multiple CUDA environments.
4. **Follows an established convention** — PyTorch's `-cu132` suffix is widely
   recognised; users who install PyTorch wheels will find `faiss-gpu-cu132`
   immediately legible.

---

## `FAISS_VARIANT` Naming Table

The build system exposes a `FAISS_VARIANT` environment variable that is passed
to `setup.py` at wheel-build time. The resulting package name is
`faiss-{FAISS_VARIANT}` (or plain `faiss` when the variable is unset).

| Scenario | `FAISS_VARIANT` | Resulting wheel name | Alignment |
|----------|-----------------|----------------------|-----------|
| **This branch** — CUDA 13.2 GPU | `gpu-cu132` | `faiss-gpu-cu132` | Novel; follows torch convention |
| **DGX Spark** — CUDA 13.2, SM 103, aarch64 | `gpu-cu132-spark` | `faiss-gpu-cu132-spark` | Spark-specific variant |
| CUDA 12.8 GPU build | `gpu-cu128` | `faiss-gpu-cu128` | Same pattern |
| Generic GPU (no CUDA-version lock) | `gpu` | `faiss-gpu` | Matches Anaconda + archived PyPI |
| CPU-only build | `cpu` | `faiss-cpu` | Exact match for active PyPI package |
| Upstream canonical/untagged | *(unset)* | `faiss` | Plain upstream name |

---

## Library Naming Convention (C++ / Shared Object)

When building the C++ libraries directly (without a Python wheel), the following
naming scheme is used. The base names follow CMake target names; GPU-specific and
variant builds append a suffix derived from the same `-{variant}-cu{ver}` pattern
used for wheels.

### Standard (x86_64) build — `build_lib_cuda132.sh`

| Library | Filename | Notes |
|---------|----------|-------|
| Main C++ library | `libfaiss.so` | Default CMake target name |
| AVX2 variant | `libfaiss_avx2.so` | CPU SIMD opt-level |
| AVX512 variant | `libfaiss_avx512.so` | CPU SIMD opt-level |
| C API wrapper | `libfaiss_c.so` | Thin C shim over `libfaiss` |
| C API AVX2 | `libfaiss_c_avx2.so` | Paired with `libfaiss_avx2` |
| cuVS companion | `libcuvs.so` | From `rapidsai/cuvs` |

### DGX Spark (aarch64, SM 103) build — `build_lib_spark.sh`

| Library | Filename | Notes |
|---------|----------|-------|
| Main C++ library | `libfaiss-spark-cu132.so` | `FAISS_OUTPUT_NAME=faiss-spark-cu132` |
| C API wrapper | `libfaiss_c-spark-cu132.so` | `FAISS_C_OUTPUT_NAME=faiss_c-spark-cu132` |
| cuVS companion | `libcuvs-spark.so` | From `zbrad/cuvs`, SM 103 only |

### DGX Spark Python wheel — `build_wheel_spark.sh`

| Artifact | Name | Notes |
|----------|------|-------|
| Python wheel | `faiss-gpu-cu132-spark` | `FAISS_VARIANT=gpu-cu132-spark` in `setup.py` |
| Built by | `build_wheel_spark.sh` | Orchestrates lib → pkg → wheel steps |
| Stage dir | `_libfaiss_stage_spark/` | Mirrors `_libfaiss_stage/` for x86_64 build |

The `FAISS_OUTPUT_NAME` and `FAISS_C_OUTPUT_NAME` cmake variables are defined in
`faiss/CMakeLists.txt` and `c_api/CMakeLists.txt` respectively and have no effect
when left unset (standard build produces `libfaiss.so` / `libfaiss_c.so`).

The cuVS companion library name mirrors the zbrad/cuvs project convention:
`libcuvs-spark-{version}-cuda{ver}-{arch}-sm{arch}.tar.gz` → `libcuvs-spark.so`.

---

## References

- faiss-wheels README: <https://github.com/faiss-wheels/faiss-wheels>
- PyPI `faiss-cpu`: <https://pypi.org/project/faiss-cpu/>
- PyPI `faiss-gpu` (archived): <https://pypi.org/project/faiss-gpu/>
- Anaconda pytorch channel `faiss-gpu`: <https://anaconda.org/pytorch/faiss-gpu>
- GPU packaging background: <https://pypackaging-native.github.io/key-issues/gpus/>
- PyTorch CUDA wheel convention: <https://download.pytorch.org/whl/torch_stable.html>
