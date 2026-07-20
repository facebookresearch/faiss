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

## CPU arch by platform tag; GPU arch by optional name suffix

Two different "architectures" are in play and are encoded differently:

- **CPU arch (x86_64 vs aarch64)** — carried by the wheel's **platform tag**,
  which `auditwheel repair` stamps onto each wheel. Not in the package name.
- **GPU arch (SM / compute capability)** — a *multi-arch* (portable) build leaves
  it out of the name; a *single-arch* build appends **`-sm<arch>`** so a wheel
  that only runs on one GPU generation is identifiable.

| Build | Package name | Platform tag (filename) | BLAS / accel |
|-------|--------------|-------------------------|--------------|
| x86_64, multi-arch (default) | `faiss-gpu-cu132` | `…-manylinux2014_x86_64.whl` | Intel MKL, AVX2/AVX512 |
| x86_64, single-arch (`CUDA_ARCHS=89`) | `faiss-gpu-cu132-sm89` | `…-manylinux2014_x86_64.whl` | Intel MKL |
| aarch64 / DGX Spark (always SM 121) | `faiss-gpu-cu132-sm121` | `…-manylinux2014_aarch64.whl` | OpenBLAS + cuVS, SVE |

For a portable multi-arch build, `pip install faiss-gpu-cu132` resolves to the
correct wheel on each host via the platform tag (there is no Linux fat-binary
format, so x86_64 and aarch64 are genuinely separate wheels sharing one name).
A single-arch wheel carries its `-sm<arch>` in the name, so you install it
explicitly: `pip install faiss-gpu-cu132-sm121`.

The `-sm<arch>` suffix is added **automatically** whenever `CUDA_ARCHS` resolves
to exactly one GPU arch (see `faiss_sm_suffix` in `cuda_env.sh`); multi-arch
builds get no suffix. The DGX Spark / aarch64 build always targets SM 121, so it
is always `-sm121`.

## `FAISS_VARIANT` Naming Table

The build system exposes a `FAISS_VARIANT` environment variable that is passed
to `setup.py` at wheel-build time. The resulting package name is
`faiss-{FAISS_VARIANT}` (or plain `faiss` when the variable is unset). The build
scripts derive `FAISS_VARIANT=gpu-${FAISS_CUDA_TAG}` and append `-sm<arch>` for
single-arch builds.

| Scenario | `CUDA_ARCHS` | `FAISS_VARIANT` | Resulting wheel name |
|----------|--------------|-----------------|----------------------|
| CUDA 13.2 GPU, x86_64, portable | `75;80;86;89;90;120` | `gpu-cu132` | `faiss-gpu-cu132` |
| CUDA 13.2 GPU, x86_64, Ada-only | `89` | `gpu-cu132-sm89` | `faiss-gpu-cu132-sm89` |
| CUDA 13.2 GPU, aarch64 / DGX Spark | `121` | `gpu-cu132-sm121` | `faiss-gpu-cu132-sm121` |
| Next CUDA release (`FAISS_CUDA_VER=13.3`) | *(any)* | `gpu-cu133[-sm…]` | `faiss-gpu-cu133[-sm…]` |
| Generic GPU (no CUDA-version lock) | — | `gpu` | `faiss-gpu` |
| CPU-only build | `cpu` | `faiss-cpu` | Exact match for active PyPI package |
| Upstream canonical/untagged | *(unset)* | `faiss` | Plain upstream name |

## Selecting / bumping the CUDA version (cu132 ↔ cu133)

The CUDA version is a single input. Specify it **per build** (no file edits) with
either variable — the other is derived (`13.3` ⇄ `cu133`):

```bash
make build FAISS_CUDA_VER=13.3              # x86_64, CUDA 13.3 → faiss-gpu-cu133
make build-aarch64 FAISS_CUDA_VER=13.3      # aarch64 / DGX Spark, CUDA 13.3
FAISS_CUDA_TAG=cu133 bash gpu-cu/scripts/build_wheel_x86_64.sh   # tag form
```

To change the **default**, edit the two values in `gpu-cu/scripts/cuda_env.sh`
(mirrored by the same-named make variables). Every wheel name
(`faiss-gpu-${FAISS_CUDA_TAG}`) and C++ library name
(`libfaiss-{arch}-${FAISS_CUDA_TAG}.so`) derives from it, so a new CUDA release
needs no script or path renames. The `gpu-cu/` directory and `environment.yml`
are version-agnostic on purpose.

**Multi-toolkit hosts.** When `CUDA_HOME` is not set explicitly, `cuda_env.sh`
resolves it to `/usr/local/cuda-${FAISS_CUDA_VER}` if that directory exists
(falling back to `/usr/local/cuda`). So on a machine with both `cuda-13.2` and
`cuda-13.3` installed, `FAISS_CUDA_VER=13.3` builds against the 13.3 toolkit
automatically. If the `nvcc` found on `PATH` reports a different version than
requested, the scripts print a warning so you can correct `CUDA_HOME`.

---

## Library Naming Convention (C++ / Shared Object)

When building the C++ libraries directly (without a Python wheel), the following
naming scheme is used. The base names follow CMake target names; the build scripts
append `-{arch}-${FAISS_CUDA_TAG}`. The tables below show the current
`FAISS_CUDA_TAG=cu132`; the `cu132` portion changes with the variable.

### Standard (x86_64) build — `build_lib_x86_64.sh`

| Library | Filename | Notes |
|---------|----------|-------|
| Main C++ library | `libfaiss-x86_64-cu132.so` | `FAISS_OUTPUT_NAME=faiss-x86_64-cu132` |
| AVX2 variant | `libfaiss_avx2.so` | CPU SIMD opt-level (variant names not suffixed) |
| AVX512 variant | `libfaiss_avx512.so` | CPU SIMD opt-level (variant names not suffixed) |
| C API wrapper | `libfaiss_c-x86_64-cu132.so` | `FAISS_C_OUTPUT_NAME=faiss_c-x86_64-cu132` |
| C API AVX2 | `libfaiss_c_avx2.so` | Paired with `libfaiss_avx2` |
| cuVS companion | `libcuvs.so` | From `rapidsai/cuvs` |

> **Note:** `FAISS_OUTPUT_NAME` renames only the base `faiss`/`faiss_c` targets
> (faiss/CMakeLists.txt). The AVX2/AVX512 SIMD variants keep their conventional
> `libfaiss_avx2.so` / `libfaiss_avx512.so` names; `auditwheel` bundles them into
> the wheel regardless of filename.

### DGX Spark (aarch64, SM 121) build — `build_lib_aarch64.sh`

| Library | Filename | Notes |
|---------|----------|-------|
| Main C++ library | `libfaiss-aarch64-cu132.so` | `FAISS_OUTPUT_NAME=faiss-aarch64-cu132` |
| C API wrapper | `libfaiss_c-aarch64-cu132.so` | `FAISS_C_OUTPUT_NAME=faiss_c-aarch64-cu132` |
| cuVS companion | `libcuvs-spark.so` | From `zbrad/cuvs`, SM 121 only |

### DGX Spark Python wheel — `build_wheel_aarch64.sh`

| Artifact | Name | Notes |
|----------|------|-------|
| Python wheel | `faiss-gpu-cu132` | `FAISS_VARIANT=gpu-cu132` in `setup.py` |
| Built by | `build_wheel_aarch64.sh` | Orchestrates lib → pkg → wheel steps |
| Stage dir | `_libfaiss_stage_aarch64/` | Mirrors `_libfaiss_stage/` for x86_64 build |

The `FAISS_OUTPUT_NAME` and `FAISS_C_OUTPUT_NAME` cmake variables are defined in
`faiss/CMakeLists.txt` and `c_api/CMakeLists.txt` respectively and have no effect
when left unset (an unconfigured upstream build produces `libfaiss.so` /
`libfaiss_c.so`). Both the x86_64 and aarch64 build scripts set them so the two
variants' libraries never collide if installed side by side.

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
