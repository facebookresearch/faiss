# Branch Changes Summary (faiss-gpu-cu132)

What this branch adds on top of `upstream/main`, and the conventions a reviewer
should know. Everything lives under [`gpu-cu/`](../) plus a few small,
backward-compatible hooks in the upstream build files. There are **no FAISS
algorithm changes** — only packaging metadata and CMake output-name options.

## What it adds

An optional, self-contained GPU **wheel build toolkit** under `gpu-cu/`:

```
gpu-cu/
├── scripts/        build + packaging scripts (see below)
│   └── cuda_env.sh   single source of truth for the CUDA version
├── wsl/            WSL-on-Windows entry path (env.sh, build.sh, verify.sh)
├── docs/           BUILD_arch_x86_64, BUILD_arch_aarch64, WHEEL_NAMING, this file, …
├── environment.yml conda environment
└── verify_environment.py
```

Standard FAISS usage is unaffected; none of this builds unless invoked.

## Key conventions

**Two architecture pipelines** (named by CPU arch, not by platform):

| | x86_64 | aarch64 / DGX Spark |
|---|---|---|
| Scripts | `*_x86_64.sh` | `*_aarch64.sh` |
| BLAS / accel | Intel MKL, AVX2/AVX512 | OpenBLAS + cuVS, SVE |
| GPU arch | 75–120 (multi) | SM 121 only |
| C++ libs | `libfaiss-x86_64-${TAG}.so` | `libfaiss-aarch64-${TAG}.so` |

**CUDA version is a single build-time input.** `gpu-cu/scripts/cuda_env.sh`
(`FAISS_CUDA_VER` / `FAISS_CUDA_TAG`, mirrored by Makefile vars) drives every
derived name. Specify either form per build — the other is derived:

```bash
make build FAISS_CUDA_VER=13.3        # x86_64 → faiss-gpu-cu133
make build-aarch64 FAISS_CUDA_VER=13.3
```

On a host with multiple toolkits, `CUDA_HOME` auto-resolves to
`/usr/local/cuda-<ver>` (override to force a path); a mismatched `nvcc` warns.

**Wheel naming** (details in [WHEEL_NAMING.md](WHEEL_NAMING.md)):
- CPU arch (x86_64 / aarch64) is carried by the wheel **platform tag**, not the name.
- Package name is `faiss-gpu-${TAG}` — shared across CPU arches for portable builds.
- A build targeting a **single GPU arch** appends `-sm<arch>` (e.g.
  `faiss-gpu-cu133-sm121` for DGX Spark, `-sm89` for an Ada-only x86 build).

## Upstream touch points (backward-compatible)

- `faiss/python/setup.py` — `FAISS_VARIANT` → package name; arch/CUDA metadata in
  the wheel description (arch detected via `platform.machine()`).
- `faiss/CMakeLists.txt` — `FAISS_OUTPUT_NAME`, `FAISS_CUVS_SPARK_LIBRARY` (no-ops when unset).
- `c_api/CMakeLists.txt` — `FAISS_C_OUTPUT_NAME` (no-op when unset).
- `Makefile` — `build` / `build-aarch64` targets delegating to `gpu-cu/scripts/`.
- `.github/workflows/create-gpu-release.yml` — publishes `gpu-cu/RELEASE_NOTES.md` on tag push.
- `README.md`, `INSTALL.md` — point at the toolkit and split BLAS guidance by arch.

## Review focus

1. **Version knob**: `cuda_env.sh` derivation + `CUDA_HOME` resolution on a
   multi-toolkit host; Makefile passes `FAISS_CUDA_VER` / `CUDA_ARCHS` through.
2. **Naming**: platform-tag selection for portable wheels; `-sm<arch>` only on
   single-arch builds (`faiss_sm_suffix` in `cuda_env.sh`).
3. **Build paths**: `Makefile` and `gpu-cu/wsl/*.sh` resolve `FAISS_ROOT` and
   call the renamed `*_x86_64.sh` / `*_aarch64.sh` scripts.
4. **Release workflow**: tag-triggered notes publishing.
