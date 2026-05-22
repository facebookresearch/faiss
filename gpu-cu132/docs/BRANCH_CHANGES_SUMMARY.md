# Branch Changes Summary (faiss-gpu-cu132)

This summary captures semantic (non-whitespace) branch changes against `upstream/main` and the file reorganization performed for this branch.

## Semantic Change Scope

- Total semantic files changed: 30
- Total semantic diff: 2,897 insertions, 25 deletions
- Core FAISS algorithm scope: no algorithmic changes in `faiss/` except packaging metadata in `faiss/python/setup.py`

## Changes By Top-Level Area

- `(root)` / `gpu-cu132`: CUDA 13.2 build + packaging scripts, release notes, test results, setup guides, environment files
- `scripts/wsl`: WSL build and verification orchestration
- `docs/branches/faiss-gpu-cu132`: branch-specific build and setup documentation
- `.github/workflows`: release workflow to publish notes on tag push
- `.vscode`: terminal profile settings for WSL automation
- `faiss/python/setup.py`: wheel metadata and variant information
- `faiss/CMakeLists.txt`: added `FAISS_OUTPUT_NAME` and `FAISS_CUVS_SPARK_LIBRARY` cmake options
- `c_api/CMakeLists.txt`: added `FAISS_C_OUTPUT_NAME` cmake option
- `gpu-cu132/scripts/build_lib_spark.sh`: DGX Spark C++ lib — SM 121, aarch64, `libcuvs-spark`, produces `libfaiss-spark-cu132.so` / `libfaiss_c-spark-cu132.so`
- `gpu-cu132/scripts/build_pkg_spark.sh`: DGX Spark SWIG Python bindings
- `gpu-cu132/scripts/package_wheel_spark.sh`: DGX Spark wheel packaging (`faiss-gpu-cu132-spark`)
- `gpu-cu132/scripts/build_wheel_spark.sh`: Unified DGX Spark wheel orchestrator

## Reorganization Performed

Branch-specific files were moved from repository root to `gpu-cu132/`:

- `BUILD_MANIFEST.txt`
- `QUICKSTART.md`
- `SETUP_COMPLETE.md`
- `RELEASE_NOTES.md`
- `TEST_RESULTS.md`
- `build_lib_cuda132.sh`
- `build_pkg_cuda132.sh`
- `build_wheel.sh`
- `clean_build.sh`
- `environment_cuda132_py314.yml`
- `fix_line_endings.ps1`
- `package_wheel.sh`
- `test_install.sh`
- `verify_environment.py`
- `wsl_build.sh`

## Path Updates Completed

References were updated across:

- `Makefile` targets now call `gpu-cu132/*` scripts
- `.github/workflows/create-gpu-release.yml` now reads `gpu-cu132/RELEASE_NOTES.md`
- `gpu-cu132/wsl/build.sh` now normalizes CRLF for `gpu-cu132/*` scripts
- Root and branch docs now reference `gpu-cu132/*` file locations
- Moved scripts now resolve repository root correctly via `FAISS_ROOT`

## Review Focus Recommendations

When reviewing this branch, prioritize:

1. Build execution path correctness from `Makefile` and `gpu-cu132/wsl/*.sh`
2. Release workflow behavior for tag-triggered releases
3. Documentation command correctness after path changes
4. Wheel output location consistency (`build_output/`)
