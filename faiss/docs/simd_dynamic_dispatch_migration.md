# SIMD Dynamic Dispatch — Migration Guide

## Overview

Single Instruction, Multiple Data (SIMD) is used heavily in Faiss to speed up
many types of operations. This includes AVX2, AVX512 (various flavors) for
x86_64 CPUs and NEON, SVE for ARM CPUs. SIMD code that is run on a machine
that does not support it will crash with SIGILL (illegal instruction signal),
therefore it is important to select the right implementation for the current
machine.

Faiss is transitioning from a **monolithic SIMD** model to a **dynamic
dispatch** model. New code should be written with dynamic dispatch in mind.

In  the following we'll use AVX2 as a running example, but the same applies to
other SIMD flavors.

### Monolithic SIMD (legacy)

The whole library is compiled with e.g. `-mavx2`, which enables the `__AVX2__`
macro. Code explicitly optimized for AVX2 is guarded with `#ifdef __AVX2__`.
The compiler may also auto-vectorize scalar code using AVX2.

This is easy for developers but means Faiss must be compiled separately for
each SIMD level (AVX2, AVX512, AVX512_SPR). Clients must link to the right
variant.

### Why not just runtime-check and branch?

A tempting approach is:

```cpp
if (avx2_detected()) {
    // AVX2 intrinsics
} else {
    // scalar fallback
}
```

This does **not** work. When the file is compiled with `-mavx2`, the compiler
will auto-vectorize the scalar branch too, issuing AVX2 instructions even in
the "fallback" path — the `-mavx2` flag is a blanket permission for the
*entire translation unit* (`.cpp` file after preprocessing), not just specific
code blocks. The scalar branch will crash with SIGILL on non-AVX2 machines.

NOTE: yes there [are ways](https://gcc.gnu.org/onlinedocs/gcc/Function-Specific-Option-Pragmas.html)
to set flags specifically for a code block, but they are not portable.

This forces two requirements:

1. SIMD-optimized code must live in **separate compilation units** compiled
   with specific flags (`-mavx2`, `-mavx512f`, etc.)
2. Common code must be compiled **without** SIMD flags to prevent
   auto-vectorization

### Dynamic Dispatch (DD)

SIMD code is isolated in specific files (e.g., `distances_avx2.cpp`). Only
these files are compiled with `-mavx2`; common files get baseline flags
(`-mpopcnt -msse4 -mno-avx -mno-avx2` on x86; no special flags on ARM) so
the compiler cannot accidentally issue AVX2 instructions.

Of course none of these functions should use SIMD-specific data types like
`__m256` or `float32x4_t`.

Functions are tagged with a `SIMDLevel` template parameter (e.g.,
`fvec_L2sqr<SIMDLevel::AVX2>`). A non-templated wrapper dispatches at runtime
based on `SIMDConfig::level`. To avoid per-call dispatch overhead in tight
loops, [`with_simd_level`](https://github.com/facebookresearch/faiss/blob/645a742b5/faiss/impl/simd_dispatch.h#L135)
dispatches once for an entire code block:

```cpp
with_simd_level([&]<SIMDLevel SL>() {
    for (size_t i = 0; i < n; i++) {
        distances[i] = fvec_L2sqr<SL>(query, vectors + i * d, d);
    }
});
```

In DD mode, the build system passes `COMPILE_SIMD_AVX2` (and other
`COMPILE_SIMD_*` defines) to **all** source files in the target — not just the
SIMD TUs. This is what allows the common TU's dispatch switch to include a
`case SIMDLevel::AVX2:` that references `fvec_L2sqr<SIMDLevel::AVX2>`, even
though that specialization is defined in a different file. The `COMPILE_SIMD_*`
define is a **link-time promise** that the specialization will be found by the
linker.

At runtime, the SIMD level defaults to the highest available but can be
overridden via `SIMDConfig::set_level` or the `FAISS_SIMD_LEVEL` environment
variable (DD mode only).

## The Conversion Recipe

Every module conversion follows a 4-step recipe. We use a simple function as a
running example.

### Step 1: Templatize on SIMDLevel

Replace `#ifdef __AVX2__` guards with `template <SIMDLevel SL>` enclosed in `#ifdef COMPILE_SIMD_AVX2`.

Optionally replace raw intrinsic types (`__m256`, `float32x4x2_t`) with portable `simdlib` wrappers
(`simd8float32`, `simd8uint32`) where possible — see the note on simdlib
coverage below.

**Before:**

```cpp
// functions.h
void fvec_madd(size_t n, const float* a, float bf, const float* b, float* c);

// functions.cpp
void fvec_madd(size_t n, const float* a, float bf, const float* b, float* c) {
#ifdef __AVX2__
    // AVX2 implementation
#else
    for (size_t i = 0; i < n; i++)
        c[i] = a[i] + bf * b[i];
#endif
}
```

**After:**

```cpp
// functions.h
#include <faiss/utils/simd_levels.h>

template <SIMDLevel SL>
void fvec_madd(size_t n, const float* a, float bf, const float* b, float* c);

// functions.cpp
// Non-template public API (dispatches at runtime in DD mode)
void fvec_madd(size_t n, const float* a, float bf, const float* b, float* c);
```



### Step 2: Split into per-SIMD translation units

Move SIMD specializations into per-ISA `.cpp` files, each guarded by
`#ifdef COMPILE_SIMD_*`. The original file retains only the `NONE` (scalar)
specialization and the dispatch wrapper.

```cpp
// functions_avx2.cpp — compiled with -mavx2
#ifdef COMPILE_SIMD_AVX2
#include "functions.h"
#include <immintrin.h>

template <>
void fvec_madd<SIMDLevel::AVX2>(
        size_t n, const float* a, float bf, const float* b, float* c) {
    // AVX2 intrinsics implementation
}
#endif
```

```cpp
// functions.cpp — compiled with baseline flags (no SIMD)
#include "functions.h"
#include <faiss/impl/simd_dispatch.h>

namespace faiss {

template <>
void fvec_madd<SIMDLevel::NONE>(
        size_t n, const float* a, float bf, const float* b, float* c) {
    for (size_t i = 0; i < n; i++)
        c[i] = a[i] + bf * b[i];
}

void fvec_madd(size_t n, const float* a, float bf, const float* b, float* c) {
    with_simd_level([&]<SIMDLevel SL>() {
        fvec_madd<SL>(n, a, bf, b, c);
    });
}

} // namespace faiss
```

**Key rules for per-SIMD TUs:**

- Guard the entire file with `#ifdef COMPILE_SIMD_AVX2` (not `#ifdef __AVX2__`).
  `COMPILE_SIMD_*` is a build-system define — a **link-time promise** that the
  specialization will be available when linking. `__AVX2__` is a compiler
  define that says intrinsics are available in this TU. Both are true for SIMD
  TUs, but only `COMPILE_SIMD_*` is true for common TUs in DD mode. This is
  what allows the common TU to reference `fvec_madd<SIMDLevel::AVX2>` without
  defining it — `COMPILE_SIMD_AVX2` promises the linker will find it.
- Include the ISA-specific intrinsic headers (`<immintrin.h>`,
  `<arm_neon.h>`, `<arm_sve.h>`) inside the guard.
- Never pass SIMD data types (`__m256`, `float32x4_t`) across TU boundaries.
  The public interface uses plain `float*`, `uint8_t*`, etc.

### Step 3: Wire up dispatch

Use `with_simd_level` at
the **dispatch boundary** — the point where a hot kernel or factory function is
first called. Dispatch happens once, outside inner loops.

For factory functions that return objects (distance computers, scanners), the `SIMDLevel` is passed as a template parameter into the object at construction time:

```cpp
FlatCodesDistanceComputer* get_distance_computer() {
    return with_simd_level([&]<SIMDLevel SL>() -> FlatCodesDistanceComputer* {
        FlatCodesDistanceComputer * dc = new MyDistanceComputer<SL>();
        // ... configure ...
        return dc;
    });
}
```

**Dispatch masks:** `with_simd_level` assumes NONE + AVX2 + AVX512 +
ARM_NEON implementations exist. If your function has another subset of available
implementations, it can be passed with
`with_selected_simd_levels<mask>` with a bitmask of available levels. Missing
levels in the mask cause the dispatch to **fall through** to the next lower
level in the same architecture family (x86: AVX512_SPR → AVX512 → AVX2 →
NONE; ARM: ARM_SVE → ARM_NEON → NONE — x86 and ARM chains are independent):

```cpp
// Only NONE, AVX2, and ARM_SVE implementations exist.
// AVX512 will fall through to AVX2; ARM_NEON will fall through to NONE.
constexpr int MY_LEVELS = (1 << int(SIMDLevel::NONE))
                        | (1 << int(SIMDLevel::AVX2))
                        | (1 << int(SIMDLevel::ARM_SVE));

with_selected_simd_levels<MY_LEVELS>([&]<SIMDLevel SL>() {
    my_kernel<SL>(...);
});
```

Predefined convenience masks in `simd_dispatch.h` (you can always construct
your own with `(1 << int(SIMDLevel::X)) | ...`):

| Mask | Levels | Use for |
|------|--------|---------|
| `AVAILABLE_SIMD_LEVELS_NONE` | NONE only | Scalar-only functions |
| `AVAILABLE_SIMD_LEVELS_AVX2_NEON` | NONE, AVX2, ARM_NEON | 256-bit `simdlib` ops (`with_simd_level_256bit`) |
| `AVAILABLE_SIMD_LEVELS_A0` | NONE, AVX2, AVX512, ARM_NEON | Default (`with_simd_level`) |
| `AVAILABLE_SIMD_LEVELS_A1` | A0 + ARM_SVE | Functions with dedicated SVE implementations |
| `AVAILABLE_SIMD_LEVELS_ALL` | All levels | Identity / diagnostic functions |

### Step 4: Register in build system

Add new source files to the build system. Edit `faiss/CMakeLists.txt` (not the
top-level `CMakeLists.txt`).

#### CMake

```cmake
set(FAISS_SIMD_AVX2_SRC
  # ... existing entries ...
  path/to/functions_avx2.cpp        # <-- add
)
set(FAISS_SIMD_AVX512_SRC
  # ... existing entries ...
  path/to/functions_avx512.cpp      # <-- add
)
set(FAISS_SIMD_NEON_SRC
  # ... existing entries ...
  path/to/functions_neon.cpp        # <-- add
)
set(FAISS_SIMD_SVE_SRC
  # ... existing entries ...
  path/to/functions_sve.cpp         # <-- add (if SVE implementation exists)
)
# Also add any new headers to FAISS_HEADERS
```

To enable DD mode with CMake, configure with `FAISS_OPT_LEVEL=dd`:

```bash
cmake -B build_dd -DFAISS_OPT_LEVEL=dd -DBUILD_TESTING=ON .
```

This builds a single `faiss` library with all SIMD variants compiled in and
runtime dispatch enabled. Compare with the monolithic modes (`avx2`, `avx512`,
`generic`) which build separate per-level libraries.

Remove the old monolithic source files from `FAISS_SRC` if they were there.

#### Buck (Meta internal)

```python
# In xplat.bzl:
SIMD_FILES = {
    # ... existing entries ...
    "path/to/functions_avx2.cpp": (X86_64, AVX2),
    "path/to/functions_avx512.cpp": (X86_64, AVX512),
    "path/to/functions_neon.cpp": (AARCH64, ARM_NEON),
}
# Also add headers to header_files()
```

The two build systems must stay in sync — both have `# Keep in sync` comments
referencing each other.

## Common Patterns

### Shared kernel code across ISA levels

When the same kernel code works across multiple ISA levels, it can be shared
between per-ISA `.cpp` files. Sometimes this is done via a shared `-inl.h`
header. The key is that each per-ISA TU compiles the shared code with its own
SIMD flags.
When using a shared `-inl.h`, each per-ISA `.cpp` includes the `-inl.h`, setting
a macro `THE_SIMD_LEVEL` that indicates which template to instanciate.



### Two ways to trigger template instantiation

The per-ISA `.cpp` file needs to cause the compiler to emit the SIMD-specific
code. There are two patterns used in the codebase:

**Explicit template instantiation** — list the concrete types/functions to
instantiate. Used when you have a small, known set of template arguments. See
[`approx_topk/avx2.cpp`](https://github.com/facebookresearch/faiss/blob/645a742b5/faiss/impl/approx_topk/avx2.cpp#L19-L23)
for an example:

```cpp
// avx2.cpp
template struct HeapWithBucketsCMaxFloat<8, 3, SIMDLevel::AVX2>;
template struct HeapWithBucketsCMaxFloat<16, 2, SIMDLevel::AVX2>;
// ...
```

**Macro-driven definition** — `#define` a level constant, then `#include` a
dispatch header that stamps out specializations for that level. Used when
many interdependent types need to be instantiated together. See
[`sq-avx2.cpp`](https://github.com/facebookresearch/faiss/blob/645a742b5/faiss/impl/scalar_quantizer/sq-avx2.cpp#L456-L457)
and
[`distances_avx2.cpp`](https://github.com/facebookresearch/faiss/blob/645a742b5/faiss/utils/simd_impl/distances_avx2.cpp#L12)
for examples:

```cpp
// sq-avx2.cpp
#define THE_LEVEL_TO_DISPATCH SIMDLevel::AVX2
#include <faiss/impl/scalar_quantizer/sq-dispatch.h>

// distances_avx2.cpp
#define THE_SIMD_LEVEL SIMDLevel::AVX2
#include <faiss/utils/simd_impl/distances_autovec-inl.h>
```

### Auto-vectorization via shared inline header

For functions where the compiler's auto-vectorization is sufficient, a scalar
loop in a shared `-inl.h` can be compiled at different ISA levels. The
`FAISS_PRAGMA_IMPRECISE_LOOP` macro enables fast-math-style optimizations for
the annotated loop, improving auto-vectorization quality:

```cpp
// autovec-inl.h
template <SIMDLevel SL>
float fvec_L2sqr_autovec(const float* x, const float* y, size_t d) {
    float result = 0;
    FAISS_PRAGMA_IMPRECISE_LOOP
    for (size_t i = 0; i < d; i++) {
        float diff = x[i] - y[i];
        result += diff * diff;
    }
    return result;
}

// distances_avx2.cpp — compiled with -mavx2, auto-vectorizer uses AVX2
template <>
float fvec_L2sqr<SIMDLevel::AVX2>(...) {
    return fvec_L2sqr_autovec<SIMDLevel::AVX2>(...);
}
```

### simdlib coverage

The `simdlib` wrappers (`simd8float32_tpl`,
`simd8uint32_tpl`) provide portable 256-bit and 512-bit operations
across AVX2, AVX512 and NEON (two 128 bit NEON registers are clumped
together in 256 bits)
There is **no simdlib for SVE** (`simdlib_sve.h` does not exist).
Use raw intrinsics when you need SVE
(variable-length vectors via `svcntw()`).
An example of usage is with `-inl.h` files

**The include order matters** —
the `-inl.h` needs the ISA-specific specializations of
`simd8float32_tpl<SL>` to be visible:

```cpp
// kernel-inl.h — shared kernel using simd8float32_tpl<SL>
template <size_t DIM, SIMDLevel SL>
void my_kernel(...) {
    using simd_float = simd8float32_tpl<SL>;
    // ... kernel using simd_float ...
}

// avx2.cpp
#include <faiss/impl/simdlib/simdlib_avx2.h>  // must come first
#include "kernel-inl.h"   // sees AVX2 specialization of simd8float32_tpl
// trigger instantiation here (see below)

// neon.cpp
#include <faiss/impl/simdlib/simdlib_neon.h>   // must come first
#include "kernel-inl.h"   // sees NEON specialization of simd8float32_tpl
// trigger instantiation here (see below)
```


## Key Principles

1. **Static dispatch is zero-overhead.** In static mode, `with_simd_level`
   compiles to a direct call — no switch, no indirection.

2. **Per-file SIMD flags.** Only SIMD TUs get `-mavx2` etc. Common files get
   baseline flags to prevent auto-vectorization that could cause SIGILL.

3. **`COMPILE_SIMD_*` over compiler macros.** Use `COMPILE_SIMD_AVX2` (build
   system control) instead of `__AVX2__` (compiler feature detection) for
   dispatch guards. Lower-level headers like `simdlib.h` still use compiler
   macros for intrinsic type selection.

4. **Dispatch once, outside the loop.** Call `with_simd_level` at the
   factory/constructor boundary. The constructed object carries its
   `SIMDLevel` as a compile-time template parameter.

5. **Private dispatch machinery.** `simd_dispatch.h` is internal — do not
   include in public headers. The public API is `SIMDConfig` and `SIMDLevel`
   in `utils/simd_levels.h`.

6. **Build system parity.** Every change must be reflected in both
   CMakeLists.txt and Buck's xplat.bzl.

## Conversion approach

The conversion to dynamic dispatch is progressive, over several months. During the process,
the static dispatch should continue functioning and there should be no performance degradation.

`SINGLE_SIMD_LEVEL` is a compile-time constant from
[`simd_levels.h`](https://github.com/facebookresearch/faiss/blob/645a742b5/faiss/utils/simd_levels.h#L33-L48)
that equals the compiled-in SIMD level in static mode, or `NONE` in DD mode
on x86 / `ARM_NEON` on ARM. It exists as migration scaffolding so that
pre-DD code compiles unchanged in DD mode. The converted DD code should use
`with_simd_level` dispatch, not rely on the default template parameter.


## Modules Converted

| Module | Per-SIMD files | Key pattern |
|--------|---------------|-------------|
| Distance functions | [`distances_avx2.cpp#L12`](https://github.com/facebookresearch/faiss/blob/645a742b5/faiss/utils/simd_impl/distances_avx2.cpp#L12-L14) | Auto-vectorization via `THE_SIMD_LEVEL` + shared `-inl.h` |
| PQ code distance | [`pq_code_distance-avx2.cpp#L164`](https://github.com/facebookresearch/faiss/blob/645a742b5/faiss/impl/pq_code_distance/pq_code_distance-avx2.cpp#L164-L165) | Struct template as dispatch unit |
| ScalarQuantizer | [`sq-avx2.cpp#L456`](https://github.com/facebookresearch/faiss/blob/645a742b5/faiss/impl/scalar_quantizer/sq-avx2.cpp#L456-L457) | X-macro dispatch header (`sq-dispatch.h`) |
| PQ4 fast scan | [`impl-avx2.cpp#L10`](https://github.com/facebookresearch/faiss/blob/645a742b5/faiss/impl/fast_scan/impl-avx2.cpp#L10-L12) | `THE_LEVEL_TO_DISPATCH` + scanner dispatch |
| Approximate top-k | [`approx_topk/avx2.cpp#L19`](https://github.com/facebookresearch/faiss/blob/645a742b5/faiss/impl/approx_topk/avx2.cpp#L19-L23) | Explicit template instantiation of `simdlib`-based class |

## Testing

Build and test in multiple modes to catch DD issues.

### CMake

```bash
# Build with dynamic dispatch
cmake -B build_dd -DFAISS_OPT_LEVEL=dd -DBUILD_TESTING=ON .
cmake --build build_dd -j$(nproc)

# Run C++ tests
cd build_dd && ctest --output-on-failure

# Verify dispatch at different levels (DD mode only)
FAISS_SIMD_LEVEL=NONE ctest --output-on-failure
FAISS_SIMD_LEVEL=AVX2 ctest --output-on-failure

# Also build/test static modes for comparison
cmake -B build_avx2 -DFAISS_OPT_LEVEL=avx2 -DBUILD_TESTING=ON .
cmake --build build_avx2 -j$(nproc) && cd build_avx2 && ctest --output-on-failure
```

### Buck (internal)

```bash
# Static (default AVX2)
buck2 build fbcode//faiss:faiss
buck2 test fbcode//faiss/tests:test_your_module

# Dynamic Dispatch
buck2 build -c faiss.dynamic_dispatch=true fbcode//faiss:faiss
buck2 test -c faiss.dynamic_dispatch=true fbcode//faiss/tests:test_your_module
```

### Common pitfalls

- Adding a SIMD file to one source list but forgetting the others (e.g.,
  `FAISS_SIMD_AVX2_SRC` but not `FAISS_SIMD_NEON_SRC` for ARM) — causes
  linker errors on the missing architecture.
- Using `#ifdef __AVX2__` instead of `#ifdef COMPILE_SIMD_AVX2` in dispatch
  guards — the SIMD code silently disappears in DD mode because `__AVX2__` is
  only defined in TUs compiled with `-mavx2`, not in common TUs.
- Forgetting the `NONE` specialization — linker error in scalar / generic
  builds.
- Dispatching inside a hot loop instead of once at the boundary — measurable
  overhead from the switch on `SIMDConfig::level`.
- Building with CMake's default `FAISS_OPT_LEVEL=generic` and thinking DD is
  enabled — generic mode has no SIMD and no dispatch. Use
  `FAISS_OPT_LEVEL=dd` explicitly.
