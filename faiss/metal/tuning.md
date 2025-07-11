# Metal Performance Tuning

This document tracks performance tuning efforts for the Metal backend.

## Areas for Improvement

*   **Kernel Launch Overhead**: Investigate ways to reduce the overhead of launching compute kernels.
*   **Threadgroup Sizing**: Tune threadgroup sizes for optimal performance on different Apple Silicon chips.
*   **Memory Access Patterns**: Profile memory access patterns and optimize for coalesced access.
*   **SIMD Group Operations**: Utilize `simdgroup_matrix` and other SIMD-group functions for matrix multiplication and other operations.
