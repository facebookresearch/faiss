# Metal Error Handling and Diagnostics

This document describes error handling and diagnostics for the Metal backend.

## Error Handling

-   Metal API calls can return `NSError` objects. These should be checked and converted to `faiss::FaissException`.
-   Shader compilation errors should be logged with as much detail as possible.
-   The Metal validation layer should be enabled in debug builds to catch errors early.

## Diagnostics

-   The Metal System Trace tool in Xcode can be used to profile and debug Metal applications.
-   GPU counters can be used to get detailed performance metrics.
