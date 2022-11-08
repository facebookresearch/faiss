# Changelog
All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project DOES NOT adhere to [Semantic Versioning](https://semver.org/spec/v2.0.0.html)
at the moment.

We try to indicate most contributions here with the contributor names who are not part of
the Facebook Faiss team.  Feel free to add entries here if you submit a PR.

## [Unreleased]

## [1.7.3] - 2022-11-3
### Added
- Added sparse k-means routines and moved the generic kmeans to contrib
- Added FlatDistanceComputer for all FlatCodes indexes
- Support for fast accumulation of 4-bit LSQ and RQ
- Added product additive quantization
- Support per-query search parameters for many indexes + filtering by ids
- write_VectorTransform and read_vectorTransform were added to the public API (by @AbdelrahmanElmeniawy)
- Support for IDMap2 in index_factory by adding "IDMap2" to prefix or suffix of the input String (by @AbdelrahmanElmeniawy)
- Support for merging all IndexFlatCodes descendants (by @AbdelrahmanElmeniawy)
- Remove and merge features for IndexFastScan (by @AbdelrahmanElmeniawy)
- Performance improvements: 1) specialized the AVX2 pieces of code speeding up certain hotspots, 2) specialized kernels for vector codecs (this can be found in faiss/cppcontrib)


### Fixed
- Fixed memory leak in OnDiskInvertedLists::do_mmap when the file is not closed (by @AbdelrahmanElmeniawy)
- LSH correctly throws error for metric types other than METRIC_L2 (by @AbdelrahmanElmeniawy)

## [1.7.2] - 2021-12-15
### Added
- Support LSQ on GPU (by @KinglittleQ)
- Support for exact 1D kmeans (by @KinglittleQ)

## [1.7.1] - 2021-05-27
### Added
- Support for building C bindings through the `FAISS_ENABLE_C_API` CMake option.
- Serializing the indexes with the python pickle module
- Support for the NNDescent k-NN graph building method (by @KinglittleQ)
- Support for the NSG graph indexing method (by @KinglittleQ)
- Residual quantizers: support as codec and unoptimized search
- Support for 4-bit PQ implementation for ARM (by @vorj, @n-miyamoto-fixstars, @LWisteria, and @matsui528)
- Implementation of Local Search Quantization (by @KinglittleQ)

### Changed
- The order of xb an xq was different between `faiss.knn` and `faiss.knn_gpu`.
Also the metric argument was called distance_type.
- The typed vectors (LongVector, LongLongVector, etc.) of the SWIG interface have
been deprecated. They have been replaced with Int32Vector, Int64Vector, etc. (by h-vetinari)

### Fixed
- Fixed a bug causing kNN search functions for IndexBinaryHash and
IndexBinaryMultiHash to return results in a random order.
- Copy constructor of AlignedTable had a bug leading to crashes when cloning
IVFPQ indices.

## [1.7.0] - 2021-01-27

## [1.6.5] - 2020-11-22

## [1.6.4] - 2020-10-12
### Added
- Arbitrary dimensions per sub-quantizer now allowed for `GpuIndexIVFPQ`.
- Brute-force kNN on GPU (`bfKnn`) now accepts `int32` indices.
- Nightly conda builds now available (for CPU).
- Faiss is now supported on Windows.

## [1.6.3] - 2020-03-24
### Added
- Support alternative distances on GPU for GpuIndexFlat, including L1, Linf and
Lp metrics.
- Support METRIC_INNER_PRODUCT for GpuIndexIVFPQ.
- Support float16 coarse quantizer for GpuIndexIVFFlat and GpuIndexIVFPQ. GPU
Tensor Core operations (mixed-precision arithmetic) are enabled on supported
hardware when operating with float16 data.
- Support k-means clustering with encoded vectors. This makes it possible to
train on larger datasets without decompressing them in RAM, and is especially
useful for binary datasets (see https://github.com/facebookresearch/faiss/blob/main/tests/test_build_blocks.py#L92).
- Support weighted k-means. Weights can be associated to each training point
(see https://github.com/facebookresearch/faiss/blob/main/tests/test_build_blocks.py).
- Serialize callback in python, to write to pipes or sockets (see
https://github.com/facebookresearch/faiss/wiki/Index-IO,-cloning-and-hyper-parameter-tuning).
- Reconstruct arbitrary ids from IndexIVF + efficient remove of a small number
of ids. This avoids 2 inefficiencies: O(ntotal) removal of vectors and
IndexIDMap2 on top of indexIVF. Documentation here:
https://github.com/facebookresearch/faiss/wiki/Special-operations-on-indexes.
- Support inner product as a metric in IndexHNSW (see
https://github.com/facebookresearch/faiss/blob/main/tests/test_index.py#L490).
- Support PQ of sizes other than 8 bit in IndexIVFPQ.
- Demo on how to perform searches sequentially on an IVF index. This is useful
for an OnDisk index with a very large batch of queries. In that case, it is
worthwhile to scan the index sequentially (see
https://github.com/facebookresearch/faiss/blob/main/tests/test_ivflib.py#L62).
- Range search support for most binary indexes.
- Support for hashing-based binary indexes (see
https://github.com/facebookresearch/faiss/wiki/Binary-indexes).

### Changed
- Replaced obj table in Clustering object: now it is a ClusteringIterationStats
structure that contains additional statistics.

### Removed
- Removed support for useFloat16Accumulator for accumulators on GPU (all
accumulations are now done in float32, regardless of whether float16 or float32
input data is used).

### Fixed
- Some python3 fixes in benchmarks.
- Fixed GpuCloner (some fields were not copied, default to no precomputed tables
with IndexIVFPQ).
- Fixed support for new pytorch versions.
- Serialization bug with alternative distances.
- Removed test on multiple-of-4 dimensions when switching between blas and AVX
implementations.

## [1.6.2] - 2020-03-10

## [1.6.1] - 2019-12-04

## [1.6.0] - 2019-09-24
### Added
- Faiss as a codec: We introduce a new API within Faiss to encode fixed-size
vectors into fixed-size codes. The encoding is lossy and the tradeoff between
compression and reconstruction accuracy can be adjusted.
- ScalarQuantizer support for GPU, see gpu/GpuIndexIVFScalarQuantizer.h. This is
particularly useful as GPU memory is often less abundant than CPU.
- Added easy-to-use serialization functions for indexes to byte arrays in Python
(faiss.serialize_index, faiss.deserialize_index).
- The Python KMeans object can be used to use the GPU directly, just add
gpu=True to the constuctor see gpu/test/test_gpu_index.py test TestGPUKmeans.

### Changed
- Change in the code layout: many C++ sources are now in subdirectories impl/
and utils/.

## [1.5.3] - 2019-06-24
### Added
- Basic support for 6 new metrics in CPU IndexFlat and IndexHNSW (https://github.com/facebookresearch/faiss/issues/848).
- Support for IndexIDMap/IndexIDMap2 with binary indexes (https://github.com/facebookresearch/faiss/issues/780).

### Changed
- Throw python exception for OOM (https://github.com/facebookresearch/faiss/issues/758).
- Make DistanceComputer available for all random access indexes.
- Gradually moving from long to uint64_t for portability.

### Fixed
- Slow scanning of inverted lists (https://github.com/facebookresearch/faiss/issues/836).

## [1.5.2] - 2019-05-28
### Added
- Support for searching several inverted lists in parallel (parallel_mode != 0).
- Better support for PQ codes where nbit != 8 or 16.
- IVFSpectralHash implementation: spectral hash codes inside an IVF.
- 6-bit per component scalar quantizer (4 and 8 bit were already supported).
- Combinations of inverted lists: HStackInvertedLists and VStackInvertedLists.
- Configurable number of threads for OnDiskInvertedLists prefetching (including
0=no prefetch).
- More test and demo code compatible with Python 3 (print with parentheses).

### Changed
- License was changed from BSD+Patents to MIT.
- Exceptions raised in sub-indexes of IndexShards and IndexReplicas are now
propagated.
- Refactored benchmark code: data loading is now in a single file.

## [1.5.1] - 2019-04-05
### Added
- MatrixStats object, which reports useful statistics about a dataset.
- Option to round coordinates during k-means optimization.
- An alternative option for search in HNSW.
- Support for range search in IVFScalarQuantizer.
- Support for direct uint_8 codec in ScalarQuantizer.
- Better support for PQ code assignment with external index.
- Support for IMI2x16 (4B virtual centroids).
- Support for k = 2048 search on GPU (instead of 1024).
- Support for renaming an ondisk invertedlists.
- Support for nterrupting computations with interrupt signal (ctrl-C) in python.
- Simplified build system (with --with-cuda/--with-cuda-arch options).

### Changed
- Moved stats() and imbalance_factor() from IndexIVF to InvertedLists object.
- Renamed IndexProxy to IndexReplicas.
- Most CUDA mem alloc failures now throw exceptions instead of terminating on an
assertion.
- Updated example Dockerfile.
- Conda packages now depend on the cudatoolkit packages, which fixes some
interferences with pytorch. Consequentially, faiss-gpu should now be installed
by conda install -c pytorch faiss-gpu cudatoolkit=10.0.

## [1.5.0] - 2018-12-19
### Added
- New GpuIndexBinaryFlat index.
- New IndexBinaryHNSW index.

## [1.4.0] - 2018-08-30
### Added
- Automatic tracking of C++ references in Python.
- Support for non-intel platforms, some functions optimized for ARM.
- Support for overriding nprobe for concurrent searches.
- Support for floating-point quantizers in binary indices.

### Fixed
- No more segfaults due to Python's GC.
- GpuIndexIVFFlat issues for float32 with 64 / 128 dims.
- Sharding of flat indexes on GPU with index_cpu_to_gpu_multiple.

## [1.3.0] - 2018-07-10
### Added
- Support for binary indexes (IndexBinaryFlat, IndexBinaryIVF).
- Support fp16 encoding in scalar quantizer.
- Support for deduplication in IndexIVFFlat.
- Support for index serialization.

### Fixed
- MMAP bug for normal indices.
- Propagation of io_flags in read func.
- k-selection for CUDA 9.
- Race condition in OnDiskInvertedLists.

## [1.2.1] - 2018-02-28
### Added
- Support for on-disk storage of IndexIVF data.
- C bindings.
- Extended tutorial to GPU indices.

[Unreleased]: https://github.com/facebookresearch/faiss/compare/v1.7.2...HEAD
[1.7.3]: https://github.com/facebookresearch/faiss/compare/v1.7.2...v1.7.3
[1.7.2]: https://github.com/facebookresearch/faiss/compare/v1.7.1...v1.7.2
[1.7.1]: https://github.com/facebookresearch/faiss/compare/v1.7.0...v1.7.1
[1.7.0]: https://github.com/facebookresearch/faiss/compare/v1.6.5...v1.7.0
[1.6.5]: https://github.com/facebookresearch/faiss/compare/v1.6.4...v1.6.5
[1.6.4]: https://github.com/facebookresearch/faiss/compare/v1.6.3...v1.6.4
[1.6.3]: https://github.com/facebookresearch/faiss/compare/v1.6.2...v1.6.3
[1.6.2]: https://github.com/facebookresearch/faiss/compare/v1.6.1...v1.6.2
[1.6.1]: https://github.com/facebookresearch/faiss/compare/v1.6.0...v1.6.1
[1.6.0]: https://github.com/facebookresearch/faiss/compare/v1.5.3...v1.6.0
[1.5.3]: https://github.com/facebookresearch/faiss/compare/v1.5.2...v1.5.3
[1.5.2]: https://github.com/facebookresearch/faiss/compare/v1.5.1...v1.5.2
[1.5.1]: https://github.com/facebookresearch/faiss/compare/v1.5.0...v1.5.1
[1.5.0]: https://github.com/facebookresearch/faiss/compare/v1.4.0...v1.5.0
[1.4.0]: https://github.com/facebookresearch/faiss/compare/v1.3.0...v1.4.0
[1.3.0]: https://github.com/facebookresearch/faiss/compare/v1.2.1...v1.3.0
[1.2.1]: https://github.com/facebookresearch/faiss/releases/tag/v1.2.1
