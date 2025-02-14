# Changelog
All notable changes to this project will be documented in this file.

## [Unreleased]

## [1.10.0] - 2025-01-30


Added
- Add desc_name to dataset descriptor (#3935)
- implement ST_norm_from_LUT for the ResidualQuantizer (#3917)
- Add example of how to build, link, and test an external SWIG module (#3922)
- add copyright header (#3948)
- Add some SVE implementations (#3933)
- Enable linting: lint config changes plus arc lint command (#3966)
- Re-add example of how to build, link, and test an external SWIG module (#3981)
- demo: IndexPQ: separate codes from codebook (#3987)
- add all wrapped indexes to the index_read (#3988)
- add validity check AlignedTableTightAlloc clear method (#3997)
- Add index binary to telemetry (#4001)
- Add VectorTransform read from filename to the C API (#3970)
- Added IndexLSH to the demo (#4009)
- write distributed_kmeans centroids and assignments to hive tables (#4017)
- introduce data splits in dataset descriptor (#4012)
- Faiss GPU: bfloat16 brute-force kNN support (#4018)
- ROCm support for bfloat16 (#4039)
- Unit tests for distances_simd.cpp (#4058)
- add cuda-toolkit for GPU (#4057)
- Add more unit testing for IndexHNSW [1/n] (#4054)
- Add more unit testing for IndexHNSW [2/n] (#4056)
- Add more unit testing for HNSW [3/n] (#4059)
- Add more unit testing for HNSW [4/n] (#4061)
- Add more unit tests for index_read and index_write (#4068)
- Add testing for utils/hamming.cpp (#4079)
- Test sa_decode methd on IndexIVFFlat (#4098)
- Conditionally compile extras like benchmarks and demos (#4094)
- Add a new architecture mode: 'avx512_spr'. (#4025)
- Use _mm512_popcnt_epi64 to speedup hamming distance evaluation. (#4020)
- PQ with pytorch (#4116)
- add range_search() to IndexRefine (#4022)
- Expose accumulate_to_mem from faiss interface (#4099)
- Windows Arm64 support (#4087)
- add test to cover GPU (#4130)
- Added support for building without MKL (#4147)

Changed
- Move train, build and search to their respective operators (#3934)
- PQFS into Index trainer (#3941)
- Place a useful cmake function 'link_to_faiss_lib' into a separate file (#3939)
- Cache device major version value to avoid multiple calls of getCudaDeviceProperties (#3950)
- Consolidate set_target_properties() calls in faiss/CMakeLists.txt (#3973)
- Removing Manual Hipify Build Step (#3962)
- Allow to replace graph structure for NSG graphs (#3975)
- Adjust nightly build (#3978)
- Update RAFT CI with pytorch 2.4.1 (#3980)
- Moved add_sa_codes, sa_code_size to Index, IndexBinary base classes (#3989)
- Update autoclose.yml (#4000)
- Migrate from RAFT to CUVS (#3549)
- Pin to numpy<2 (#4033)
- (1/n) - Preload datasets in manifold so that subsequent stages of training, indexing and search can use those instead of each trainer or indexer downloading data. (#4034)
- Constrain conda version for Windows build (#4040)
- Updates to faiss-gpu-cuvs nightly pkg (#4032)
- pin the dependecies version for x86_64 (#4046)
- pin arm64 dependency (#4060)
- Pin conda build (#4062)
- Improve naming due to codemod (#4063)
- Improve naming due to codemod (#4064)
- Improve naming due to codemod (#4065)
- separare the github build into two conditions (#4066)
- Improve naming due to codemod (#4070)
- improve naming due to codemod (#4067)
- improve naming due to codemod (#4071)
- improve naming due to codemod (#4072)
- fix nightily build (#4080)
- Change github action workflows name (#4083)
- Resolve Packaging Issues (#4044)
- Update __init__.py (#4086)
- Exhaustive IVF probing in scalar quantizer tests (#4075)
- Pin Nightlies with testing on PR (#4088)
- Update benchmarking library code to work for IdMap index as well (#4093)
- Update action.yml (#4100)
- Upgrade CUVS to 24.12 (#4021)
- Link cuVS Docs (#4084)
- Set KnnDescriptor.desc_name in the Benchmarking core framework in FAISS like other descriptors (#4109)
- enable quiet mode for conda install (#4112)
- Disable retry build (#4124)
- Add ngpu default argument to knn_ground_truth (#4123)
- Update code comment to reflect the range of IF from [1, k] (#4139)
- Reenable auto retry workflow (#4140)
- Migration off defaults to conda-forge channel (#4126)
- Benchmarking Scripts for cuVS Index, more docs updates (#4117)

Fixed
- Fix total_rows (#3942)
- Fix INSTALL.md due to failure of conflict resolving (#3915)
- Back out "Add example of how to build, link, and test an external SWIG module" (#3954)
- Fix shadowed variable in faiss/IndexPQ.cpp (#3959)
- Fix shadowed variable in faiss/IndexIVFAdditiveQuantizer.cpp (#3958)
- Fix shadowed variable in faiss/impl/HNSW.cpp (#3961)
- Fix shadowed variable in faiss/impl/simd_result_handlers.h (#3960)
- Fix shadowed variable in faiss/utils/NeuralNet.cpp (#3952)
- Resolve "incorrect-portions-license" errors: add no license lint to top of GPU files with both licenses (#3965)
- Resolve "duplicate-license-header": Find and replace duplicate license headers (#3967)
- fix some more nvidia licenses that get erased (#3977)
- fix merge_flat_ondisk stress run failures (#3999)
- Fix reverse_index_factory formatting of ScalarQuantizers (#4003)
- Fix shadowed variable in faiss/IndexAdditiveQuantizer.cpp (#4011)
- facebook-unused-include-check in fbcode/faiss (#4029)
- fix linter (#4035)
- Some chore fixes (#4010)
- Fix unused variable compilation error (#4041)
- stop dealloc of coarse quantizer when it is deleted (#4045)
- Fix SCD Table test flakiness (#4069)
- Fix IndexIVFFastScan reconstruct_from_offset method (#4095)
- more fast-scan reconstruction (#4128)
- Fix nightly cuVS 11.8.0 failure (#4149)
- Correct capitalization of FAISS to Faiss (#4155)
- Fix cuVS 12.4.0 nightly failure (#4153)

Deprecated
- Remove unused-variable in dumbo/backup/dumbo/service/tests/ChainReplicatorTests.cpp (#4024)
- remove inconsistent oom exception test (#4052)
- Remove unused(and wrong) io macro (#4122)


## [1.9.0] - 2024-10-04
### Added
- Add AVX-512 implementation for the distance and scalar quantizer functions. (#3853)
- Allow k and M suffixes in IVF indexes (#3812)
- add reconstruct support to additive quantizers (#3752)
- introduce options for reducing the overhead for a clustering procedure (#3731)
- Add hnsw search params for bounded queue option (#3748)
- ROCm support (#3462)
- Add sve targets (#2886)
- add get_version() for c_api (#3688)
- QINCo implementation in CPU Faiss (#3608)
- Add search functionality to FlatCodes (#3611)
- add dispatcher for VectorDistance and ResultHandlers (#3627)
- Add SQ8bit signed quantization (#3501)
- Add ABS_INNER_PRODUCT metric (#3524)
- Interop between CAGRA and HNSW (#3252)
- add skip_storage flag to HNSW (#3487)
- QT_bf16 for scalar quantizer for bfloat16 (#3444)
- Implement METRIC.NaNEuclidean (#3414)
- TimeoutCallback C++ and Python (#3417)
- support big-endian machines (#3361)
- Support for Remove ids from IVFPQFastScan index (#3354)
- Implement reconstruct_n for GPU IVFFlat indexes (#3338)
- Support of skip_ids in merge_from_multiple function of OnDiskInvertedLists (#3327)
- Add the ability to clone and read binary indexes to the C API. (#3318)
- AVX512 for PQFastScan (#3276)

### Changed
- faster hnsw CPU index training (#3822)
- Some small improvements. (#3692)
- First attempt at LSH matching with nbits (#3679)
- Set verbosoe before train (#3619)
- Remove duplicate NegativeDistanceComputer instances (#3450)
- interrupt for NNDescent (#3432)
- Get rid of redundant instructions in ScalarQuantizer (#3430)
- PowerPC, improve code generation for function fvec_L2sqr (#3416)
- Unroll loop in lookup_2_lanes (#3364)
- Improve filtering & search parameters propagation (#3304)
- Change index_cpu_to_gpu to throw for indices not implemented on GPU (#3336)
- Throw when attempting to move IndexPQ to GPU (#3328)
- Skip HNSWPQ sdc init with new io flag (#3250)

### Fixed
- FIx a bug for a non-simdlib code of ResidualQuantizer (#3868)
- assign_index should default to null (#3855)
- Fix an incorrectly counted the number of computed distances for HNSW (#3840)
- Add error for overflowing nbits during PQ construction (#3833)
- Fix radius search with HSNW and IP (#3698)
- fix algorithm of spreading vectors over shards (#3374)
- Fix IndexBinary.assign Python method (#3384)
- Few fixes in bench_fw to enable IndexFromCodec (#3383)
- Fix the endianness issue in AIX while running the benchmark. (#3345)
- Fix faiss swig build with version > 4.2.x (#3315)
- Fix problems when using 64-bit integers. (#3322)
- Fix IVFPQFastScan decode function (#3312)
- Handling FaissException in few destructors of ResultHandler.h (#3311)
- Fix HNSW stats (#3309)
- AIX compilation fix for io classes (#3275)


## [1.8.0] - 2024-02-27
### Added
- Added a new conda package faiss-gpu-raft alongside faiss-cpu and faiss-gpu
- Integrated IVF-Flat and IVF-PQ implementations in faiss-gpu-raft from RAFT by Nvidia [thanks Corey Nolet and Tarang Jain]
- Added a context parameter to InvertedLists and InvertedListsIterator
- Added Faiss on Rocksdb demo to showing how inverted lists can be persisted in a key-value store
- Introduced Offline IVF framework powered by Faiss big batch search
- Added SIMD NEON Optimization for QT_FP16 in Scalar Quantizer. [thanks Naveen Tatikonda]
- Generalized ResultHandler and supported range search for HNSW and FastScan
- Introduced avx512 optimization mode and FAISS_OPT_LEVEL env variable [thanks Alexandr Ghuzva]
- Added search parameters for IndexRefine::search() and IndexRefineFlat::search()
- Supported large two-level clustering
- Added support for Python 3.11 and 3.12
- Added support for CUDA 12

### Changed
- Used the benchmark to find Pareto optimal indices. Intentionally limited to IVF(Flat|HNSW),PQ|SQ indices
- Splitted off RQ encoding steps to another file
- Supported better NaN handling
- HNSW speedup + Distance 4 points [thanks Alexandr Ghuzva]

### Fixed
- Fixed DeviceVector reallocations in Faiss GPU
- Used efSearch from params if provided in HNSW search
- Fixed warp synchronous behavior in Faiss GPU CUDA 12


## [1.7.4] - 2023-04-12
### Added
- Added big batch IVF search for conducting efficient search with big batches of queries
- Checkpointing in big batch search support
- Precomputed centroids support
- Support for iterable inverted lists for eg. key value stores
- 64-bit indexing arithmetic support in FAISS GPU
- IndexIVFShards now handle IVF indexes with a common quantizer
- Jaccard distance support
- CodePacker for non-contiguous code layouts
- Approximate evaluation of top-k distances for ResidualQuantizer and IndexBinaryFlat
- Added support for 12-bit PQ / IVFPQ fine quantizer decoders for standalone vector codecs (faiss/cppcontrib)
- Conda packages for osx-arm64 (Apple M1) and linux-aarch64 (ARM64) architectures
- Support for Python 3.10

### Removed
- CUDA 10 is no longer supported in precompiled packages
- Removed Python 3.7 support for precompiled packages
- Removed constraint for using fine quantizer with no greater than 8 bits for IVFPQ, for example, now it is possible to use IVF256,PQ10x12 for a CPU index

### Changed
- Various performance optimizations for PQ / IVFPQ for AVX2 and ARM for training (fused distance+nearest kernel), search (faster kernels for distance_to_code() and scan_list_*()) and vector encoding
- A magnitude faster CPU code for LSQ/PLSQ training and vector encoding (reworked code)
- Performance improvements for Hamming Code computations for AVX2 and ARM (reworked code)
- Improved auto-vectorization support for IP and L2 distance computations (better handling of pragmas)
- Improved ResidualQuantizer vector encoding (pooling memory allocations, avoid r/w to a temporary buffer)

### Fixed
- HSNW bug fixed which improves the recall rate! Special thanks to zh Wang @hhy3 for this.
- Faiss GPU IVF large query batch fix
- Faiss + Torch fixes, re-enable k = 2048
- Fix the number of distance computations to match max_codes parameter
- Fix decoding of large fast_scan blocks


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

[Unreleased]: https://github.com/facebookresearch/faiss/compare/v1.9.0...HEAD
[1.9.0]: https://github.com/facebookresearch/faiss/compare/v1.8.0...v1.9.0
[1.8.0]: https://github.com/facebookresearch/faiss/compare/v1.7.4...v1.8.0
[1.7.4]: https://github.com/facebookresearch/faiss/compare/v1.7.3...v1.7.4
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
