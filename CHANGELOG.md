# Changelog
All notable changes to this project will be documented in this file.

## [Unreleased]

## [1.13.1] - 2025-12-02


Added
- add dataset DINO10B (#4686)
- Implement multi-bit RaBitQ quantization (nb_bits 2-9) (#4679)
- Add copyright header to test_flat_l2_panorama.py (#4688)
- Integrate Panorama into `IndexHNSWFlatPanorama` (#4621)
- Implement `IndexFlatL2Panorama` (#4645)


Changed
- clamping variable used for computing percentile (#4687)
- Remove unused variable
- Revert D85902427 (#4690)
- Optimize ScalarQuantizer (#4652)
- Update comment to clarify useFloat16 in GpuClonerOptions.h (#4682)
- facebook-hte-SharedPtrFromNew in StandardGpuResources.cpp (#4680)


Fixed
- Fix deprecated this capture in faiss/gpu/GpuIcmEncoder.cu +1
- Refactor sharding to not oom (#4678)
- Fix significant GOMP barrier overhead in exhaustive_L2sqr_blas. (#4663)
- Fix typos in tests and contrib directories (#4672)



## [1.13.0] - 2025-11-11


Added
- 2cf82cabf2b2150ca76b9949377b484f109a94d1 Implement PanoramaStats (#4628)
- 859127cd0b5e9f2ee52e095211202b0f5dbb6414 Implement serialization for `IndexIVFFlatPanorama` (#4636)
- 7744239dc59c0cc9c665949f0d533daac40e567a Add getter/setter for balanced_bins in PCAMatrix C API issue #4617 (#4630)
- e6510bd00478d563e380604374f41a27e8697ce6 Add immediate notification when autoclose label is applied (#4624)
- f983e3ab69c46ecb7728395613ace22729bbd4a8 Integrate Panorama into `IndexIVFFlatPanorama` (#4606)
- 3af3e00103079554b1071d2dcea7ccedc9693a44 Implementation of IndexIVFRaBitQFastScan (#4596)
- f58fd4c8ca0fdf6c230c062ed70b93586ae9ec28 Add InvertedListScanner for IVFPQFastScan (#4537)
- 01d394e5837f10c6270c7284c9f5e306c5cd87c4 RaBitQ Fast Scan (#4595)
- 752832cac9c4038f52916a0ed2a839401671ce00 Add missing Thrust includes (#4597)
- 6470b8d9d0f9c0adc71df6d5a1ce64199be85305 Simplify RaBitQ slightly, improving speed and recall (#4550)
- bc3e3a1ef1464f1b90c6aa4ba4f075582e4f539a RaBitQ: SIMD helper simplification, use faster popcount for doc-side sum (#4573)
- b7c88eaff82766c69cd3ab7a3a57fa4d4efb603f Microbenchmarks for rabitq_simd.h (#4572)
- 786e4051fc7940825e8f6f5247bc269f4bc8bd11 RabitQ test coverage for SIMD codepaths (#4571)
- fa5532734e40dbf66b571634ff597fe314ace054 Adding Idx tags in extended APIs (#4532)
- fbbb290a5762e265a501324b3bfeb04c1cb5b006 Set NN Descent Metric From CAGRA Params (#4540)
- 514b44fca8542bafe8640adcbf1cccce1900f74c OSS changes: Enable ROCm to work with Faiss on BUCK (#4485)


Changed
- a9cc039aed084bfb824a84eb7fc749ff0e84370c Upgrade cuVS to 25.10 and build pkg with CUDA=12.6 (#4639)
- 6ca45d23a7c10761d380daa6b4029c347f30e4ef Optimize IVFRaBitQFastScan query factors: n*nlist to n*nprobe (#4643)
- 49df7737f868eeb75df450718fc1155d93c43d35 Update GitHub workflow to use clang-format-21 (#4644)
- a0bd7aac694cd43e3ba0a371cbd7bacf31ded564 clang-format | Format fbsource with clang-format 21.
- 595c8aa8a23815abd3682a2c08b0b359b37eb621 Allow unaligned fast-scan (#4623)
- 4fab13c9c67b5402343ca722c83ff7a65a9a48ba Upgrade Faiss OSS side to numpy2 (#4523)
- 2505168e1870318126aee2ac2cb416b1fe55376f Expose Remaining IVF-PQ params for CAGRA (#4593)
- 1ed2611c032fe2c509b0f781ee8eee3760142e52 Remove unused imports from faiss directory files (#4565)
- e5de66e2c24c319e73df22858b520fcfa269d300 Use Development.Module component in CMake FindPython (#4549)
- 3c7235c6c7040c1c09e39095cc60d794f73b4a35 Update Install Docs with Correct cuVS Version (#4547)
- f361df8385ccbf9c47b85a6f8e45c71f8f656283 Use c++11 only in headers (#4421)
- dd637c98d60f51b96c9d2457ebfa319e3f881a47 Change extended API suffix from `Ex` to `_ex` (#4530)


Fixed
- ff1a2d4f6eaf4216c74e6006bf55791cf135b767 Fix typos in core library (root + impl/) (#4670)
- d94c33065a240e460603352770ec6aa2dc70b8cf chore: fix typos in some files (#4669)
- 5d9f8d484cffd0961c374c1c7253c8c406b889df Fix additional typos found in second review (#4667)
- 451ca8429d39f82ed6a933ac874068fa5f13252c Fix typos in documentation, tutorials, and C API (#4666)
- d0d066e430ec0711121926cadaaeb85fd3cbad1f Fix typos in comments and documentation (#4664)
- 3ffec120fe411495939dc070e8588afb1db1b7b1 Fix typos in comments and documentation (#4662)
- 2af54a46bb0dd3a5ba84388c0830f44067bf6e53 Fix typos in comments and documentation (#4661)
- ec877e794e26cc0778e3665349060e3c224dcb8c Fix typos in comments and documentation (#4660)
- 3de200f5afb0452b934458c1b235860b4f2089ee try to fix nightly (#4657)
- 18f5574a50416ab9b61d4806cc802166663912b7 typo: SIFT -> GIST (#4642)
- 675661b4d363b0eba314d38624324b16b548883f Fix IndexIVFRaBitQFastScan nprobe handling in search_with_parameters (#4629)
- 64b1f3a04cf730c78f43bb071fcc6afa1b48a992 Fix autoclose workflow - add GH_REPO for immediate notification
- e5fee1015a90dd3667c0d1f7d866bbc2c023c475 Fix IVFFlatPanorama bench (#4622)
- 70df32b2e989e0c47c04cfd78be3221c79c8d408 Fix IndexIVFRaBitQFastScan by overriding search_preassigned (#4618)
- 61c1c76929809914240b9407162f54513cd2b7db Fix nightly build on windows due to c++20 initializer (#4612)
- 513eabc91d52a9ebc11545ba0b21f3fd1ca99b32 Replace static constexpr with inline constexpr in header (#4613)
- 97dc014472480ff23dd0bf77c5aa46c073d59687 fix: initializing order of gpu compile options (#4581)
- dbc75506150e923abdaf4d09baf4fac56a20740e Fix ARM64 compilation error in IndexRaBitQFastScan (#4611)
- 484dd97ce07ed56129660cccba1d634f765e73ea Add IndexBinaryIDMap2 support to index binary factory (#4603)
- 041ac84318dac299739e568fafee318317ede6ef Fix memory bloat in IndexBinaryHash search due to argument copying in dispatch_HammingComputer (#4600)
- 2964a374dbe5d9359cb0f794f73cc334d95c3ac4 Fix nightly by updating mkl version (#4604)
- 7f7b518df5f1a21fd4591ed68222c26532580467 Unable to import faiss in python in AIX (#4602)
- 266b71285aabfe3ca66715b7878b1d8fd1e472f7 Added code to catch 'ModuleNotFoundError' exception (#4577)
- 3b14dad6d9ac48a0764e5ba01a45bca1d7d738ee Fix AIX compilation issue while building python extension (#4587)
- 1deba7b90f21d952c86affe79721a06ec5800907 Add attribute validation to prevent silent failures in SWIG wrappers (#4583)
- 3671c61af1455d50c169e5ddd5d6a18d81174c3a more formatting (#4568)
- d98ff432b98a4d7bea770c0cc221d8a0dab2e4a9 Fix FAISS build with ROCm7 (#4567)
- 50b3eb48d26b32a2beedb570d1fcf7757830bfcb add "override" to overriden destructors (#4566)
- 2135e5a28d0f2a932dd02b64b566ac1a52266540 add missing explicit specifier (#4564)
- 0031d61da81dab33579f7481f6773549ca816fa8 Revert D80734790 (#4563)
- ee6b7ddd9df954492b597b49b63e8760923317b7 add #pragma once to some header files (#4562)
- 8b83ebbf6271b713264671fcd128c33def70e8d3 fix python blank space lints (#4560)
- f46ac530b534f98b8cd5b78373e9ccabad0fdee4 Fix missing object reference in Faiss python wrapper for IndexIVFRaBitQ (#4554)
- 69f1ac0f0fa635812fdba5180074981f5d7ba2de resolve Open Source requirement violations (#4551)
- dca887a656e943352bcf1cae6dbaaf6771ac272e Remove unused include from IVFlib.cpp (#4534)
- 3b3bf5e4e53eb0846fd1109432773a25a879b01b bugfix: add a macro guard for avx512 operation (#4539)
- fe5e77f1cb0bd26a5a86d9fc1e99f0b2ecacab93 Remove unnecessary std::move on temporaries to fix `-Wpessimizing-move` warnings in DeviceTensor-inl.cuh (#4545)
- 2fca92f477063a1188a1cf52d3414b3eba05936b fix test_binary_cagra tests failure (#4546)
- 8d9d3bea7c2d9879e6ce4919d1a25954ef533c8f Remove unused includes from AutoTune.cpp (#4533)
- 75be84d086190cfda8a516a5f276b5ba0e8cc706 Remove unused imports from fbcode/faiss (#4543)
- 5c61ed8c94296d0ca97cecd53cbcef91f63e2de4 Fix E302 lint errors: Add required blank lines before class definitions (#4541)
- daceaac99aeb9dfc8577a0dc6f47c3f6f9b39055 replace type(x) == y with isinstance(x, y) (#4542)


Deprecated
- 2705d7a5d30f121ffc7553d2d7087fce1a448d6b Delete rocm runner for now until it is fixed (#4658)
- f9ccd582f9a9b8400428625d1ff3217ae83422b9 Remove invalid assertion checking #neighbors == graph degree (#4528)



## [1.12.0] - 2025-08-11


Added
- Adding `Ex` suffix for extended API (#4512)
- SIMD optimization RaBitQ (#4515)
- Binary CAGRA with NN Descent (#4445)
- BinaryHNSWCagra Struct; Allow base_level_only (#4478)
- Added libgflags-dev to Dockerfile (#4460)
- try to add nightly conda publish for Faiss classic GPU with CUDA 12.4 (#4442)
- Add support for IndexIDMap with Cagra fp16 (#4411)
- Faiss + Cuvs Example Notebook (#4434)
- Common ancestor to top-k result handlers (#4414)
- Add rabitq to reverse factory string so telemetry wrapper will log it (#4428)
- Add option to link cuda statically (#4422)
- support extra metrics in IVF (#4409)
- Support decode_vectors (sa_decode but no decode_listnos) in multiple IVF index (#4400)
- Add workaround to check SVE support when numpy.distutils is not available (#4416)
- Dsweet/gowers distance (#4371)
- cuVS Cagra FP16 support (#4384)
- Add cuVS filter conversion utility (#4378)
- GpuIndexBinaryCagra - Binary CAGRA index (#4331)
- Add guarantee_connectivity parameter to GpuIndexCagra (#4388)
- Add Virtual Destructor to FlatIndex Class (#4381)
- Adding unit tests for supporting pre-built KNN while using `IndexNSG` (#4368)
- Pass in "own_invlists" to ivf index constructor (#4353)
- Add new centroid_id_column to support previous_assignment_table (#4343)
- Copy IVF Centroids to Host for CPU Quantizer (#4336)
- Expose IndexBinaryIVF to C API (#4302)
- Add rabitq bench to source control (#4307)

Changed
- Remove unused cmath include from IndexPQ.cpp (#4518)
- Allow Odd Degree Binary CAGRA graph to HNSW conversion (#4516)
- Remove unused cassert and cmath headers from Index2Layer.cpp (#4514)
- Remove unused headers from AutoTune.cpp and clone_index.cpp (#4513)
- Remove unused import header from IndexBinaryHNSW.cpp (#4507)
- Remove unused import header from IndexBinaryHash.cpp (#4506)
- Remove unused import header from IndexIDMap.cpp (#4505)
- Remove unused import header from AutoTune.cpp (#4504)
- Make GpuIndexCagra reuse existing memory block when converting to CPU index (#4477)
- Remove unused standard library includes from index_io.h (#4495)
- Remove unused standard library includes from Index.h (#4494)
- Remove unused standard library includes from IndexBinary.h (#4493)
- Remove unused platform_macros.h include from MetricType.h (#4490)
- fbcode//faiss:faiss (#4462)
- fbcode//faiss/gpu:faiss (#4463)
- fbcode//faiss/gpu/test:test_utils (#4464)
- BW compatibility for read & write with `numeric_type_` in cagra (#4441)
- Upgrade cuVS version to 25.08 (#4394)
- Improve naming of the residual_quantizer_encode_steps.cpp file (#4433)
- Github actions: upgrade windows-2019 to windows-2022 (#4417)
- Update the dependency for submitit in faiss_bench_fw (#4410)
- Increase top-k limit on GPU for cuVS (#4325)
- Let IndexShards Pass down SearchParams (#4387)
- Change uint8_t* => const uint8_t* in faiss::ZeroCopyIOReader (#4376)
- Set code_size for more InvertedListScanner implementations (#4365)
- Improving variable name post codemod changes (#4369)
- Improve naming due to codemod (#4367)
- Use packaging.version for correct version parsing (#4330)
- Update Swig File for CAGRA Params (#4314)

Fixed
- Fix the warning that numpy.core._multiarray_umath is deprecated and has been renamed to numpy._core._multiarray_umath (#4501)
- Fix inline code syntax (#4437)
- Bug fix for faiss_Index_sa_code_size (#4492)
- Fix CQS signal modernize-use-using in fbcode/faiss/tests [B] [B] (#4483)
- Fix CQS signal modernize-use-using in fbcode/faiss/tests [B] [A] (#4484)
- Fix CQS signal readability-braces-around-statements in fbcode/faiss/utils [B] [B] (#4479)
- Fix CQS signal readability-braces-around-statements in fbcode/faiss/utils [B] [A] (#4480)
- Fix CQS signal modernize-use-using in fbcode/faiss/tests [A] (#4476)
- Fix CQS signal readability-braces-around-statements in fbcode/faiss/utils [A] (#4475)
- Fix ALL nightlies (#4471)
- Fix CQS signal readability-braces-around-statements in fbcode/faiss/invlists (#4467)
- Fix CQS signal readability-braces-around-statements in fbcode/faiss/perf_tests (#4458)
- Fix CQS signal readability-braces-around-statements in fbcode/faiss/tests (#4457)
- `int8` support for cuVS cagra (#4439)
- Fix indexes after fp16 change (#4452)
- Fix CQS signal readability-braces-around-statements in fbcode/faiss (#4454)
- Fix CQS signal readability-braces-around-statements in fbcode/faiss/gpu (#4453)
- Fix CQS signal facebook-unused-include-check in fbcode/faiss (#4451)
- Fix CQS signal readability-braces-around-statements in fbcode/faiss/benchs (#4449)
- Fix CQS signal readability-braces-around-statements in fbcode/faiss/impl (#4447)
- Revert D78330300 (#4448)
- Fix CQS signal readability-braces-around-statements in fbcode/faiss/tutorial (#4446)
- Fix CQS signal facebook-unused-include-check in fbcode/faiss/utils (#4435)
- Fix CQS signal facebook-unused-include-check in fbcode/faiss/perf_tests (#4443)
- Fix CQS signal facebook-unused-include-check in fbcode/faiss/impl (#4436)
- Remove Debug Statement (#4425)
- Fix invalid long_description in setup.py (#4398)
- Fix building on mingw (#4420)
- fix: add avx2 in python if avx512 enabled (#4419)
- IndexFlat: Reconstruct validate if key < ntotal (#4415)
- cmake: disable installing external documents under docs/faiss (#4406)
- Pin openblas 0.3.30 to fix nightly breakage (#4404)
- Add override to IndexBinaryCagra (#4401)
- Fix CQS signal facebook-unused-include-check in fbcode/faiss/gpu (#4390)
- Fix CQS signal facebook-unused-include-check in fbcode/faiss/impl (#4395)
- Fix unreachable-break issue in faiss/IndexAdditiveQuantizer.cpp +1 (#4391)
- Fix CQS signal facebook-unused-include-check in fbcode/faiss/utils (#4389)
- fix IndexIVFFastScan ndis/nlist stat (#4383)
- fix: update broken links (#4382)
- Fix incorrect constructor docstring for IndexScalarQuantizer (#4350)
- Fix CQS signal facebook-unused-include-check in fbcode/faiss/tests (#4375)
- Fix input param for IndexIVFScalarQuantizer in index_factory (#4358)
- Fix openMP index bug (#4348)
- Fix CQS signal. Id] 95408353 -- performance-faster-string-find in fbcode/faiss (#4345)
- cmake: install missing header `impl/zerocopy_io.h` (#4328)
- Fix CQS signal. Id] 62183176 -- readability-redundant-string-init in fbcode/faiss/impl (#4332)
- Fix CQS signal. Id] 57328835 -- performance-unnecessary-value-param in fbcode/faiss/impl (#4329)
- Fix IndexBinaryIVF::merge_from (#4305)
- FreeBSD compatibility patch (#4316)

Deprecated
- Deprecate CUDA 11 from Faiss nightly and releases (#4496)
- Deprecate CUDA 11 nightly, add 12.4 to build-release, fix cuvs nightly (#4482)
- Remove cuVS CUDA 11.8 CI (#4444)
- remove ABS_INNER_PRODUCT metric (#4408)
- Disable failing test until we support CUDA 12.8 in CI (#4392)
- Disable flaky ivfflat test_mem_leak



## [1.11.0] - 2025-04-24


Added
- RaBitQ implementation (#4235)
- Add RaBitQ to the swigfaiss so we can access its properties correctly in python (#4304)
- Add date and time to the codec file path so that the file doesn't get overridden with each run (#4303)
- Add missing header in faiss/CMakeLists.txt (#4285)
- Implement is_spherical and normalize_L2 booleans as part of the training APIs (#4279)
- Add normalize_l2 boolean to distributed training API
- re-land mmap diff (#4250)
- SearchParameters support for IndexBinaryFlat (#4055)
- Support non-partition col and map in the embedding reader (#4229)
- Support cosine distance for training vectors (#4227)
- Add missing #include in code_distance-sve.h (#4219)
- Add the support for IndexIDMap with Cagra index (#4188)
- Add bounds checking to hnsw nb_neighbors (#4185)
- Add sharding convenience function for IVF indexes (#4150)
- Added support for building for MinGW, in addition to MSVC (#4145)

Changed
- Skip mmap test case in AIX. (#4275)
- Handle insufficient driver gracefully (#4271)
- relax input params for IndexIVFRaBitQ::get_InvertedListScanner() (#4270)
- Allow using custom index readers and writers (#4180)
- Upgrade to libcuvs=25.04 (#4164)
- ignore regex (#4264)
- Publish the C API to Conda (#4186)
- Pass row filters to Hive Reader to filter rows (#4256)
- Back out "test merge with internal repo" (#4244)
- test merge with internal repo (#4242)
- Revert D69972250: Memory-mapping and Zero-copy deserializers
- Revert D69984379: mem mapping and zero-copy python fixes
- mem mapping and zero-copy python fixes (#4212)
- Memory-mapping and Zero-copy deserializers (#4199)
- Use `nullptr` in faiss/gpu/StandardGpuResources.cpp (#4232)
- Make static method in header inline (#4214)
- Upgrade openblas to 0.3.29 for ARM architectures (#4203)
- Pass `store_dataset` argument along to cuVS CAGRA (#4173)
- Handle plain SearchParameters in HNSW searches (#4167)
- Update INSTALL.md to remove some raft references, add missing dependency (#4176)
- Update README.md (#4169)
- Update CAGRA docs (#4152)
- Expose IDSelectorBitmap in the C_API (#4158)

Fixed
- fix: algorithm of spreading vectors over shards (#4299)
- Fix overflow of int32 in IndexNSG (#4297)
- Fix Type Error in Conditional Logic (#4294)
- faiss/gpu/GpuAutoTune.cpp: fix llvm-19-exposed -Wunused-but-set-variable warnings
- Fix nightly by pinning conda-build to prevent regression in 25.3.2 (#4287)
- Fix CQS signal. Id] 88153895 -- readability-redundant-string-init in fbcode/faiss (#4283)
- Fix a placeholder for 'unimplemented' in mapped_io.cpp (#4268)
- fix bug: IVFPQ of raft/cuvs does not require redundant check (#4241)
- fix a serialization problem in RaBitQ (#4261)
- Grammar fix in FlatIndexHNSW (#4253)
- Fix CUDA kernel index data type in faiss/gpu/impl/DistanceUtils.cuh +10 (#4246)
- fix `IVFPQFastScan::RangeSearch()` on the `ARM` architecture (#4247)
- fix integer overflow issue when calculating imbalance_factor (#4245)
- Fix bug with metric_arg in IndexHNSW (#4239)
- Address compile errors and warnings (#4238)
- faiss: fix non-templated hammings function (#4195)
- Fix LLVM-19 compilation issue in faiss/AutoTune.cpp (#4220)
- Fix cloning and reverse index factory for NSG indices (#4151)
- Remove python_abi to fix nightly (#4217)
- Fix IVF quantizer centroid sharding so IDs are generated (#4197)
- Pin lief to fix nightly (#4211)
- Fix Sapphire Rapids never loading in Python bindings (#4209)
- Attempt to nightly fix (#4204)
- Fix nightly by installing earlier version of lief (#4198)
- Check for not completed
- Fix install error when building avx512_spr variant (#4170)
- fix: gpu tests link failure with static lib (#4137)
- Fix the order of parameters in bench_scalar_quantizer_distance. (#4159)

Deprecated
- Remove unused exception parameter from faiss/impl/ResultHandler.h (#4243)
- Remove unused variable (#4205)



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

[Unreleased]: https://github.com/facebookresearch/faiss/compare/v1.11.0...HEAD
[1.11.0]: https://github.com/facebookresearch/faiss/compare/v1.10.0...v1.11.0
[1.10.0]: https://github.com/facebookresearch/faiss/compare/v1.9.0...v1.10.0
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
