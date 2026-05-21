# Changelog
All notable changes to this project will be documented in this file.

## [Unreleased]

## [1.14.2] - 2026-05-21

Added
- b7618fadc7fdf9b677277fbe96864c1902a6d8dc Add SuperKMeans: faster k-means via ADSampling+PDX progressive pruning (#5168)
- 66c9d082651659fb6e78d67c547b035d3c2c8228 Add SuperKMeans foundations: math primitives, PDX layout, SIMD kernels
- dc7afcfdf0d4bf7cc694f4974db77e471422d4cb Support SVSVamana as IVF coarse quantizer via index_factory (#5175)
- 7a8e4ddaca3d25363a79942e9fcd26de78436607 CI: cross-compile for riscv64 with RVV dynamic dispatch (#5184)
- 03202795b6451f2c6942262c998de3b980e96157 Introduce RVV (#5156)
- 417c53e0ce23e8c3f2843bc727c9493e35b1d5d0 Add NoneSIMDLevel context manager for cross-level reference checks (#5158)
- 46def46902231ed57552ddbcfd751e17720c6d32 Enable SVS IVF Index in FAISS (#4801)
- 585ba7954afb4da180a897bf195acd7516a44632 Add pip install support via scikit-build-core + cibuildwheel (#4862)
- 799bf3aaeca910113fd2f1b0287568f07eb002fa Introduce early stop facilities for IVF, attempt 2 (#5160)
- 66cea524339ee7cd1b3954e3b5a677edb2adad74 Add Metal GPU backend for Apple Silicon (IndexFlat) (#5144)
- 17fd3332c77c2801da4b467de67b5cadf045b328 Add database-parallel flat search for few-query workloads (#5000)
- c7f07bbc0ad0e9c0b68c6c7a3696850f42500906 Add C++ per-SIMD-level tests for distance utility functions (#5154)
- 0242758a3d073be0f5b0117b524a1fbdfcf02860 Add bit-exact RQ encode cross-level test (#5155)
- 2f68a3730777f88da34d562bc90f637bb39c1cb3 Add per-SIMD-level tests for hamming.h utils + fix crosshamming bug (#5153)
- a7116ed4ec2ab4f0889d305592f579d5b2fb5745 Add TurboQuant (CPU) (#5049)
- 11e5d3916054b23bf60253763abf5e9033fca39e Add optional persistent locks to IndexHNSW for incremental adds (#5031)
- e82aa28b838d1cad66044c3888b75eb832a39fb4 Add OMP exception capture helpers and migrate IndexIVF (#5111)
- 6299bff1f60e5423fa8cfe8e8d3926f8d87bdafb Add QT_0bit to ScalarQuantizer for centroid-only IVF distance (#5079)
- 3030fe0635b743ccfddf3684752438269a5ab0c5 Add filtered search for cuVS indexes (#4858)
- f83ea51eaa60987315fcd2deca153924abd7995d Support IVFRaBitQSearchParameters in RaBitQFastScan scanner (#5081)
- 82a823513d8a7a7e497de11e7da0cba2aae7721d Add iterator callbacks for distance computation and heap admission (#5082)
- b068fd985a7460c04ec6c4f909900409392520d0 Support IO_FLAG_MMAP_IFC on Darwin (#5058)
- 113bf6ceac4e9d1bfeaf29279b29a8293759f83b Expose fast_scan_code_size from non-IVF FastScan indices (#5077)
- 786d993dbfb8b74b2a0d4d9543b2b08310acabbe Expose a fast scan code size from fast scan indices (#5064)
- 95f1e45221ede8caf1b0139a3c8864d473e0e47d OMP build diagnostics (#4842)
- c4ac852e5b5aa2ba4bde8a977633c367aa85d5db Add a balanced assignment function (#5050)
- 8402a6b764c024ca78fb87d768cb81de3f60c925 Add unit tests for HNSW deserialization validation (#5033)
- 1dc5e1fb60405eda268a7d55b1d5a14feef962ba V2 enable specifying a subset of SIMD levels that are implemented at dispatching time. (#4959)
- e6f5c0c9a9f70f63c00d3ac5651edce6aebd734d Add SIMD dynamic dispatch migration guide (#4973)
- 8177a3bca62af1042d955277cafe9db73d5a0c1d Add nvidia-smi diagnostic call (#4960)
- 67fc664fe0ac1774bd4dcf2ab1aaa488cc7d0545 Add CappedInvertedLists for per-list size limiting (#4880)
- e57a8939c91244009c7ef5745d59f33139dc46d2 introduce `SingleQueryResultCollectHandler` (#4926)
- 748c0311e423844f25d46a95d27d1621c520a4c7 HNSW: add prune_headroom to avoid O(n^2) pruning/locking, headroom test (#4847)
- 1e4d2271a6b27c70c4d824cc47067d36998d9daf Introduce an early stop threshold for Kmeans (#4894)
- 4a0fba1fe2c5e5c8fb3405e614ef9b11524b1064 Add FastScanCodeScanner dispatch boundary with per-SIMD TUs (#4897)
- 0fcba2284c266f12ead208ef61559c81e2d7ac72 Add sentinels to index format enums. (#4907)
- 40480fb303fe06a7126e78c71965041911b687c5 Support limits on index deserialization loops — useful for tests (#4902)
- 1e4544a37e62f89bce0baf3253d214d37fba5192 Add defaulted SIMDLevel template parameter to handler and scaler types (#4867)
- abdd37bd964b87be5283334ca49ff603a8e46f4c Add Pixi installation option to docs (#5214)

Changed
- 9d5491a117b1628dd911d942353156db63553fcf Templatize SIMDResultHandler on SL, mark handle() final (#5223)
- b63f236a5252580e1f24158ca7bc6f58517983f2 Upgrade GitHub Actions CI from CUDA 12.6 to CUDA 13.2 (#5207)
- a9f5baa58c3bacf208d9f02362ea9db47ffb9f21 Move RISC-V fast_scan forwarders to dedicated impl-riscv.cpp (#5216)
- 8932716646db11b5f03f1d6e64d2b96ad6cd202b Validate SVS storage_kind via shared helper at all deserialization read sites (#5204)
- bc490b56b0896a56d8fc68df42b00798ab5d7eee Make IVFPQSearchCagraConfig dtype fields settable from Python (#5191)
- ffd3727d4815a819b3c229adc1ec4fb0e584156c Update pattern for index factory string with SVSVamana (#5201)
- f37041d0732bed52e186263466e0cc02fd2b6fe2 Extract Metal shaders to standalone .metal file with kernel wrapper class (#5167)
- d98364eb490f0f53b9bece0046b2933f1ec7e646 perf: reserve() to avoid rehashing/reallocation (#5193)
- 4eee3d7859d4872dc30172092e3ff2a7b919df99 Add tolerance and assertion log to test_OIVFPQ (#5188)
- 2322afd2d8489bcaae4cb50d105786121335b1a7 faiss: parallelize post-BLAS reduction loop and end_multiple() in result handlers (#5185)
- 967eda65dec2515ccd18c5abe464b485f8a883d0 Replace FAISS_ASSERT with FAISS_THROW_FMT for unsupported SVSStorageKind (#5182)
- 715725df362a04d7f1424942583c309ee79c14c6 Enable verbose mode for TestPyPI publish step (#5181)
- 5e123115f109ae17062c0a5b120a21e7723bbc43 Filter wheel artifacts before publishing to PyPI (#5179)
- 28b2b66cf279ff2bdb7b8a7fd387c7ea328cb642 Inline PQ code distance kernels into scanner TUs (#5159)
- c41405d5253cfd45673728f5544acad07de1eb32 Enforce minimum training set size for ScalarQuantizer, check nullptr (#5141)
- 44c1972aec4e5d19720263c64dcc5592b06edbdc Enable ccache in GitHub Actions cmake builds (#5157)
- e09de2d070ea13ad4fdc65b5c1df97d5455732de Harden compute_query_factors: add preconditions, handle aliasing, remove dead checks
- 6a9936c60de22ef41138e8b6a8f19df5f180c4ea Hoist per-iteration vector allocations in IndexIVFRaBitQFastScan::compute_LUT
- c3460c980be78afac11f6dc1f88b170fd7e70693 Convert binary index Hamming callers to dynamic dispatch (#5071)
- ecac6511a271e5441d942054c619dbd34d71e83f Convert hamming.cpp public API to dynamic dispatch (#5070)
- 8d1b964c11a943f72c6d6f8d7e4e250fd15b5697 Enable dynamic dispatch build on Windows (#5127)
- a25bcb1e22e06d4626941bcc0d7b1365bd8b428b Rely on cuVS for default values (#5140)
- 06228410c6a60c4ca127b00a16e4007d5f0d5353 Panorama Optimizations (#5041)
- f4338ade1bfae983ff03164f8ef53921b193f46c Parameterize existing tests with @for_all_simd_levels (#5134)
- 9d567497ecae53cebafa7c81dc4a75b7d1aff8cd Wire 512-bit QBS kernels into fast scan DD dispatch (#5075)
- bdf877f90c955581d74d559a5b85719cee78ba0f facebook-unused-include-check in IndexNNDescent.cpp (#5132)
- 632427fd0d9d11990f9520d09794203151308253 fbcode/faiss/invlists (#5048)
- e23c66175d7d5469aee00c9939ccb1ecf7bb4ac5 Replace dispatch_HammingComputer with with_HammingComputer (#5126)
- ed7e1f2435b6c7d76d049473758ef50001547812 Hoist SIMD dispatch outside loops in 4 call sites (#5074)
- 582246b6bd7ad71bf60d6ddf5b4b1066de24ffd3 Remove use of __AMDGCN_WAVEFRONT_SIZE macro with it being deprecated in ROCm 7 (#4619)
- 2dff119f138167b915d03fa8df23068802f38e51 Add braces around single-line control flow statements
- cc0c08754851a6d95870f259c91271fd4560416d Hoist SIMD dispatch out of approx_topk hot loops (#5073)
- 18f93fd82f1de86245b4566c53553316671112e5 Replace SSE3 intrinsics in Index2Layer with portable scalar code (#5076)
- 42765abc0061a00e0afc5f32e53c5d798a3ded61 Optimize RaBitQ FastScan LUT construction for high-dimensional data (#5110)
- ec5e70c6555c2f95f131d33c731f72f13065d3af Optimize multibit sign-bit unpacking in RaBitQ FastScan handlers (#5097)
- 3f127ee3bad4922b1ba0cf2f17f1368d99a241ec Thread qb/centered search params through FastScan LUT and handler (#5095)
- 3c19c52ca0380bbbd032ba15cd9ff8143e4b7b8e Use std:: qualified math functions to avoid float-to-double promotion (#5088)
- 48d2b0c946de235f6f5cc2a6fb072f90544a2aee Replace typedef with using declarations (#5086)
- e12debcb6123eee4dd49eafd143ac20952ad75b1 Remove extra semicolons after inline member function definitions (#5085)
- 5679d3ad42533e6f367fdd746b60e3b16fd692c7 HNSW Cagra base-level entrypoint sampling parallel (#5068)
- c4efc422192b6e43a7f5b2014f5cdd53e8bd8e84 Remove bare simdlib type aliases from simdlib_dispatch.h (#5069)
- 4a63c9e415078a79a7e95d52eb8c974d3ddefdaa Convert partitioning to dynamic dispatch (#5062)
- f226be432c261df5f24c163434e3b45d3abf4193 Roll out @for_all_simd_levels to key test files (#5061)
- afd69493777f8a95c3b2c810f35a24aef8105c8c fbcode/faiss/impl/HNSW.cpp (#5059)
- 658c4340c0a6ff548c291d26d7182e09e703344a Replace bare assert() with FAISS_THROW_IF_NOT in core index files (#5052)
- 364749e88624ae2c93c099e1a798c48a47c1ab2d Re-enable ROCm runner for AMD GPU CI (#4854)
- faeac8636c4c6c15c47fa60fc7f2c18db775b3b7 Update outdated C++17 and Python 2 references (#5045)
- 77e602870bd860043db843ccbf458c2ef1c534b2 Use HTTPS for GitHub links in README (#5043)
- 91843bdbd3eaf1b4e409b1d61763cb2e35170a12 Use explicit ISA flags instead of -march/-mtune=sapphirerapids (#5034)
- acac823db4cfff3fdd4e6a033543e0649a9f9f27 Merge .gitignore updates from faiss-gpu-cu132 into main (#4996)
- 8d8881ee3f6a5a9f8de827742ca38f9db89b78ec Make distance_compute_blas_threshold dimension-aware (#5022)
- 553c78b25685d0aa38a2844ab35fdd9ed6801e57 HNSW SIMD dispatch (#5015)
- 499b4885b8aaf439013a820b6f0f14e57ad4ff1a close the loop on ivf rabitq fast scan filter integration (#5026)
- aef066ac0ebc914a7de2d6ab1770df15a1b10c69 SVS 0.3.0 (#4999)
- cc34603106c16d30edbdd9cb1fcd35ce8f2fbc5a reuse unpack buffer in multibit IVFRaBitQFastScan refinement (#5003)
- d81584d548191ff002c304f5d5c8e3f519e6e4e2 Update Python type stubs: fix class hierarchy and add missing base classes (#5004)
- d1c432fb5fd7d1bf96a73299e95fa6476b2825e3 Convert rabitq_simd.h to runtime SIMD dispatch (#4912)
- 6bca9613b7822ee7392957bf19cc53c237133bd4 Convert distances_fused/ to runtime SIMD dispatch (#4911)
- 04d0d56d0e24ad8da8283a210f8f0527ae2ac91c facebook-unused-include-check in distances_simd.cpp (#5013)
- 7eb05f01fab7bce7750f84f4ce63fb17354cdd43 facebook-unused-include-check in partition.cpp (#5012)
- 7e04fa26212bbfba8749f837fc5e6be1360613b4 Update Python type stubs (__init__.pyi) to match current API (#4998)
- db0e798d5a93a97589d7705ae44f0f7b222bea90 Use omp_get_max_threads() for OpenMP thread count (#4991)
- 9af8384f14edb7c47c8ac6869c6dd45884f72369 Bump openblas from 0.3.30 to 0.3.32 in conda recipes (#4994)
- c048917dcd686fcdc7dd386c6d2b345767085bac Enable quieting FAISS warning "inverted lists not stored" (#4964)
- b2ddb465bf8fe6a3cf470a67d3f8d70df8cc86c9 Loosen SVS findpackage version requirement (#4969)
- d8c3d26130552d70b32c7d61acb9a315f66b045a Upgrade to C++20 for CUVS (#4881)
- 94f2b51fdbf9d72956e84131a12dc6d911e7c362 Rename train() with queries to train_with_queries (#4955)
- 02d609caff186bbbf04a2b541ffa1fd7a7a967fc Upgrade cuVS Version to 26.02 (#4945)
- 55a47c927cd93f22c2b60953bc9e1138e31f1ec4 Upgrade cuVS Version to 26.02, switch various BUCK files over to 26.02 (#4788)
- 0d147a78bc60574d6ae03d09e9e04c9b32d5b6c5 Make READVECTOR byte limit configurable (#4928)
- 9962fbec6ee8cd0ea1d8a05c9e8d435652d6a096 HNSW: narrow critical section during add to avoid lock contention (#4915)
- b1feeb79f88ea11e2c7cbebb2a9fed7e9aa67bc2 Convert approx_topk and residual_quantizer_encode_steps to SIMDLevel dynamic dispatch (#4923)
- 58a57e6bca746821b8a018a35bd6d7057565de03 Re-enable backwards compatibility tests for rabitq now that 1.14.1 released (#4924)
- 5dcef1858d960b22226a47d1b9ee326033e73b63 Refactor binary HNSW stats to use OpenMP reduction instead of destructor sync (#4910)
- d8c1a9785f21c84e7517226f6d7f84bfbfd26a36 move distances implementation into SIMD specific compile units (#4906)
- ea11f0ef9c04d346a65e9096e13d6d5cb6ba8aab Switch all search paths to FastScanCodeScanner and remove make_knn_handler (#4904)
- 57190f91cddffc8ac168dd5ac88044976d4be864 Wire RaBitQ search through FastScanCodeScanner dispatch (#4903)
- 79e8acb07228e2b706ce784358a9bfc42672761f Parallelize compute_residuals in IndexIVFPQ (#4654)
- eddea2a6078c362f19deb39fd16ccff875121a9e Extract RaBitQ result handler to impl/fast_scan/ (#4895)
- 734751a16f6babc42a850b1c78b7cdb09bc9ddbb Extract PQ4 kernels to includable headers in impl/pq_4bit/ (#4868)
- bb298a2323c1df5fd4ee4cdf8ddae23d6614e45c convert simdlib in distances_simd.cpp (#4884)
- 8d8268c8aeea11cb0f25f37789882d4800ac89f1 Templatize simdlib types on SIMDLevel (#4866)
- c74809a7c6f0830d1b6090494979533252465269 Improve exception safety for the 'own_fields'/allocation pattern (#4864)
- c6cf004ecefebfc71822b72be617ef96839ae004 use `level` but not `0` in neighbor_range(i, level, &begin, &end) (line 198) (#5005)

Fixed
- cb69d7c5665241d1e84f476105ffb6fd6c7c192c Fix Dq=None crash and np.empty nondeterminism in search_preassigned wrappers (#5221)
- 5c92c5c1ee744545a77f6be9f14d4fdccaad7869 Fix flaky test_hnsw smoke test by increasing efSearch (#5222)
- 8d3cc92536e340e210f05c8265b8523e1efd1168 Fix clone_index null return for IndexRowwiseMinMax (#5220)
- ff1d54397780abc22dd90e958bef11c77ea88a1b Fix python_unnecessary_generator_set_comprehension issues in faiss/tests/test_fastscan_filter.py (#5215)
- e2359156e08c96bdc9a6ddb2ed28e97963f8f3c4 Fix FastScan DD regression by threading SIMD level through kernel functions (#5210)
- 172324aea695fca9f5d59c7a6a7098f28e314e52 Fix IDSelectorBitmap conversion to cuVS bitset (#5211)
- 6bd749e39bd17d45acfdb6f92d5027fc64ffeced Fix IDSelector leak via SearchParameters.sel setter (#5208)
- 6376bc35e3db976d55a8fcec039dd1ffb444aca9 Fix CI: (cuVS) conda shards cache lock, (ARM) bump openblas, (SVS) fix LeanVec double-destruction (#5209)
- 4b5a73577ee90c8466139f5448ef2b5db033808b Fix python_unnecessary_generator_set_comprehension issues in faiss/.../bench_fw/optimize.py (#5205)
- 6789bf5eb7ff1ffac95ca24b4473af8ba8024ac5 Validate IndexIDMap id_map and inner-index ntotal consistency to prevent search-time null deref (#5203)
- f7508155124af8de6ca427b9842feb770d0075f2 Pin MKL to <2026 to avoid soname mismatch with pytorch (#5192)
- bb2ce7137d32901898b81fb519abee00c2ce841e Fix 7 broken tests (#5197)
- f323c0fa11a3e17b4417d81fb75159304b5a550f Fix backward compat test (#5195)
- 23cd94cd51062f014cb48a1b1a00884311ad786e Reduce BinaryCagra test parameters to prevent CI timeouts (#5194)
- 85fc627c1e7559557e1d0c427eb506e124944524 Validate ProductQuantizer M*ksub during deserialization to prevent oversized allocations (#5187)
- 6cef1bb08c47ff7fc772e8d70a931b014eab771c Fix cuVS and ROCm CI conda environment failures (#5180)
- 4a471eebf9e41059ad4f677172bdda6a0af504a2 Pin conda <25.7 to fix Windows build (#5176)
- a9c0d4187b508565599677e4421143a676d3625c Work around GCC 12 miscompilation of AVX2 histogram (#5124)
- 40a8cc003460855a810b2063191eaac0382bb85e Fix lints associated with early stop facilities for IVF (#5172)
- 1e69d5ab20223212e23df9630a5240b60d075f04 Fix peak-memory spike when loading IVF invlists via IO_FLAG_MMAP_IFC (#5122)
- 9942229393c6a161cff4061634805d2e5c24adf6 Fix add_sa_codes silently accepting non-int64 ids, corrupting stored labels (#5171)
- c5ddbc710427d6b323f0ee0bdfa406f77c42d63e c_api: fix IndexShards own_indices getter/setter name mismatch (#5165)
- 8380e2543db6ae43feba7fd210207bd916eca953 Fix avx512 unit test (#5161)
- 71448c07d309a5058703a81e1dda6059c4a9f947 Fix issue with svs tests (#5162)
- 6c70444d0843d6de7e1e8376c0cc4526cae1025a Validate code_size during deserialization to prevent oversized allocations (#5151)
- 9dbb81ca9d674b031090c4ac31cf001a65fdd78f Set own_fields after reading Index2Layer sub-quantizer during deserialization (#5147)
- 01b22b4cc94fe4121de30f04b54a1f451a1e6188 Fix memory leak warning for unordered_multimap in IndexIVFFlatDedup Python binding (#1667) (#5145)
- 130fc2481a0fa768848387cd3f724395157cebe1 Fix flaky test (#5150)
- dfce6e948e153b783d8bd0785c1c30432e84a166 Fix OMP exception safety in IndexHNSW search (#5133)
- c6273349f948dc2df83be83c8f49c12ca63842e8 Fix sa_decode offset bug and integer overflow in IndexRefine (#5143)
- 1b4b995b97c97e503c051e7b3087b2937aed5633 Fix implicit integer precision loss from 64-bit to 32-bit (#5091)
- b0568942efcb980f91fb286e1fdebfa0925f1150 Fix miscellaneous lint warnings (#5093)
- ed11f283d63915c014d1e176033a1bd0f1fd72ff Fix broken fbcode//faiss/tests:test_contrib - test_checkpoint (test_contrib.TestBigBatchSearch) (#5139)
- 277c53d299ac7e059d2791d2cb15956819287586 Fix PCA training bug and memory safety issues in VectorTransform.cpp (#5138)
- 83125407637fa69358de00ffcf7681947109d92a Fix integer type truncation in IDSelectorBatch bloom filter mask (#5136)
- 9cbc8da28efb865d9f8fed2f20ccf80ca2db0d20 Fix integer overflow and unbounded loop in Clustering.cpp (#5130)
- 3c4056d822129f2e211d9aa8fc1ee1436fed82a9 Fix race condition in HNSW::add_with_locks (#5129)
- 6e64c5d6873e9ba7a66918fe6f743997178b665b Bump GCC pin in faiss-gpu conda recipe to fix AVX2 SIMD miscompilation (#5125)
- 58f6ebb7fa84a995a3c11c7b4e8d3a9a0adaa489 Add per-read byte limit to SVS ReaderStreambuf to prevent OOM from corrupt index data (#5118)
- 817ecf9cd444dbd7461d3a09a0035b16448d2c6f Validate inverted list entry sizes against deserialization byte limit (#5117)
- e6be16241a10e1143dff263150315c220cc2a100 Validate VectorTransform dimension consistency during deserialization (#5115)
- 6707eacfaffc012bdb2ac95eae37b4baceef11f0 Add is_trained check to IndexIVF search and range_search to prevent querying untrained indexes (#5114)
- 349df7035bd2ddfa30f148c2e6225e90be91cdfa Validate IndexHNSW2Level storage type during deserialization and search (#5113)
- edd6f3b54c38e3ca063368f5079d2662b33098a1 Reject null quantizer during IVF index deserialization (#5112)
- 27078c52809b8ad43fbc73b6857a0865fe9c59c0 Fix OMP exception safety in IndexNNDescent and IndexNSG search (#5106)
- 99db159afd3f5e85f55dd959ef4cdb62e15086ad Fix OMP exception safety in IndexFlatCodes search (#5105)
- 40ad64694d4096b606a72a72622647a9421e8789 Fix cuVS build (#5107)
- 57bf47410f58616d1a104523327d474cdc264fb1 Fix int/size_t signedness mismatches in HNSW add (#5116)
- f9f116d303e93abd5bf79bc89d4227505e086448 Validate k_factor during deserialization to prevent search-time OOM (#5104)
- 2d8232c981f4148719bedbe9d0dcc2cd98f5fa3a Set code_size on IVFRaBitQFastScanScanner (#5099)
- b923091c0d87c68d2ff5e7076c5c1b1cf6489aed fix: GPU CAGRA copyFrom host-memory lifetime dependency and add regression test (#4968)
- 796977a1abc1ebfc42ce27fda655593089bcc70e Fix IndexBinaryMultiHash::reset() not clearing hash maps and avoid unnecessary copies (#5100)
- b86066a9989ea7223826e06d25170b9e03c67b66 fix: cagra bug (#4963)
- 91f6636b2a4d08c13c06e99dd0a1cb0bc77ceaf9 Add value-initialization to uninitialized member variables (#5089)
- 61ed7dc917b2db31edef74e9f825f1d04efbe80e Fix orphaned OpenMP directives in IndexRowwiseMinMax training functions (#5096)
- 740d3a396575114156cb2cd5e36ba98669a9998d Fix FastScan for indices without own inverted lists + performance optimizations in RaBitQ handler (#5080)
- 7cc63422d701ee9178ae0f7c8afa8aa6b9533a83 Suppress unused static constexpr variable warnings (#5087)
- aa3ce376ced6741b170ceef26bebbbf1f139cbec Fix race condition, memory management, debug output, and hashtable lookup in sorting.cpp (#5078)
- 86d33710702b668c1413d3bafe7be6048718f138 Validate FastScan M2 consistency during deserialization (#5056)
- 67f066f7a02f76d3178baccf4c31b4839ff0fee8 Validate SVS storage_kind during deserialization (#5055)
- e899476ceb8e21d1739851e2d7d6c0c70cac43e5 Default-initialize SVSStorageKind in IndexSVSVamana (#5054)
- c1e48a2ce12a008cd741f4d340af8026807cd020 Add search-path bounds checking to IndexIVF::search1 (#5040)
- 97d257433a59d7150a52491439f80314f2474cfb Fix correctness bugs in NNDescent Nhood copy/move operations and gen_random bounds (#5072)
- 06ef481afa0910b07b33eee47d52c542e06a988a Fix uncaught exceptions in IndexIVF::range_search_preassigned OpenMP region (#5053)
- 255250985a74e81a9a81822959588339dfae0cd0 Fix bugs in NSG search_on_graph and sync_prune (#5063)
- 7047dc5346abcfcf0a71b58f29aa9c138093af68 Fix priority_queue constructor inheritance for Apple Clang on macOS (#5065)
- e0a1aec811bb78af64c140fe47720cf9a80431cc Assorted Dynamic Dispatch cleanup fixes (#5060)
- 25296bad4f726aacff414a1f467f8a5f6eb0e306 Fix type mismatch bug and remove dead code in IndexRowwiseMinMax::sa_decode_impl (#5051)
- 7b2705c2698354ca75732970ba2e3526dc8330c7 Debug manifold crash with NaN (#5025)
- 9d3f97e620a24543f5e4d6e921473de0d63f2cd9 Fix duplicate words and grammar typos across codebase (#5044)
- 84262f09d4d0bc83c7147ddb53a97b402c2e0539 Fix d_out check bug and add descriptive error messages to VectorTransform assertions (#5047)
- ef28c6beb081d265cb6bdace8a3e85606bfd8a56 Fix uncaught exceptions in IndexIVF::search_preassigned OpenMP region (#5037)
- 8008ca2a70d7806694be80e365ff4b242982b699 Fix deadly signal in to_svs_metric() for unsupported metrics (#5032)
- 4b8da6820ead7ddd07f7d54e55bb4eb8bb2b8417 Fix IndexSVSVamana null-deref in deserialize_impl (#5029)
- ab3da522a4da1cd370052e36106f9743428d06b7 Cap ZnSphereCodecRec decode cache size (#5028)
- d68050d269c0aeb1bf7fe0ef5f5f662dba006477 Add missing input validation to IndexFlat::range_search and compute_distance_subset (#5038)
- c6fa50a4510760be0075e9cd0383b2290dec5057 Validate IndexIDMap id_map size during deserialization (#5024)
- 408786bec9d016f2b9893373a2ad40b82868eeb4 Validate AdditiveQuantizer dimension consistency during deserialization (#5023)
- 0ce4dfb2069019f9390fbfbf50be3052443325d5 Fix AVX512_SPR build failure for 512-bit SIMD types (#5030)
- a308adae08ae7b589e2f11da1ec08fa0d46b93c1 Harden ScalarQuantizer deserialization validation (#5020)
- ab46dcbeaed4817617f214604bdffc40026b5922 Validate qbs at deserialization time in FastScan index formats (#5019)
- 668da4a38684b95083e1ce737483d3ca64279c24 Validate IndexBinaryHash b <= code_size*8 during deserialization (#5011)
- 87bc9b08722d3a5ec01a91f5b16236fc56caf3bc Validate IndexFastScan quantizer state during deserialization (#5009)
- 4c391564a60e9e4441d5aeb3744347d855530e79 Validate codebooks size at deserialization time in read_AdditiveQuantizer (#5008)
- 8c1f3f48d48147777a0a2adaf3ceb661e0c43a7d read-heap-buffer-overflow (size 32) in float faiss::fvec_norm_L2sqr<>()
- 226ccf3e84d4439070262ff72dadab7de1d0e5f7 Fix OSX arm64 nightly: bump libopenblas from 0.3.30 to 0.3.32 (#5006)
- aeb170c15a105d640118bd32c37e9de81c7432be large-malloc (15023964028 bytes) in __sanitizer::RunMallocHooks() (#5018)
- 8550f2304af244a4a3ccbd8edb0f2d00aea11320 Validate VectorTransform dimensions during deserialization (#5007)
- 833d4aaf7b719f30a022271bab7b0d0917edfbff out-of-memory in __gnu_cxx::new_allocator<>::allocate() (#5014)
- d716be39ed1092d6e5545809b944b1742283920e Fix avx512_result_handlers build on aarch64 (#4965)
- 18c85e17e1ed5605c814d284e2704bca612f0e70 Add deserialization vector limit enforcement for ResidualCoarseQuantizer (#4997)
- b74f3cd8d9ba57a0549e91154057b8554c0fdd21 Validate IndexBinaryMultiHash b and nhash*b bounds during deserialization (#4984)
- b9f49601d109ae4c88adfec1ff84b5f40c00a2f3 Add invlists null check to IndexBinaryIVF::search (#4980)
- ba1981ca4ce1c6139e80a6d00ad6c26e44443dec Add input validation to IndexBinaryHNSW search preconditions (#4979)
- d26c15d954a1d0f372a4da59ee30622d314a666d Validate binary index consistency during deserialization (#4978)
- 2c28ef5382a21ba623afe56e3fdade97ae93b50c Validate RaBitQ qb during deserialization (#4983)
- 5a4d62146c606796b24711003bf18d87402f2400 deadly-signal (vector::_M_default_append) in __clang_call_terminate (#4986)
- 0759009bd7ddf290170c930c70a722badd05aa7d Fix compiler warnings for pip wheel builds (macOS + Windows) (#4989)
- 8a80b3dd6e2bcd1f0c6bef98aa396ac6d5a53e0d signal in faiss::ArrayInvertedListsPanorama::ArrayInvertedListsPanorama() (#4987)
- 12675affbcf1976c77a75d16813135f92584debc out-of-memory in __gnu_cxx::new_allocator<>::allocate() (#4985)
- 0a8e516058cb76e5e026858dc3ded478aa815683 Fix HNSW Panorama Perf Bug (#4974)
- 6dbf2713db8673d8007514d6f1d6375c72d5c99f Validate VectorTransform data during deserialization (#4981)
- 3f7938924d611b38adfdb00b7da27f24531decb7 Fix MSVC link failure: remove __restrict from fvec_madd specializations (#4972)
- d47b281578d98c25e3187201e6f138ba91f50e97 deadly-signal (vector::_M_default_append) in __clang_call_terminate (#4982)
- f851c54cd9fb89bda2680bbdeb28298d10c8178e large-malloc (6241124352 bytes) in __sanitizer::RunMallocHooks() (#4977)
- bd5ed00c384d5513917c51e348694d899e01718f Validate inverted lists pointer in IndexIVF operations (#4951)
- 398857b1ff7248579ab8a63ee5d570d0bfcce71e Handle empty index in IndexHNSWCagra base_level_only search (#4950)
- d647b596e01e1df4b79d5c513410b33463619aa8 Validate graph index data during deserialization (#4949)
- e5114c2368073d00316a866baa33e6595df3c396 fix: resolve compiler warnings in binary and miscellaneous Index files (#4940)
- 44f57ba4c91d2e4933f99f2c102df3ef5a3994cf fix: resolve compiler warnings in top-level FAISS source files (#4943)
- b3a2914982f5affcc61f280bfe3860b97c686f1f fix: resolve compiler warnings in test files (#4944)
- 88efcb981a22fabd9a6dbe6cd583c51362864420 fix: resolve compiler warnings in core base classes and headers (#4933)
- 7acc420929c40abc44899417b6e49e2eb28bb06e fix: resolve compiler warnings in inverted lists and cppcontrib (#4939)
- 64a2122ef1c8142287f277ec9e00cd2f8132d80a fix: resolve compiler warnings in IO, result handlers, and misc impl (#4936)
- a05f13ff6a395ad6916b6772ef4baa0d8ba72d26 fix: resolve compiler warnings in PQ, PreTransform, and Refine Index files (#4941)
- 7645426ac338b61856210a9d8a5eee9d653d47e4 fix: resolve compiler warnings in IVF Index implementations (#4942)
- f5142bfc5d8a8deed0268336c95352bab71dd781 large-malloc (7046432072 bytes) in __sanitizer::RunMallocHooks() (#4967)
- 4db01e3431b88b7363c55406b0babd6e8812977f fix: resolve compiler warnings in graph-based index implementations (#4956)
- ccc3cd619a8e2bc6ee970e61b1f4f7417724aadc Validate code vector sizes during index deserialization (#4948)
- ec30fb387c79a99454805d07f47311e77fd90bfc Use BitstringReader in IndexBinaryHash to avoid OOB reads (#4961)
- fa8ddd99a323ac289a96fd37b38e8c40f34004ad Fix flaky test for Panaorama/regular IVFFlat equivalence (#4954)
- 1e6729d53e5080e8290d1b7993969c3267f4c393 Validate quantizer data sizes during index deserialization (#4947)
- 62a19690988c0765e95d6707bb563a2233a9d2aa fix: resolve compiler warnings in distance and hamming utilities (#4937)
- 645a742b51813f11b033f9ebda909f0e60771dd5 Re add cuvs to cmakelists.txt (#4953)
- 5287db053721511ffa15a563d22665ea4ad02ae4 fix: resolve compiler warnings in graph-based index implementations (#4934)
- a48e45af3a7405a502711a1c63aeef5ba9b35ad1 fix: resolve compiler warnings in quantizer implementations (#4935)
- 32ca42f873053080435e57e306c3db123b3ea901 fix: resolve compiler warnings in SIMD, sorting, and misc utilities (#4938)
- 52d9fc705391db885138e86df42dbaf97d7f0a55 Add overflow check in READVECTOR macro (#4946)
- 0f7e65e53de6dd4a0273a3791fa70075b9de77f4 Enforce memory limit in read_ProductQuantizer (#4930)
- 9ea026cc93f349eeca25b1246af1e97745df84f7 Validate codes vector size in BlockInvertedLists deserialization (#4920)
- a4385dfd079f3de6046f6d68d56322600efdbaad Validate n_per_block and block_size in BlockInvertedLists deserialization (#4919)
- cdb725466736f1667d2f2a9426594e639097a967 Validate id_map size matches ntotal in IndexBinaryIDMap deserialization (#4917)
- 05ca04e469a6458f9abe25073f034987cb9844e4 Validate per-entry ilsz in read_binary_multi_hash_map (#4916)
- b023178c5dfc8980dd7dee40ef228198b13ef65a Fix MSVC OpenMP build: use idx_t instead of size_t in parallel for loop (#4922)
- 5b83ec68569b82ece3ca182f45b6951c761f8aff Fix OpenMP critical section contention in IndexBinaryHNSW search (#4909)
- e96ba2d13cb827a19cc3c6fe4996e8f963a07d79 fix: decouple coarse quantizer from cuvs index reset (#4885)
- 9d6b2e7b186cdc889c91866aff2ca2b5907ffb4d Additional input validation for index deserialization (#4899)
- 796bdf94a8878ebef1725fcea40f9dab06d1c5e5 Additional binary index input validation (#4898)
- d0434be649f556fbbefb7317899805a48f077553 Fix cloning for IVFFlatPanorama (#4887)
- 3e4c103d77a7b2cd1759f359fe073385920571f5 Validate dynamic_casts during index load (#4883)
- 47e53b4468a87fe4c33609b0dfd1902328e3ea6f Fix backward compatibility to use latest version (#4855)

Deprecated
- 812010fe19e1cac03e5f0bf529ddae9cb0ea4f1b Remove flaky test (#5169)
- 74dceee9038339bb57e232cfd655df3cdde3891c Remove RaBitQStats debug counters from all RaBitQ handlers (#5102)
- 000eec283a89342c411820de35bf9b118f39602c Delete old pq4_accumulate_loop files (replaced by dispatching.h) (#4905)


## [1.14.1] - 2026-03-04

Added
- 5cf2c4203f0e52f67504f154ae4dbea84906bc1f Expose IndexBinaryFlat to the C API. (#4834)
- db9ba35118d5230f92d466e17e19f5019ff8601d add hadamard transformation as an index for IVF (#4856)

Changed
- d2f8d3514003986ec9ed37c9b29d70818ccf686a removed conda-forge install documentation (#4843)
- c90c9dc544a8a82108d6499d7fafb3c3dc6fda2f Update python to include 3.13 and 3.14 (#4859)
- 8af77fe730f141d58fa7b0de8d3a33663e8c4b23 SIMD-optimize multi-bit RaBitQ inner product (#4850)
- ccc934f58660f42da677d5c253b550e61b153d5f ScalarQuantizer: split SIMD specializations into per-SIMD TUs + DD dispatch (#4839)

Fixed
- 28f79bd98efcb00c2bbf50a7eb30abc507ae49b6 Fix SWIG 4.4 multi-phase init: replace import_array() with import_array1(-1) (#4846)


## [1.14.0] - 2026-03-02

Added
- Add PEP 561 Python type stubs for the faiss package (#4840)
- Add conda-forge channel to INSTALL.md install commands (#4819)
- Add post_init_hook call to Python init (#4795)
- Add ARM SVE support for distance functions (#4798)
- Add Dynamic Dispatch OSS CI workflow (#4779)
- Add IndexFlatIPPanorama (#4787)
- Add benchmark to measure the ResultHandler overhead (#4778)
- Demo for a diversity filter (#4765)
- Add SVS binary size comparison demo and documentation (#4777)
- Add InvertedListScanner support for IndexIVFRaBitQFastScan (#4760)
- Add comprehensive ScalarQuantizer correctness tests (#4766)
- add IDSelector for knn_extra_metrics() (#4753)
- Add early stopping to k-means clustering (#4741)
- Add k-means++ and AFK-MC² centroid initialization methods (#4740)

Changed
- ScalarQuantizer: refactor SIMDWIDTH int → SIMDLevel enum (#4838)
- Fold IndexIVFPQ scanner helpers into templatized lambdas (#4836)
- Temporarily disable RaBitQ FastScan from backward compatibility test (#4841)
- Eliminate flat_storage by embedding auxiliary data in SIMD blocks (#4816)
- Rework PQ code distance for Dynamic Dispatch (#4808)
- fbcode/faiss/impl (#4832)
- fbcode/faiss/utils/simd_impl (#4833)
- fbcode/faiss/IndexFlat.cpp (#4831)
- fbcode/faiss (#4829)
- Implement distance_to_code for IVFRaBitQFastScanScanner (#4822)
- distance_to_code for IVFPQFastScan invertedlistscanner (#4821)
- Make dispatch_VectorDistance more compact (#4820)
- Update callers to use read_index_up API (#4818)
- fbcode/faiss/utils/simd_impl/distances_avx2.cpp (#4813)
- fbcode/faiss/impl/PolysemousTraining.cpp (#4814)
- fbcode/faiss/utils/sorting.cpp (#4815)
- VisitedTable -> unordered_set if ntotal is large (#4735)
- resulthandlers with AVX512 (#4806)
- put dispatch one level above (#4802)
- dynamic dispatch distances_simd (#4781)
- Introduce Dynamic Dispatch infrastructure with SIMDConfig (#4780)
- make runtime template selection more compact (#4793)
- support SearchParameters for IndexBinary (#4761)
- Support sharding of RaBitQ indices (#4790)
- Refactor ScalarQuantizer headers to use SIMD wrapper types (#4772)
- Split ScalarQuantizer.cpp into modular headers (NOOP) (#4786)
- Move factory_tools to main library and fix unaligned SIMD store (#4782)
- inline scanning code for fast distance computations (#4785)
- Enable Faiss for internal use (#4737)
- Address review comments on SQ correctness tests (#4771)
- Enable use of svs runtime conda package instead of tarball (#4747)
- generic result handlers for most indexes (#4762)
- Use nth_element for median computation in IndexLSH (#4653)
- Change default qb from 0 to 4 in RaBitQ indexes (#4757)
- Move reorder_2_heaps() into Heap.h (#4752)
- Improve naming due to codemod. simd_result_handlers (#4351)
- Dot Product Support Similarity Metric for IndexIVFFlatPanorama (#4732)
- Panorama Refactor and Code Cleanup (#4728)
- Update serialization backwards compatibility test with panorama and rabitq (#4736)

Fixed
- Additional index deserialization validation (#4844)
- Validate HNSW levels array entries during deserialization (#4827)
- Additional memory exception handling fixes for index_read.cpp (#4837)
- Catch attempts to deserialize undefined MetricTypes (#4823)
- BlockInvertedListsIOHook::read(): Don't leak on exception. (#4824)
- Harden ZnSphere lattice codec against invalid parameters (#4826)
- Validate n_levels > 0 in Panorama (#4825)
- Additional hardening of index load path (#4817)
- Deploy std::unique_ptr<> in index_read.cpp for exception safety (#4809)
- Fix to graph deserialization (#4812)
- Harden deserialization against integer overflow and buffer overflows (#4811)
- Fix CMake/Buck build discrepancies (#4807)
- Fix NSG off-by-one neighbor ID check (#4804)
- Fix CMake static targets missing SIMD sources and definitions (#4800)
- Enable -Wstring-conversion in faiss/PACKAGE +1
- Fix backward compat CI: use isolated conda environments (#4799)
- Fix string-conversion issue in faiss/impl/lattice_Zn.cpp +1 (#4794)
- Fix build pr 4761 (#4792)
- Fix: Remove -Wignored-attributes warning in mapped_io.cpp (#4775)
- Fix string-conversion issue in faiss/IndexHNSW.cpp
- Fix: Remove -Wswitch-unreachable warning in generic-inl.h (#4776)
- Fix string-conversion issue in faiss/invlists/OnDiskInvertedLists.cpp +5 (#4791)
- Fix OSX arm64 nightly by disabling hidden visibility on macOS (#4789)
- Fix FindMKL.cmake to detect Intel oneAPI MKL (2021+) (#4769)
- Fix lint errors in SVS integration code (#4774)
- Fix typos in demos, benchs, and other directories (#4743)
- Fix weak external symbol leakage (#4758)
- Fix compilation on macOS ARM64: Use faiss::idx_t instead of long test_hamming (#4755)
- Fix multi-bit RaBitQ IP metric filtering and f_add_ex computation (#4754)
- Fix IP metric distance computation in multi-bit RaBitQ (#4751)
- Reduce memory usage in timeout callback tests (#4745)
- Fix c++20 compilation in OSS Faiss for OSX ARM64 (#4733)

Deprecated
- Remove deprecated RAFT headers (#4731)


## [1.13.2] - 2025-12-19

Added
- 033e6acc6995d1adb9ea5317fadff152df3116bc Add RaBitQStats for tracking two-stage search filtering effectiveness (#4723)
- 64a236744b9ec4ca18d6b5d4e21c898f861a242a Add multi-bit support for IndexIVFRaBitQFastScan (#4722)
- cd2af8bc37628ac63dad736067200ac1291a77e7 Implement `IndexRefinePanorama` (#4683)
- 18d20febb579788be74724cd5a0bbc71632f978b Add multi-bit support for IndexRaBitQFastScan (#4721)
- 7372bc7982e6d15cd2048744094d510ddeb7495b Reapply `IndexHNSWFlatPanorama` with backward compatible serialization (#4692)
- 1721ebff6de6ed5a8481302123479be9d85059a2 Also add backwards compatible check for binary (#4714)
- 98bf8b3808ba325660006623afe177951579f3d9 Implement remaining `IndexFlatPanorama` functions (#4694)
- a695814f4c108a2ba7a82da9ac2b526b1fff118c Enable Intel ScalableVectorSearch support (#4548)
- d81a08e2409bb0ec2d2d6a2442d4beb7b2a8cbc9 Index serialization backward compatibility test (#4706)

Changed
- 9a6c02b061c4142a8e566d9d3360326140c95ad8 Rename RaBitQ factor structs for clarity and reorganize tests (#4730)
- 281a999abab90aed5b145b193aaff043b52045c2 Enable cuVS in Faiss (#4729)
- 6452d192cfaf67faf21808c285dfad1ec13b3d39 Update SVS binary to v0.1.0 (#4726)
- 1ea99d8073bfd5d20bfee4d63c4bb049b7d63154 clean up serialization tests (#4700)
- 89dd5a7b4ec9eb1f8c540828fc93c540f74c6699 Enable `-Wunused-exception-parameter` in faiss/PACKAGE +1

Fixed
- 5b19fca3f057b837ac898af52a8eb801c4744892 Allow over-writing centroid files to mitigate [S603653] (#4725)
- 337dfe8043a9bd9b8f4e2f3ec3c23fffb7b02654 Fix typos in demos, benchs, and other directories" (#4719)
- aea2b6bc8543f8a9b1b38e537cd55bcd9f6eb059 fix(docs): broken link to SVS in INSTALL.md (#4724)
- 4627695179e304adba1addd342355f366500149c Fix SVS Python tutorial (#4720)
- ac2e3abe3890fc7eaff06888878915bbda9c25b0 Update c_api install docs for CMake build system (#4702)
- abc294419ae2d235aea4a15813e168e742e34995 fix broken test due to renaming to avoid lint (#4712)
- 3d4d59fc3bd1986b31334c1a8bc6192a773b5666 Fix typos in demos, benchs, and other directories (#4709)


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
- separate the github build into two conditions (#4066)
- Improve naming due to codemod (#4070)
- improve naming due to codemod (#4067)
- improve naming due to codemod (#4071)
- improve naming due to codemod (#4072)
- fix nightly build (#4080)
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
- Set verbose before train (#3619)
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
- Fix a bug for a non-simdlib code of ResidualQuantizer (#3868)
- assign_index should default to null (#3855)
- Fix incorrectly counted the number of computed distances for HNSW (#3840)
- Add error for overflowing nbits during PQ construction (#3833)
- Fix radius search with HNSW and IP (#3698)
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
- HNSW bug fixed which improves the recall rate! Special thanks to zh Wang @hhy3 for this.
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
- The order of xb and xq was different between `faiss.knn` and `faiss.knn_gpu`.
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
- Support for interrupting computations with interrupt signal (ctrl-C) in python.
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

[Unreleased]: https://github.com/facebookresearch/faiss/compare/v1.14.2...HEAD
[1.14.2]: https://github.com/facebookresearch/faiss/compare/v1.14.1...v1.14.2
[1.14.1]: https://github.com/facebookresearch/faiss/compare/v1.14.0...v1.14.1
[1.14.0]: https://github.com/facebookresearch/faiss/compare/v1.13.2...v1.14.0
[1.13.2]: https://github.com/facebookresearch/faiss/compare/v1.13.1...v1.13.2
[1.13.1]: https://github.com/facebookresearch/faiss/compare/v1.13.0...v1.13.1
[1.13.0]: https://github.com/facebookresearch/faiss/compare/v1.12.0...v1.13.0
[1.12.0]: https://github.com/facebookresearch/faiss/compare/v1.11.0...v1.12.0
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
