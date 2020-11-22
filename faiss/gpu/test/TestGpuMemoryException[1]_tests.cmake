add_test( TestGpuMemoryException.AddException /home/lvtingxun/Project/tools/faiss/faiss/gpu/test/TestGpuMemoryException [==[--gtest_filter=TestGpuMemoryException.AddException]==] --gtest_also_run_disabled_tests)
set_tests_properties( TestGpuMemoryException.AddException PROPERTIES WORKING_DIRECTORY /home/lvtingxun/Project/tools/faiss/faiss/gpu/test SKIP_REGULAR_EXPRESSION [==[\[  SKIPPED \]]==])
set( TestGpuMemoryException_TESTS TestGpuMemoryException.AddException)
