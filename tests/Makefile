MAKEFILE_INC=../makefile.inc

-include $(MAKEFILE_INC)

CXX_TESTS = test_ivfpq_indexing test_ivfpq_codec test_merge test_ondisk_ivf \
	test_index_binary_flat test_pairs_decoding

all: $(CXX_TESTS) ../py

run: all
	./run_tests.sh

%: %.cpp  ../libfaiss.a gtest/make/gtest_main.a
	$(CXX) $(CFLAGS) -o $@ $^ $(LDFLAGS) $(BLASLDFLAGS) \
	-Igtest/include -I../..

../libfaiss.a:
	cd ../ && make libfaiss.a

../py:
	cd ../ && make py

gtest/make/gtest_main.a: gtest
	cd gtest/make && make gtest_main.a

gtest:
	curl -L https://github.com/google/googletest/archive/release-1.8.0.tar.gz | tar xz && \
	mv googletest-release-1.8.0/googletest gtest && \
	rm -rf googletest-release-1.8.0

clean:
	rm -rf $(CC_TESTS) gtest
