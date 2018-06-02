# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD+Patents license found in the
# LICENSE file in the root directory of this source tree.

-include ../makefile.inc

TESTS_SRC = $(wildcard *.cpp)
TESTS     = $(TESTS_SRC:.cpp=.o)

all: run

run: tests
	./tests

tests: $(TESTS) ../libfaiss.a gtest/make/gtest_main.a
	$(CXX) -o $@ $^ $(LDFLAGS) $(LIBS)

%.o: %.cpp gtest
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(CPUFLAGS) -c -o $@ $< -Igtest/include -I../..

gtest/make/gtest_main.a: gtest
	make -C gtest/make CXX="$(CXX)" CXXFLAGS="$(CXXFLAGS)" gtest_main.a

gtest:
	curl -L https://github.com/google/googletest/archive/release-1.8.0.tar.gz | tar xz && \
	mv googletest-release-1.8.0/googletest gtest && \
	rm -rf googletest-release-1.8.0

clean:
	rm -f test_runner
	rm -f *.o
	rm -rf gtest


.PHONY: all clean run
