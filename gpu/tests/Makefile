# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD+Patents license found in the
# LICENSE file in the root directory of this source tree.

-include ../../makefile.inc

demo_ivfpq_indexing_gpu: demo_ivfpq_indexing_gpu.cpp ../../libfaiss.a ../libgpufaiss.a
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(NVCCLDFLAGS) $(LIBS) $(NVCCLIBS)

clean:
	rm -f demo_ivfpq_indexing_gpu

.PHONY: clean
