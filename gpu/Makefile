# Copyright (c) 2015-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the BSD+Patents license found in the
# LICENSE file in the root directory of this source tree.

-include ../makefile.inc

all: libgpufaiss.a libgpufaiss.$(SHAREDEXT)

CPPOBJ =    GpuResources.o \
            IndexProxy.o \
            StandardGpuResources.o \
            GpuAutoTune.o \
            GpuClonerOptions.o \
            impl/RemapIndices.o \
            utils/DeviceMemory.o \
            utils/StackDeviceMemory.o \
            utils/DeviceUtils.o \
            utils/Timer.o \
            utils/MemorySpace.o \
            utils/WorkerThread.o


INS = 1 32 64 128 256 F512 T512 F1024 T1024

CUOBJ =     impl/BinaryDistance.o \
            impl/BinaryFlatIndex.o \
            impl/BroadcastSum.o \
            impl/Distance.o \
            impl/FlatIndex.o \
            impl/InvertedListAppend.o \
            impl/IVFBase.o \
            impl/IVFFlat.o \
            impl/IVFFlatScan.o \
            impl/IVFPQ.o \
            impl/IVFUtils.o \
            impl/IVFUtilsSelect1.o \
            impl/IVFUtilsSelect2.o \
            impl/L2Norm.o \
            impl/L2Select.o \
            impl/PQCodeDistances.o \
            impl/PQScanMultiPassNoPrecomputed.o \
            impl/PQScanMultiPassPrecomputed.o \
            impl/VectorResidual.o \
            GpuDistance.o \
            GpuIndex.o \
            GpuIndexBinaryFlat.o \
            GpuIndexFlat.o \
            GpuIndexIVF.o \
            GpuIndexIVFFlat.o \
            GpuIndexIVFPQ.o  \
            utils/Float16.o \
            utils/MatrixMult.o \
            utils/BlockSelectFloat.o \
            utils/BlockSelectHalf.o \
            utils/WarpSelectFloat.o \
            utils/WarpSelectHalf.o \
            utils/nvidia/fp16_emu.o \
            $(foreach v,$(INS),   \
                utils/blockselect/BlockSelectHalf$(v).o \
                utils/blockselect/BlockSelectFloat$(v).o \
                utils/warpselect/WarpSelectHalf$(v).o \
                utils/warpselect/WarpSelectFloat$(v).o \
             )

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(CPUFLAGS) -c $< -o $@ $(CUDACFLAGS)

%.o: %.cu
	$(NVCC) $(NVCCFLAGS) -g -O3 -c $< -o $@

libgpufaiss.a: $(CPPOBJ) $(CUOBJ)
	ar r $@ $^

libgpufaiss.$(SHAREDEXT): $(CPPOBJ) $(CUOBJ)
	$(CXX) $(SHAREDFLAGS) $(LDFLAGS) $(NVCCLDFLAGS) \
		-o libgpufaiss.$(SHAREDEXT) $^ $(LIBS) $(NVCCLIBS)

clean:
	rm -rf *.o impl/*.o utils/*.o libgpufaiss.a \
		libgpufaiss.$(SHAREDEXT) \

-include depend

depend:
	for i in $(patsubst %.o,%.cpp,$(CPPOBJ)) \
		 $(patsubst %.o,%.cu,$(CUOBJ)); do \
	    $(CXXCPP) $(CPPFLAGS) -x c++ -MM $$i; \
	done > depend

.PHONY: all clean
