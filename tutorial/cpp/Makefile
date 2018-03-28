MAKEFILE_INC=../../makefile.inc

-include $(MAKEFILE_INC)

NVCCLDFLAGS =	-Xcompiler \"-Wl,-rpath=../../:../../gpu/\" \
		-L../.. -L../../gpu -lfaiss -lgpufaiss

LDFLAGS = -L../.. -Wl,-rpath=../.. -lfaiss

all: cpu gpu

cpu: 1-Flat 2-IVFFlat 3-IVFPQ

gpu: 4-GPU 5-Multiple-GPUs

1-Flat: 1-Flat.cpp ../../libfaiss.$(SHAREDEXT)
	$(CXX) -o $@ $(CXXFLAGS) $< -I../../../ $(LDFLAGS)

2-IVFFlat: 2-IVFFlat.cpp ../../libfaiss.$(SHAREDEXT)
	$(CXX) -o $@ $(CXXFLAGS) $< -I../../../ $(LDFLAGS)

3-IVFPQ: 3-IVFPQ.cpp ../../libfaiss.$(SHAREDEXT)
	$(CXX) -o $@ $(CXXFLAGS) $< -I../../../ $(LDFLAGS)

4-GPU: 4-GPU.cpp ../../libfaiss.$(SHAREDEXT) ../../gpu/libgpufaiss.$(SHAREDEXT)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(NVCCLDFLAGS) -I../../../

5-Multiple-GPUs: 5-Multiple-GPUs.cpp ../../libfaiss.$(SHAREDEXT) \
	../../gpu/libgpufaiss.$(SHAREDEXT)
	$(NVCC) $(NVCCFLAGS) -o $@ $< $(NVCCLDFLAGS) -I../../../

../../libfaiss.$(SHAREDEXT):
	cd ../../ && make libfaiss.$(SHAREDEXT)

../../gpu/libgpufaiss.$(SHAREDEXT):
	cd ../../gpu/ && make libgpufaiss.$(SHAREDEXT)

clean:
	rm -f 1-Flat 2-IVFFlat 3-IVFPQ 4-GPU 5-Multiple-GPUs
