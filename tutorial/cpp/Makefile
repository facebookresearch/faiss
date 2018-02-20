MAKEFILE_INC=../../makefile.inc

-include $(MAKEFILE_INC)

all: cpu gpu

cpu: tutorial1 tutorial2 tutorial3

gpu: tutorial4 tutorial5

tutorial1: 1-Flat.cpp ../../libfaiss.a
	$(CC) -o $@ $(CFLAGS) $^ -I../../../ $(LDFLAGS) $(BLASLDFLAGS)

tutorial2: 2-IVFFlat.cpp ../../libfaiss.a
	$(CC) -o $@ $(CFLAGS) $^ -I../../../ $(LDFLAGS) $(BLASLDFLAGS)

tutorial3: 3-IVFPQ.cpp ../../libfaiss.a
	$(CC) -o $@ $(CFLAGS) $^ -I../../../ $(LDFLAGS) $(BLASLDFLAGS)

tutorial4: 4-GPU.cpp ../../libfaiss.a ../../gpu/libgpufaiss.a
	$(NVCC) $(NVCCFLAGS) -o $@ $^ -I../../../ -Xcompiler -fopenmp -lcublas \
	$(BLASLDFLAGSNVCC)

tutorial5: 5-Multiple-GPUs.cpp ../../libfaiss.a ../../gpu/libgpufaiss.a
	$(NVCC) $(NVCCFLAGS) -o $@ $^ -I../../../ -Xcompiler -fopenmp -lcublas \
	$(BLASLDFLAGSNVCC)
