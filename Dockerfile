FROM nvidia/cuda:8.0-devel-ubuntu16.04
MAINTAINER Pierre Letessier <pletessier@ina.fr>

RUN apt-get update -y
RUN apt-get install -y libopenblas-dev python-numpy python-dev swig git python-pip wget

RUN pip install matplotlib

COPY . /opt/faiss

WORKDIR /opt/faiss

ENV BLASLDFLAGS /usr/lib/libopenblas.so.0

RUN mv example_makefiles/makefile.inc.Linux ./makefile.inc

RUN make tests/test_blas -j $(nproc) && \
    make -j $(nproc) && \
    make demos/demo_sift1M -j $(nproc) && \
    make py

RUN cd gpu && \
    make -j $(nproc) && \
    make test/demo_ivfpq_indexing_gpu && \
    make py

ENV PYTHONPATH $PYTHONPATH:/opt/faiss

# RUN ./tests/test_blas && \
#     tests/demo_ivfpq_indexing


# RUN wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz && \
#     tar xf sift.tar.gz && \
#     mv sift sift1M

# RUN tests/demo_sift1M
