FROM nvidia/cuda:8.0-devel-ubuntu16.04
MAINTAINER Pierre Letessier <pletessier@ina.fr>

RUN apt-get update -y
RUN apt-get install -y libopenblas-dev libpcre3 libpcre3-dev python-numpy python-dev git python-pip wget

RUN mkdir /opt/swig && cd /opt/swig && \
    wget http://downloads.sourceforge.net/swig/swig-3.0.1.tar.gz && \
    tar -xvzf swig-3.0.1.tar.gz && cd swig-3.0.1 && \
    ./configure --prefix=/usr && \
    make && \
    make install && \
    cd .. && \
    rm -fr swig-3.0.1.tar.gz swig-3.0.1

RUN pip install matplotlib

COPY . /opt/faiss

WORKDIR /opt/faiss

ENV BLASLDFLAGS /usr/lib/libopenblas.so.0

RUN mv example_makefiles/makefile.inc.Linux ./makefile.inc

RUN make tests/test_blas -j $(nproc) && \
    make -j $(nproc) && \
    make tests/demo_sift1M -j $(nproc)

RUN make py

RUN cd gpu && \
    make -j $(nproc) && \
    make test/demo_ivfpq_indexing_gpu && \
    make py

# RUN ./tests/test_blas && \
#     tests/demo_ivfpq_indexing


# RUN wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz && \
#     tar xf sift.tar.gz && \
#     mv sift sift1M

# RUN tests/demo_sift1M
