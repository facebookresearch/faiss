FROM nvidia/cuda:8.0-devel-ubuntu16.04
MAINTAINER Pierre Letessier <pletessier@ina.fr>

RUN apt-get update -y && \
    apt-get install -y libopenblas-dev python-numpy python-dev swig python-pip curl

RUN pip install matplotlib

COPY . /opt/faiss

WORKDIR /opt/faiss

ENV BLASLDFLAGS=/usr/lib/libopenblas.so.0

RUN ./configure && \
    make -j $(nproc) && \
    make test && \
    make install

RUN make -C gpu -j $(nproc) && \
    make -C gpu/test

RUN make -C python gpu && \
    make -C python build && \
    make -C python install