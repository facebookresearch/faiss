FROM nvidia/cuda:8.0-devel-centos7

# Install MKL
RUN yum-config-manager --add-repo https://yum.repos.intel.com/mkl/setup/intel-mkl.repo
RUN rpm --import https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS-2019.PUB
RUN yum install -y intel-mkl-2019.3-062
ENV LD_LIBRARY_PATH /opt/intel/mkl/lib/intel64:$LD_LIBRARY_PATH
ENV LIBRARY_PATH /opt/intel/mkl/lib/intel64:$LIBRARY_PATH
ENV LD_PRELOAD /usr/lib64/libgomp.so.1:/opt/intel/mkl/lib/intel64/libmkl_def.so:\
/opt/intel/mkl/lib/intel64/libmkl_avx2.so:/opt/intel/mkl/lib/intel64/libmkl_core.so:\
/opt/intel/mkl/lib/intel64/libmkl_intel_lp64.so:/opt/intel/mkl/lib/intel64/libmkl_gnu_thread.so

# Install necessary build tools
RUN yum install -y gcc-c++ make swig3

# Install Python and necesary headers/libs
RUN yum -y install https://centos7.iuscommunity.org/ius-release.rpm && \
    yum -y install python36u-devel python36u-pip && \
    ln -sf /usr/bin/python3.6 /usr/bin/python && \
    ln -sf /usr/bin/pip3.6 /usr/bin/pip && \
    pip install -U pip numpy==1.17.0 scipy==1.3.1 && \
    rm -rf /root/.cache/pip

COPY . /opt/faiss

WORKDIR /opt/faiss

# --with-cuda=/usr/local/cuda-8.0
RUN ./configure --prefix=/usr --libdir=/usr/lib64 --without-cuda PYTHONFLAGS="-I/usr/include/python3.6m/ -I/usr/lib64/python3.6/site-packages/numpy/core/include/"
RUN make -j $(nproc)
RUN make -C python
RUN make test
RUN make install
RUN make -C demos demo_ivfpq_indexing && ./demos/demo_ivfpq_indexing

ENV PYTHONPATH="$PYTHONPATH:/opt/faiss/python"