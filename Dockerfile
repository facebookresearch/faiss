FROM nvidia/cuda:8.0-devel-ubuntu16.04
MAINTAINER Alues <alues@icloud.com>

# PIP source
ARG pip_srouce=https://pypi.tuna.tsinghua.edu.cn/simple

# Install Env
RUN apt-get update && \
    apt-get install -y wget curl git swig gcc && \
    apt-get install -y libopenblas-dev python-dev python-pip python-numpy

RUN pip install -i ${pip_srouce} --no-cache-dir --upgrade --ignore-installed pip
RUN pip install -i ${pip_srouce} cython matplotlib pandas jupyter sklearn scipy

# Clone FAISS
# WORKDIR /opt
# RUN git clone https://github.com/facebookresearch/faiss.git

COPY . /opt/faiss
WORKDIR /opt/faiss

# Compiling FAISS
RUN ./configure && \
    make -j $(nproc) && \
    make -j $(nproc) -C gpu && \
    make -j $(nproc) -C python gpu && \
    make -j $(nproc) -C python build && \
    make -j $(nproc) -C python install

# Jupyter
RUN mkdir -p /root/jupyter
WORKDIR /root/jupyter

CMD ["jupyter-notebook",  "--no-browser", "--ip='*'", "--notebook-dir=/root/jupyter", "--allow-root", "--port=8000"]
