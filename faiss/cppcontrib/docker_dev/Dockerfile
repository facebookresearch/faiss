FROM ubuntu:22.04

RUN apt update && apt install -y python3 python3-pip gcc g++ mc git swig sudo libomp-dev libopenblas-dev wget
RUN pip3 install numpy==1.26.4 scipy pytest
RUN cd /root && git clone https://github.com/facebookresearch/faiss
RUN wget -qO- "https://cmake.org/files/v3.26/cmake-3.26.5-linux-x86_64.tar.gz" | sudo tar --strip-components=1 -xz -C /usr/local
RUN cd /root/faiss && /usr/local/bin/cmake -B build -DFAISS_ENABLE_GPU=OFF -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release .
RUN cd /root/faiss && make -C build -j 8 faiss
RUN cd /root/faiss && make -C build -j 8 swigfaiss
RUN cd /root/faiss/build/faiss/python && python3 setup.py install
