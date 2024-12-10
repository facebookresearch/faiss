#!/bin/bash

# ビルドディレクトリの削除と再作成
rm -rf build
mkdir build

# CMakeの実行
cmake -B build . \
    -DFAISS_ENABLE_GPU=OFF \
    -DFAISS_ENABLE_PYTHON=OFF \
    -DBLA_VENDOR=Intel10_64_dyn \
    -DMKL_LIBRARIES=/usr/lib/x86_64-linux-gnu/libmkl_rt.so

# makeの実行
make -C build -j faiss

# C++ライブラリのインストール（管理者権限が必要な場合がある）
sudo make -C build -j install
