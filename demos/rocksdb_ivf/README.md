# Storing Faiss inverted lists in RocksDB

Demo of storing the inverted lists of any IVF index in RocksDB or any similar key-value store which supports the prefix scan operation.

# How to build

We use conda to create the build environment for simplicity. Only tested on Linux x86.

```
conda create -n rocksdb_ivf
conda activate rocksdb_ivf
conda install pytorch::faiss-cpu conda-forge::rocksdb cmake make gxx_linux-64 sysroot_linux-64
cd ~/faiss/demos/rocksdb_ivf
cmake -B build .
make -C build -j$(nproc)
```

# Run the example

```
cd ~/faiss/demos/rocksdb_ivf/build
./rocksdb_ivf test_db
```
