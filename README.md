

= Introduction =

Faiss contains several methods for similarity search. It assumes that the instances are represented as vectors and are identified by a 64-bit integer, and that the vectors can be compared with L2 distances or dot products. Vectors that are similar to a query vector are those that have the lowest L2 distance or the highest dot product with the query vector. It also supports cosine similarity, since this is a dot product on normalized vectors.

Faiss is built around an index type that stores a set of vectors, and provides a function to search in them with L2 and/or dot product vector comparison.

{pxlcld/nVck, width=640}

Some index types are simple baselines, such as exact search. Most of the available indexing structures correspond to various trade-offs with respect to

- search time
- search quality
- memory used per index vector 
- training time
- need for external data for unsupervised training

Most of the methods, like those based on binary vectors and compact quantization codes, solely use a compressed representation of the vectors and do not require to keep the original vectors. This generally comes at the cost of a less precise search but these methods can scale to billions of vectors in main memory on a single server. 

The library is mostly implemented in C++, with optional GPU support provided via CUDA. It compiles either with a standalone Makefile or within fbcode. It currently offers the following interfaces:

- C++
- Lua
- Python
- C (for a subset of the methods)

The GPU implementation is optional; it is similarly provided via C++ and Lua, and can accept input from either CPU or GPU memory. On a server with GPUs, the GPU indexes can be used a drop-in replacement for the CPU indexes (e.g., replace `IndexFlatL2` with `GpuIndexFlatL2`) and copies to/from GPU memory are handled automatically.

