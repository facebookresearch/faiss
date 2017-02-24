# Faiss 

Faiss is a library for efficient similarity search and clustering of dense vectors. It contains algorithms that search in sets of vectors of any size, up to ones that possibly do not fit in RAM. It also contains supporting code for evaluation and parameter tuning. Faiss is written in C++ with complete wrappers for Python/numpy. Some of the most useful algorithms are implemented on the GPU. It is developed by [Facebook AI Research](https://research.fb.com/category/facebook-ai-research-fair/).

## Introduction

Faiss contains several methods for similarity search. It assumes that the instances are represented as vectors and are identified by an integer, and that the vectors can be compared with L2 distances or dot products. Vectors that are similar to a query vector are those that have the lowest L2 distance or the highest dot product with the query vector. It also supports cosine similarity, since this is a dot product on normalized vectors.

Most of the methods, like those based on binary vectors and compact quantization codes, solely use a compressed representation of the vectors and do not require to keep the original vectors. This generally comes at the cost of a less precise search but these methods can scale to billions of vectors in main memory on a single server. 

The GPU implementation can accept input from either CPU or GPU memory. On a server with GPUs, the GPU indexes can be used a drop-in replacement for the CPU indexes (e.g., replace `IndexFlatL2` with `GpuIndexFlatL2`) and copies to/from GPU memory are handled automatically.

## Building 

The library is mostly implemented in C++, with optional GPU support provided via CUDA, and an optional Python interface. It compiles with a Makefile. See [INSTALL](INSTALL) for details.

## How Faiss works

Faiss is built around an index type that stores a set of vectors, and provides a function to search in them with L2 and/or dot product vector comparison. Some index types are simple baselines, such as exact search. Most of the available indexing structures correspond to various trade-offs with respect to

- search time
- search quality
- memory used per index vector 
- training time
- need for external data for unsupervised training

## Full documentation of Faiss

The full documentation, including a tutorial can be found in the [wiki page](http://github.com/facebookresearch/faiss/wiki). See the [doxygen documentation](http://rawgithub.com/facebookresearch/faiss/blob/master/docs/html/index.html) for per-class information. To reproduce results from research papers, refer to the [benchmarks README](benchs/README.md).


## Join the Faiss community

We monitor the [issues page](http://github.com/facebookresearch/faiss/issues) of the repository. You can report bugs, ask questions, etc.

## License

Faiss is licenced under CC-by-NC, see the LICENCE file for details. This licence may be relaxed to BSD in the future.

