
# The contrib modules

The contrib directory contains helper modules for Faiss for various tasks.

## Code structure

The contrib directory gets compiled in the module faiss.contrib.
Note that although some of the modules may depend on additional modules (eg. GPU Faiss, pytorch, hdf5), they are not necessarily compiled in to avoid adding dependencies. It is the user's responsibility to provide them.

In contrib, we are progressively dropping python2 support.

## List of contrib modules

### rpc.py

A very simple Remote Procedure Call library, where function parameters and results are pickled, for use with client_server.py

### client_server.py

The server handles requests to a Faiss index. The client calls the remote index.
This is mainly to shard datasets over several machines, see [Distributd index](https://github.com/facebookresearch/faiss/wiki/Indexes-that-do-not-fit-in-RAM#distributed-index)

### ondisk.py

Encloses the main logic to merge indexes into an on-disk index.
See [On-disk storage](https://github.com/facebookresearch/faiss/wiki/Indexes-that-do-not-fit-in-RAM#on-disk-storage)

### exhaustive_search.py

Computes the ground-truth search results for a dataset that possibly does not fit in RAM. Uses GPU if available.
Tested in `tests/test_contrib.TestComputeGT`

### gpu.py

(requires GPU Faiss)

Interoperability functions for pytorch and Faiss: pass GPU data without copying back to CPU.
Tested in `gpu/test/test_pytorch_faiss`

### datasets.py

(may require h5py)

Defintion of how to access data for some standard datsets.
