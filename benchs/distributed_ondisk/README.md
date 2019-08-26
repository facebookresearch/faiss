# Distributed on-disk index for 1T-scale datasets 

This is code corresponding to the description in [Indexing 1T vectors](https://github.com/facebookresearch/faiss/wiki/Indexing-1T-vectors). 
All the code is in python 3 (and not compatible with Python 2). 
The current code uses the Deep1B dataset for demonstration purposes, but can scale to 1000x larger.

## Distributed k-means

To cluster 200M vectors to 10M centroids, it is useful to have a distriubuted k-means implementation. 

The distributed k-means implementation here is based on 3 files:

- [`rpc.py`](rpc.py) is a very simple remote procedure call implementation based on sockets and pickle. 
It exposes the methods of an object on the server side so that they can be called from the client as if the object was local.

- [`distributed_kmeans.py`](distributed_kmeans.py) contains the k-means implementation. 
The main loop of k-means is re-implemented in python but follows closely the Faiss C++ implementation, and should not be significantly less efficient. 
It relies on a `DatasetAssign` object that does the assignement to centrtoids, which is the bulk of the computation. 
The object can be a Faiss CPU index, a GPU index or a set of remote GPU or CPU indexes.

- [`run_on_cluster.bash`](run_on_cluster.bash) contains the shell code to run the distributed k-means on a cluster. 

The distributed k-means works with a Python install that contains faiss and scipy (for sparse matrices).
It clusters the training data of Deep1B, this can be changed easily to any file in fvecs, bvecs or npy format that contains the training set. 
The training vectors may be too large to fit in RAM, but they are memory-mapped so that should not be a problem. 
The file is also assumed to be accessible from all server machines with eg. a distributed file system.

### Local tests 

Edit `distibuted_kmeans.py` to point `testdata` to your local copy of the dataset. 

Then, 4 levels of sanity check can be run: 
```bash
# reference Faiss C++ run
python distributed_kmeans.py --test 0
# using the Python implementation
python distributed_kmeans.py --test 1
# use the dispatch object (on local datasets)
python distributed_kmeans.py --test 2
# same, with GPUs
python distributed_kmeans.py --test 3
```
The output should look like [This gist](https://gist.github.com/mdouze/ffa01fe666a9325761266fe55ead72ad).

### Distributed sanity check

To run the distributed k-means, `distibuted_kmeans.py` has to be run both on the servers (`--server` option) and client sides (`--client` option). 
Edit the top of `run_on_cluster.bash` to set the path of the data to cluster. 

Sanity checks can be run with 
```bash 
# non distributed baseline
bash run_on_cluster.bash test_kmeans_0
# using all the machine's GPUs
bash run_on_cluster.bash test_kmeans_1
# distrbuted run, with one local server per GPU
bash run_on_cluster.bash test_kmeans_2
```
The test `test_kmeans_2` simulates a distributed run on a single machine by starting one server process per GPU and connecting to the servers via the rpc protocol. 
The output should look like [this gist](https://gist.github.com/mdouze/5b2dc69b74579ecff04e1686a277d32e).

### Distributed run








