# Distributed on-disk index for 1T-scale datasets 

This is code corresponding to the description in [Indexing 1T vectors](https://github.com/facebookresearch/faiss/wiki/Indexing-1T-vectors). 
All the code is in python 3 (and not compatible with Python 2). 
The current code uses the Deep1B dataset for demonstration purposes, but can scale to 1000x larger.

## Distributed k-means

To cluster 500M vectors to 10M centroids, it is useful to have a distriubuted k-means implementation. 
The distribution simply consists in splitting the training vectors across machines (servers) and have them do the assignment. 
The master/client then synthesizes the results and updates the centroids.

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

The way the script can be distributed depends on the cluster's scheduling system. 
Here we use Slurm, but it should be relatively easy to adapt to any scheduler that can allocate a set of matchines and start the same exectuable on all of them. 

The command 
```
bash run_on_cluster.bash slurm_distributed_kmeans
```
asks SLURM for 5 machines with 4 GPUs each with the `srun` command. 
All 5 machines run the script with the `slurm_within_kmeans_server` option. 
They determine the number of servers and their own server id via the `SLURM_NPROCS` and `SLURM_PROCID` environment variables.

All machines start `distributed_kmeans.py` in server mode for the slice of the dataset they are responsible for.

In addition, the machine #0 also starts the client. 
The client knows who are the other servers via the variable `SLURM_JOB_NODELIST`. 
It connects to all clients and performs the clustering. 

The output should look like [this gist](https://gist.github.com/mdouze/8d25e89fb4af5093057cae0f917da6cd).

### Run used for deep1B

For the real run, we run the clustering on 50M vectors to 1M centroids. 
This is just a matter of using as many machines / GPUs as possible in setting the output centroids with the `--out filename` option.
Then run
```
bash run_on_cluster.bash deep1b_clustering
```

The last lines of output read like: 
```
  Iteration 19 (898.92 s, search 875.71 s): objective=1.33601e+07 imbalance=1.303 nsplit=0
 0: writing centroids to /checkpoint/matthijs/ondisk_distributed/1M_centroids.npy
```

This means that the total training time was 899s, of which 876s were used for computation. 
However, the computation includes the I/O overhead to the assignment servers. 
In this implementation, the overhead of transmitting the data is non-negligible and so is the centroid computation stage. 
This is due to the inefficient Python implementation and the RPC protocol that is not optimized for broadcast / gather (like MPI). 
However, it is a simple implementation that should run on most clusters.

## Making the trained index

After the centroids are obtained, an empty trained index must be constructed. 
This is done by: 

- applying a pre-processing stage (a random rotation) to balance the dimensions of the vectors. This can be done after clustering, the clusters are just rotated as well.

- wrapping the centroids into a HNSW index to speed up the CPU-based assignment of vectors

- training the 6-bit scalar quantizer used to encode the vectors

This is performed by the script [`make_trained_index.py`](make_trained_index.py). 

## Building the index by slices

We call the slices "vslisces" as they are vertical slices of the big matrix (see explanation in the wiki section [Split across datanbase partitions](https://github.com/facebookresearch/faiss/wiki/Indexing-1T-vectors#split-across-database-partitions)

The script [make_index_vslice.py](make_index_vslice.py) makes an index for a subset of the vectors of the input data and stores it as an independent index. 
There are 200 slices of 5M vectors each for Deep1B.
It can be run in a brute-force parallel fashion, there is no constraint on ordering. 
To run the script in parallel on a slurm cluster, use: 
```
bash run_on_cluster.bash make_index_vslices
```
For a real dataset, the data would be read from a DBMS. 
In that case, reading the data and indexing it in parallel is worthwhile because reading is very slow.

## Splitting accross inverted lists

The 200 slices need to be merged together. 
This is done with the script [merge_to_ondisk.py](merge_to_ondisk.py), that memory maps the 200 vertical slice indexes, extracts a subset of the inverted lists and writes them to a contiguous horizontal slice. 
We slice the inverted lists into 50 horizontal slices. 
This is run with 
```
bash run_on_cluster.bash make_index_hslices
```


