

README for the link & code implementation
=========================================

What is this?
-------------

Link & code is an indexing method that combines HNSW indexing with
compression and exploits the neighborhood structure of the similarity
graph to improve the reconstruction. It is described in

```
@inproceedings{link_and_code,
   author = {Matthijs Douze and Alexandre Sablayrolles and Herv\'e J\'egou},
   title = {Link and code: Fast indexing with graphs and compact regression codes},
   booktitle = {CVPR},
   year = {2018}
}
```

ArXiV [here](https://arxiv.org/abs/1804.09996)

The necessary code for this paper was removed from Faiss in version 1.8.0.
For a functioning verinsion, use Faiss 1.7.4.
