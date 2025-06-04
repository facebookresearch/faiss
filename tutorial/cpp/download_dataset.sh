#!/bin/bash

echo "====== Downloading sift 1M dataset ======"
# official dataset from http://corpus-texmex.irisa.fr/
wget ftp://ftp.irisa.fr/local/texmex/corpus/sift.tar.gz

echo "====== Unzipping sift 1M dataset into data/ directory ======"
mkdir -p data
tar -xzf sift.tar.gz -C data


