#!/bin/bash

# go one level up from faiss/gpu
top=$(dirname "${BASH_SOURCE[0]}")/..
echo "top=$top"
cd $top
echo "pwd=`pwd`"

# create all destination directories for hipified files into sibling 'gpu-rocm' directory
for src in $(find ./gpu -type d)
do
    dst=$(echo $src | sed 's/gpu/gpu-rocm/')
    echo "Creating $dst"
    mkdir -p $dst
done

# run hipify-perl against all *.cu *.cuh *.h *.cpp files, no renaming
# run all files in parallel to speed up
for ext in cu cuh h cpp
do
    for src in $(find ./gpu -name "*.$ext")
    do
        dst=$(echo $src | sed 's@./gpu@./gpu-rocm@')
        echo "$src -> $dst"
        hipify-perl -o=$dst $src &
    done
done
wait

# rename all hipified *.cu files to *.hip
for src in $(find ./gpu-rocm -name "*.cu")
do
    dst=${src%.cu}.hip
    mv $src $dst
done

# replace header include statements "<faiss/gpu/" with "<faiss/gpu-rocm"
for ext in hip cuh h cpp
do
    for src in $(find ./gpu-rocm -name "*.$ext")
    do
        sed -i 's@#include <faiss/gpu/@#include <faiss/gpu-rocm/@' $src
    done
done

# copy over CMakeLists.txt
for src in $(find ./gpu -name "CMakeLists.txt")
do
    dst=$(echo $src | sed 's@./gpu@./gpu-rocm@')
    echo "$src -> $dst"
    cp $src $dst
done
