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
        hipify-perl -o=$dst.tmp $src &
    done
done
wait

# rename all hipified *.cu files to *.hip
for src in $(find ./gpu-rocm -name "*.cu.tmp")
do
    dst=${src%.cu.tmp}.hip.tmp
    mv $src $dst
done

# replace header include statements "<faiss/gpu/" with "<faiss/gpu-rocm"
# replace thrust::cuda::par with thrust::hip::par
for ext in hip cuh h cpp
do
    for src in $(find ./gpu-rocm -name "*.$ext.tmp")
    do
        sed -i 's@#include <faiss/gpu/@#include <faiss/gpu-rocm/@' $src
        sed -i 's@thrust::cuda::par@thrust::hip::par@' $src
    done
done

# hipify was run in parallel above
# don't copy the tmp file if it is unchanged
for ext in hip cuh h cpp
do
    for src in $(find ./gpu-rocm -name "*.$ext.tmp")
    do
        dst=${src%.tmp}
        if test -f $dst
        then
            if diff -q $src $dst >& /dev/null
            then
                echo "$dst [unchanged]"
                rm $src
            else
                echo "$dst"
                mv $src $dst
            fi
        else
            echo "$dst"
            mv $src $dst
        fi
    done
done

# copy over CMakeLists.txt
for src in $(find ./gpu -name "CMakeLists.txt")
do
    dst=$(echo $src | sed 's@./gpu@./gpu-rocm@')
    if test -f $dst
    then
        if diff -q $src $dst >& /dev/null
        then
            echo "$dst [unchanged]"
        else
            echo "$dst"
            cp $src $dst
        fi
    else
        echo "$dst"
        cp $src $dst
    fi
done
