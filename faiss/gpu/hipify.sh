#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

function hipify_dir()
{
    # print dir name
    cd "$1" || exit
    echo "Hipifying $(pwd)"

    # create all destination directories for hipified files into sibling 'gpu-rocm' directory
    while IFS= read -r -d '' src
    do
        dst="${src//gpu/gpu-rocm}"

        if [ -d $dst ]; then
            #Clearing out any leftover files and directories
            echo "Removing old $dst"
            rm -rf "$dst"
        fi

        #Making directories
        echo "Creating $dst"
        mkdir -p "$dst"
    done <   <(find ./gpu -type d -print0)

    # run hipify-perl against all *.cu *.cuh *.h *.cpp files, no renaming
    # run all files in parallel to speed up
    for ext in cu cuh h cpp c
    do
        while IFS= read -r -d '' src
        do
            dst="${src//\.\/gpu/\.\/gpu-rocm}"
            hipify-perl -o="$dst.tmp" "$src" &
        done <   <(find ./gpu -name "*.$ext" -print0)
    done
    wait

    # rename all hipified *.cu files to *.hip
    while IFS= read -r -d '' src
    do
        dst=${src%.cu.tmp}.hip.tmp
        mv "$src" "$dst"
    done <   <(find ./gpu-rocm -name "*.cu.tmp" -print0)

    # replace header include statements "<faiss/gpu/" with "<faiss/gpu-rocm"
    # replace thrust::cuda::par with thrust::hip::par
    # adjust header path location for hipblas.h to avoid unnecessary deprecation warnings
    # adjust header path location for hiprand_kernel.h to avoid unnecessary deprecation warnings
    for ext in hip cuh h cpp c
    do
        while IFS= read -r -d '' src
        do
            sed -i 's@#include <faiss/gpu/@#include <faiss/gpu-rocm/@' "$src"
            sed -i 's@thrust::cuda::par@thrust::hip::par@' "$src"
            sed -i 's@#include <hipblas.h>@#include <hipblas/hipblas.h>@' "$src"
            sed -i 's@#include <hiprand_kernel.h>@#include <hiprand/hiprand_kernel.h>@' "$src"
        done <   <(find ./gpu-rocm -name "*.$ext.tmp" -print0)
    done

    # hipify was run in parallel above
    # don't copy the tmp file if it is unchanged
    for ext in hip cuh h cpp c
    do
        while IFS= read -r -d '' src
        do
            dst=${src%.tmp}
            if test -f "$dst"
            then
                if diff -q "$src" "$dst" >& /dev/null
                then
                    echo "$dst [unchanged]"
                    rm "$src"
                else
                    echo "$dst"
                    mv "$src" "$dst"
                fi
            else
                echo "$dst"
                mv "$src" "$dst"
            fi
        done <   <(find ./gpu-rocm -name "*.$ext.tmp" -print0)
    done

    # copy over CMakeLists.txt
    while IFS= read -r -d '' src
    do
        dst="${src//\.\/gpu/\.\/gpu-rocm}"
        if test -f "$dst"
        then
            if diff -q "$src" "$dst" >& /dev/null
            then
                echo "$dst [unchanged]"
            else
                echo "$dst"
                cp "$src" "$dst"
            fi
        else
            echo "$dst"
            cp "$src" "$dst"
        fi
    done <   <(find ./gpu -name "CMakeLists.txt" -print0)

    # Copy over other files
    other_exts="py"
    for ext in $other_exts
    do
        while IFS= read -r -d '' src
        do
            dst="${src//\.\/gpu/\.\/gpu-rocm}"
            if test -f "$dst"
            then
                if diff -q "$src" "$dst" >& /dev/null
                then
                    echo "$dst [unchanged]"
                else
                    echo "$dst"
                    cp "$src" "$dst"
                fi
            else
                echo "$dst"
                cp "$src" "$dst"
            fi
        done <   <(find ./gpu -name "*.$ext" -print0)
    done
}

# Convert the faiss/gpu dir
dir_name=$(dirname "${BASH_SOURCE[0]}")/..
hipify_dir $dir_name

# Convert the faiss/c_api dir
dir_name=$(dirname "${BASH_SOURCE[0]}")/../../c_api
hipify_dir $dir_name
