#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Usage: ./gpu/hipify.sh
#

function hipify_dir()
{
    # print dir name
    cd "$1" || exit
    echo "Hipifying $(pwd)"

    if [ -d ./gpu-tmp ]; then
        #Clearing out any leftover files and directories
        echo "Removing old ./gpu-tmp"
        rm -rf ./gpu-tmp
    fi

    if [ -d ./gpu ]; then
        #Making a temp directory to implement pre hipify rules
        echo "Creating ./gpu-tmp"
        cp -r ./gpu ./gpu-tmp

        # adjust __nv_bfloat162 before hipify because of inaccurate conversions
        # adjust __nv_bfloat16 before hipify because of inaccurate conversions
        for ext in hip cuh h cpp c cu cuh
        do
            while IFS= read -r -d '' src
            do
                sed -i 's@__nv_bfloat162@__hip_bfloat162@' "$src"
                sed -i 's@__nv_bfloat16@__hip_bfloat16@' "$src"
            done <   <(find ./gpu-tmp -name "*.$ext" -print0)
        done
    else
        echo "Can't find the gpu/ dir"
        exit
    fi

    # Create a backup of the original gpu directory
    if [ -d ./gpu-backup ]; then
        echo "Removing old ./gpu-backup"
        rm -rf ./gpu-backup
    fi
    echo "Creating backup of original gpu directory"
    cp -r ./gpu ./gpu-backup

    # run hipify-perl against all *.cu *.cuh *.h *.cpp files, no renaming
    # run all files in parallel to speed up
    for ext in cu cuh h cpp c
    do
        while IFS= read -r -d '' src
        do
            # Output to the original gpu directory but with .tmp extension
            dst="${src//\.\/gpu-tmp/\.\/gpu}"
            hipify-perl -o="$dst.tmp" "$src" &
        done <   <(find ./gpu-tmp -name "*.$ext" -print0)
    done
    wait

    # rename all hipified *.cu files to *.hip
    while IFS= read -r -d '' src
    do
        dst=${src%.cu.tmp}.hip.tmp
        mv "$src" "$dst"
    done <   <(find ./gpu -name "*.cu.tmp" -print0)

    if [ -d ./gpu-tmp ]; then
        #Clearing out any leftover files and directories
        echo "Removing ./gpu-tmp"
        rm -rf ./gpu-tmp
    fi

    # replace thrust::cuda::par with thrust::hip::par
    # adjust header path location for hipblas.h to avoid unnecessary deprecation warnings
    # adjust header path location for hiprand_kernel.h to avoid unnecessary deprecation warnings
    for ext in hip cuh h cpp c
    do
        while IFS= read -r -d '' src
        do
            sed -i 's@thrust::cuda::par@thrust::hip::par@' "$src"
            sed -i 's@#include <hipblas.h>@#include <hipblas/hipblas.h>@' "$src"
            sed -i 's@#include <hiprand_kernel.h>@#include <hiprand/hiprand_kernel.h>@' "$src"
        done <   <(find ./gpu -name "*.$ext.tmp" -print0)
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
        done <   <(find ./gpu -name "*.$ext.tmp" -print0)
    done

    # No need to copy CMakeLists.txt or other files as we're working directly in the gpu directory
}

# Convert the faiss/gpu dir
dir_name=$(dirname "${BASH_SOURCE[0]}")/..
hipify_dir $dir_name

# Convert the faiss/c_api dir
dir_name=$(dirname "${BASH_SOURCE[0]}")/../../c_api
hipify_dir $dir_name
