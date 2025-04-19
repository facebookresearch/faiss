#!/bin/bash
argList=()
i=1
for var in "${@:2}"
do
    if [[ -z $var ]]; then
        echo "$i = \"\""
        argList+=("")
    else
        echo "$i = $var"
        argList+=($var)
    fi
    i=$((i+1))
done

USE_NVPROF="${USE_NVPROF:-1}"

if [ $USE_NVPROF = 1 ]; then
    nvprof --print-gpu-trace -f -o $1.$OMPI_COMM_WORLD_RANK.nvprof ${argList[@]+"${argList[@]}"}
else
    nsys nvprof --profile-from-start off -f -o $1.$OMPI_COMM_WORLD_RANK.nvprof ${argList[@]+"${argList[@]}"}
fi
