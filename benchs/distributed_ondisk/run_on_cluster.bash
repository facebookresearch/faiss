#! /bin/bash

set -e

todo=$1

# the training data of the Deep1B dataset
testdata=/datasets01_101/simsearch/041218/deep1b/learn.fvecs

if [ -z "$todo" ]; then
    echo "nothing to do"
    exit 1
elif [ $todo == test_kmeans_0 ]; then
    # non distributed baseline
    python distributed_kmeans.py \
           --indata $testdata --i1 1000000 \
           --k 4000

elif [ $todo == test_kmeans_1 ]; then
    # using all the machine's GPUs
    python distributed_kmeans.py \
           --indata $testdata --i1 1000000 \
           --k 4000 --gpu -1

elif [ $todo == test_kmeans_2 ]; then
    # distrbuted run, with one local server per GPU
    ngpu=$( echo /dev/nvidia? | wc -w )
    baseport=12012

    # kill background porcesses on output of this script
    trap 'kill -HUP 0' 0

    hostports=''

    for((gpu=0;gpu<ngpu;gpu++)); do

        i0=$((1000000 * gpu / ngpu))
        i1=$((1000000 * (gpu + 1) / ngpu))
        port=$(( baseport + gpu ))

        echo "start server $gpu for range $i0:$i1"

        python distributed_kmeans.py \
               --indata $testdata \
               --i0 $i0 --i1 $i1 \
               --server --gpu $gpu \
               --port $port --ipv4 &

        hostports="$hostports localhost:$port"
    done

    # lame way of making sure all servers are running
    sleep 5s

    python distributed_kmeans.py \
           --client --servers "$hostports" \
           --k 4000 --ipv4

else
    echo "unknown todo $todo"
    exit 1

fi
