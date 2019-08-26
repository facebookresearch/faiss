#! /bin/bash

set -e

todo=$1

# the training data of the Deep1B dataset
testdata=/datasets01_101/simsearch/041218/deep1b/learn.fvecs
nvec=1000000
k=4000


if [ -z "$todo" ]; then
    echo "nothing to do"
    exit 1
elif [ $todo == test_kmeans_0 ]; then
    # non distributed baseline
    python distributed_kmeans.py \
           --indata $testdata --i1 $nvec \
           --k $k

elif [ $todo == test_kmeans_1 ]; then
    # using all the machine's GPUs
    python distributed_kmeans.py \
           --indata $testdata --i1 $nvec \
           --k $k --gpu -1

elif [ $todo == test_kmeans_2 ]; then
    # distrbuted run, with one local server per GPU
    ngpu=$( echo /dev/nvidia? | wc -w )
    baseport=12012

    # kill background porcesses on output of this script
    trap 'kill -HUP 0' 0

    hostports=''

    for((gpu=0;gpu<ngpu;gpu++)); do
        # range of vectors to assign to each sever
        i0=$((nvec * gpu / ngpu))
        i1=$((nvec * (gpu + 1) / ngpu))
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
           --k $k --ipv4

elif [ $todo == slurm_distributed_kmeans ]; then

    nserv=5

    srun -n$nserv \
         --time=48:00:00 \
         --cpus-per-task=40 --gres=gpu:4 --mem=100G \
         --partition=priority --comment='priority is the only one that works'  \
         -l bash $( realpath $0 ) slurm_within_kmeans_server

elif [ $todo == slurm_within_kmeans_server ]; then

   nserv=$SLURM_NPROCS
   [ ! -z "$nserv" ] || (echo "should be run by slurm"; exit 1)
   rank=$SLURM_PROCID

   baseport=12012

   i0=$((nvec * rank / nserv))
   i1=$((nvec * (rank + 1) / nserv))
   port=$(( baseport + rank ))

   echo "host $(hostname) start server $rank for range $i0:$i1 port $port"

   if [ $rank != 0 ]; then

       python -u distributed_kmeans.py \
              --indata $testdata \
              --i0 $i0 --i1 $i1 \
              --server --gpu -1 \
              --port $port --ipv4
   else
       # master process

       # kill background processes on output of this script
       trap 'kill -HUP 0' 0

       python -u distributed_kmeans.py \
              --indata $testdata \
              --i0 $i0 --i1 $i1 \
              --server --gpu -1 \
              --port $port --ipv4 &

       hostports=""
       port=$baseport
       for hostname in $( scontrol show hostnames $SLURM_JOB_NODELIST ); do
           hostports="$hostports $hostname:$port"
           ((port++))
       done

       echo HOSTPORTS $hostports

       sleep 20s

       # run client
       python distributed_kmeans.py \
           --client --servers "$hostports" \
           --k $k --ipv4

   fi

else
    echo "unknown todo $todo"
    exit 1
fi
