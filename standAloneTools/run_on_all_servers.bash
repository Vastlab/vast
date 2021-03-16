#!/bin/bash

SSHNAME=$1

servers=(
    "patriot"
    "pepper"
)
for i in "${servers[@]}"; do
    echo "Running on $i"
    scp cuda_nccl_cudnn_setup.bash $SSHNAME@$i.vast.uccs.edu:/tmp/
    ssh -t $SSHNAME@$i.vast.uccs.edu "sudo bash /tmp/cuda_nccl_cudnn_setup.bash"
done
