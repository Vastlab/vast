servers=(
    "patriot"
    "pepper"
)
for i in "${servers[@]}"; do
    echo "Running on $i"
    scp cuda_nccl_cudnn_setup.bash adhamija@$i.vast.uccs.edu:/tmp/
    ssh -t adhamija@$i.vast.uccs.edu "sudo bash /tmp/cuda_nccl_cudnn_setup.bash"
done

