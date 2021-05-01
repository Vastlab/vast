architectures=('alexnet' 'densenet121' 'densenet161' 'densenet169' 'densenet201' 'googlenet' 'inception_v3' 'mnasnet0_5' 'mnasnet0_75' 'mnasnet1_0' 'mnasnet1_3' 'mobilenet_v2' 'resnet101' 'resnet152' 'resnet18' 'resnet34' 'resnet50' 'resnext101_32x8d' 'resnext50_32x4d' 'shufflenet_v2_x0_5' 'shufflenet_v2_x1_0' 'shufflenet_v2_x1_5' 'shufflenet_v2_x2_0' 'squeezenet1_0' 'squeezenet1_1' 'vgg11' 'vgg11_bn' 'vgg13' 'vgg13_bn' 'vgg16' 'vgg16_bn' 'vgg19' 'vgg19_bn' 'wide_resnet101_2' 'wide_resnet50_2')
#dataset_types=('val_in_folders' 'train')
dataset_types=('val_in_folders')
datasets=('ILSVRC_2012' '360_openset')
dataset_root="/scratch/datasets/ImageNet"
output_dir="/net/reddwarf/bigscratch/adhamija/Features/"
no_of_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
all_running_PIDS=()
exp_no=0

for architecture in "${architectures[@]}"; do
  for dataset in "${datasets[@]}"; do
    for dataset_type in "${dataset_types[@]}"; do
      echo -e "Starting\t $architecture\t $dataset_type \t$dataset"
      output_dir_path="${output_dir}/${architecture}/${dataset}/"
      mkdir -p $output_dir_path
      output_file_path="${output_dir_path}/${dataset_type}.hdf5"
      images_path="${dataset_root}/${dataset}/${dataset_type}"
      set -o xtrace
      PID=$(nohup sh -c "CUDA_VISIBLE_DEVICES=$exp_no python FeatureExtraction.py --arch $architecture \
      --output-path $output_file_path --dataset-path $images_path" >/dev/null 2>&1 & echo $!)
      set +o xtrace
      echo "Started PID $PID"
      all_running_PIDS[$exp_no]=$PID
      if [ ${#all_running_PIDS[@]} -eq $no_of_gpus ]
      then
        echo "Waiting for a free GPU"
        while true; do
          running_python_processes=$(pgrep sh)
          for started_PID_indx in "${!all_running_PIDS[@]}"; do
            if ! [[ "${running_python_processes[@]}" =~ "${all_running_PIDS[$started_PID_indx]}" ]]; then
              echo "Process ID ${all_running_PIDS[$started_PID_indx]} is not running, GPU ${started_PID_indx} free"
              exp_no=started_PID_indx-1
              break 2
            fi
          done
          echo "...............Sleeping..............."
          sleep 30s
          echo "...............Awake..............."
        done
      fi
      ((exp_no+=1))
    done
  done
done
