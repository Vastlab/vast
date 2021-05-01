architectures=('deit_tiny_patch16_224' 'deit_small_patch16_224' 'deit_base_patch16_224' 'deit_tiny_distilled_patch16_224'
                'deit_small_distilled_patch16_224' 'deit_base_distilled_patch16_224' 'deit_base_patch16_384'
                 'deit_base_distilled_patch16_384')
dataset_types=('val_in_folders' 'train')
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
      PID=$(nohup sh -c "CUDA_VISIBLE_DEVICES=$exp_no python FeatureExtraction.py --DeiT_model $architecture \
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
          sleep 3m
          echo "...............Awake..............."
        done
      fi
      ((exp_no+=1))
    done
  done
done
