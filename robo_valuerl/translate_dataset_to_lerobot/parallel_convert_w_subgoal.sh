python convert_electricmeter_hdf5_skip_corrupt_with_global_task_en_patched.py \
  --raw_data_dir /mnt/dataset/toago6/block_x_humanoid_data/jidi_robot_data/electric_sub/hdf5 \
  --output_dir /mnt/dataset/toago6/block_x_humanoid_data/jidi_robot_data/electric_sub/lerobot_newnew \
  --annotation_json /mnt/dataset/toago6/block_x_humanoid_data/jidi_robot_data/electric_sub/tasklabels \
  --use_subgoal_as_task \
  --max_workers 30 \
  --overwrite