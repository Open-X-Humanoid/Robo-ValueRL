python convert_electricmeter_hdf5_skip_corrupt_with_global_task_en_modified.py \
  --raw_data_dir /mnt/dataset/toago6/block_x_humanoid_data/jidi_robot_data/electricmeter_0331_0401/classified/medium \
  --output_dir /mnt/dataset/toago6/block_x_humanoid_data/jidi_robot_data/electricmeter_0331_0401/lerobot/medium \
  --annotation_json /mnt/dataset/toago6/block_x_humanoid_data/jidi_robot_data/electricmeter_0331_0401/task_labels \
  --translation_json /mnt/dataset/toago6/workspace/code/hierarchical_rl_in_real_world/translate_dataset_to_lerobot/electricmeter_translation_map.json \
  --max_workers 16 \
  --use_subgoal_as_task \
  --overwrite