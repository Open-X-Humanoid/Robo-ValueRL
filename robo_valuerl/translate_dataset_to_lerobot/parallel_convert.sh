# python convert_hdf5_lerobot_skipbad.py \
#   --raw_data_dir /mnt/dataset/toago6/block_x_humanoid_data/jidi_robot_data/chip_0329__/h5 \
#   --output_dir /mnt/dataset/toago6/block_x_humanoid_data/jidi_robot_data/chip_0329__/le \
#   --max_workers 30 \
#   --overwrite

python hdf5_to_lerobot_with_meta_task_name.py \
  --raw_data_dir /mnt/dataset/toago6/block_x_humanoid_data/jidi_robot_data/chip_0329__/h5 \
  --output_dir /mnt/dataset/toago6/block_x_humanoid_data/jidi_robot_data/chip_0329__/le \
  --meta_task_name "insert the chip" \
  --max_workers 30 \
  --overwrite