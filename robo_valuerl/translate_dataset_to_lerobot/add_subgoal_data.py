import os
import pandas as pd
import numpy as np
if __name__ == "__main__":
    data_dir = "/mnt/dataset/toago6/block_x_humanoid_data/lerobot_version/x_humanoid_lerobot_data_without_discount_23/data/chunk-000"
    annotation_dir = "/mnt/dataset/toago6/block_x_humanoid_data/annotation/annotation_20"
    for file in os.listdir(data_dir):
        if file.endswith(".parquet"):
            file_path = os.path.join(data_dir, file)
            data = pd.read_parquet(file_path)
            
            numpy_path = os.path.join(annotation_dir, file.replace(".parquet", ".npy"))
            subgoal_data = np.load(numpy_path)
            subgoal_reward = subgoal_data * 50
            subgoal_reward = np.cumsum(subgoal_reward)
            
            discounted_value_return = data['discounted_value_return']
            
            data_len = len(discounted_value_return)
            for i in range(data_len):
                discounted_value_return[i] = subgoal_reward[i] + discounted_value_return[i]
            
            data['discounted_value_return'] = discounted_value_return
            data.to_parquet(file_path)
