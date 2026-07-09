import pandas as pd
import numpy as np
import os
# def change_reward(dataset_path):
#     dataset = pd.read_parquet(dataset_path)
#     dataset['reward'] = dataset['reward'].apply(lambda x: x if x > 0 else 0)
#     dataset.to_parquet(dataset_path)

if __name__ == "__main__":
    data_dir = "/mnt/dataset/toago6/workspace/code/rl_block_x_humanoid_data/lerobot_version/x_humanoid_lerobot_data_without_discount_24/data/chunk-000"
    for data_name in os.listdir(data_dir):
        data_path = os.path.join(data_dir, data_name)
        if data_name.endswith(".parquet"):
            state_data = pd.read_parquet(data_path)
            discount_value_return = state_data['discounted_value_return']
            len_value_return = len(discount_value_return)
            value_return = np.ones((len_value_return,1 )) * (-0.1)
            
            if discount_value_return[len_value_return-1] >= 0:
                value_return[-1:] = 100
            else:
                value_return[-1:] = -600
            discount = 1
            
            discounted_return = np.zeros_like(value_return)
            discounted_return[-1] = value_return[-1]
            for i in range(len_value_return-2, -1, -1):
                discounted_return[i] = value_return[i] + discount * discounted_return[i+1]
            
            value_return = value_return.astype(np.float32)
            discounted_return = discounted_return.astype(np.float32)
            y_pressed = state_data['y_pressed']
            y_pressed_cumsum = np.cumsum(y_pressed)
            for i in range(len_value_return):
                discounted_return[i] = discounted_return[i] + y_pressed_cumsum[i]
            
            state_data['value_reward'] = value_return
            state_data['discounted_value_return'] = discounted_return
            # state_data = embed_images(state_data)
            state_data.to_parquet(data_path)
        