import os
# data_dir = "/mnt/dataset/toago6/workspace/code/x_humanoid_value_training/lerobot_dataset_pick_place_apple_on_basket/data/chunk-000"

# max_data = 0
# min_data = 100

# for file in os.listdir(data_dir):
#     if file.endswith(".parquet"):
#         file_path = os.path.join(data_dir, file)
#         data = pd.read_parquet(file_path)
        
#         left_action = np.stack(data['action.left_arm_position'].values)
#         right_action = np.stack(data['action.right_arm_position'].values)
#         left_gripper = np.stack(data['action.left_gripper_position'].values)
#         right_gripper = np.stack(data['action.right_gripper_position'].values)
        
#         # print(left_action)
        
#         # print(np.max(left_action), np.min(left_action))
#         # print(np.max(right_action), np.min(right_action))
#         # print(np.max(left_gripper), np.min(left_gripper))
#         # print(np.max(right_gripper), np.min(right_gripper))
#         max_data = max(max_data, np.max(left_action), np.max(right_action), np.max(left_gripper), np.max(right_gripper))
#         min_data = min(min_data, np.min(left_action), np.min(right_action), np.min(left_gripper), np.min(right_gripper))
#         # break
        # print(max(right_action), min(right_action))
        # print(max(left_gripper), min(left_gripper))
        # print(max(right_gripper), min(right_gripper))
        # break
        # print(max(left_gripper.values), min(left_gripper.values))
        # print(max(right_gripper.values), min(right_gripper.values))

# print("the max data is:", max_data, "the min data is:", min_data)


# video_dir = "/mnt/dataset/toago6/workspace/code/x_humanoid_value_training/lerobot_dataset_pick_place_apple_on_basket/videos/chunk-000/observation.image.image"

# for file in os.listdir(video_dir):
#     if file.endswith(".mp4"):
#         video_path = os.path.join(video_dir, file)
#         cap = cv2.VideoCapture(video_path)
#         while cap.isOpened():
#             ret, frame = cap.read()
#             if not ret:
#                 break
#             frame = cv2.resize(frame, (224, 224))
#             cv2.imwrite(f"examples/frame_{i}.png", frame)
#             i += 1

import os


import argparse
from lerobot.common.datasets.lerobot_dataset import MultiLeRobotDataset
parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", type=str, required=True)
args = parser.parse_args()
data_dir = args.data_dir
file_names = os.listdir(data_dir)

# file_names = [
# "rtc_data_with_intervention_0424_merged",
# "rtc_data_with_intervention_0429_merged",
# "rtc_data_with_intervention_0506_merged"
# ]
# file_names = [
#             'tienkung_station_dualArm-gripper-3cameras_25_Chip_10_20260401_pm',
#             'tienkung_station_dualArm-gripper-3cameras_25_Chip_49_20260331',
#             'tienkung_station_dualArm-gripper-3cameras_25_Chip_11_20260402_AM',
#             'tienkung_station_dualArm-gripper-3cameras_24_Chip_05_20260401_AM',
#             'tienkung_station_dualArm-gripper-3cameras_24_Chip_35_20260327',
#             'tienkung_station_dualArm-gripper-3cameras_24_Chip_12_20260402_AM', 
#             'tienkung_station_dualArm-gripper-3cameras_5_Chip_45_20260330', 
#             'tienkung_station_dualArm-gripper-3cameras_5_Chip_46_20260331', 
#             'tienkung_station_dualArm-gripper-3cameras_24_Chip_17_20260402_PM', 
#             'tienkung_station_dualArm-gripper-3cameras_11_Chip_06_20260401_AM', 
#             'tienkung_station_dualArm-gripper-3cameras_11_Chip_13_20260402_AM', 
#             'tienkung_station_dualArm-gripper-3cameras_11_Chip_16_20260402_PM',
#             'tienkung_station_dualArm-gripper-3cameras_25_Chip_04_20260401_AM', 
#             'tienkung_station_dualArm-gripper-3cameras_24_Chip_48_20260331', 
#             'tienkung_station_dualArm-gripper-3cameras_25_Chip_03_20260331_pm',
#             'tienkung_station_dualArm-gripper-3cameras_11_Chip_01_20260331_pm',
#             'tienkung_station_dualArm-gripper-3cameras_25_Chip_39_20260329', 
#             'tienkung_station_dualArm-gripper-3cameras_5_Chip_02_20260331_pm',
#             'tienkung_station_dualArm-gripper-3cameras_11_Chip_09_20260401_pm', 
#             'tienkung_station_dualArm-gripper-3cameras_25_Chip_43_20260330', 
#             'tienkung_station_dualArm-gripper-3cameras_11_Chip_47_20260331'
# ]

# file_names = [
# 'tienkung_station_dualArm-gripper-3cameras_13_Electricmeter_18_20260326', 
# 'tienkung_station_dualArm-gripper-3cameras_10_Electricmeter_23_20260329', 
# 'tienkung_station_dualArm-gripper-3cameras_10_Electricmeter_25_20260330', 
# 'tienkung_station_dualArm-gripper-3cameras_07_Electricmeter_16_20260326',
# 'tienkung_station_dualArm-gripper-3cameras_10_Electricmeter_17_20260326', 
# 'tienkung_station_dualArm-gripper-3cameras_07_Electricmeter_22_20260329', 
# 'tienkung_station_dualArm-gripper-3cameras_13_Electricmeter_24_20260329'
# ]
# file_names = [
#     "tienkung_pro2_dualArm-dexHand-3cameras_215_draw_reagent_6dforce_13dtactile_checked_1_5"
# ]
# file_names = [
#                 "x_humanoid_lerobot_data_with_quality_16_01_13_processed_20260129_processed_20260202",
#                 "x_humanoid_lerobot_data_with_quality_16_01_14_processed_20260129_processed_20260202",
#                 "x_humanoid_lerobot_data_with_quality_16_01_15_processed_20260129_processed_20260202",
#                 "x_humanoid_lerobot_data_with_quality_16_01_17_processed_20260129_processed_20260202",
#                 "x_humanoid_lerobot_data_with_quality_16_01_18_processed_20260129_processed_20260202",
#                 "x_humanoid_lerobot_data_with_quality_16_01_19_processed_20260129_processed_20260202",
#                 "x_humanoid_lerobot_data_with_quality_16_01_20_processed_20260129_processed_20260202",
#                 "x_humanoid_lerobot_data_with_quality_16_01_21_processed_20260129_processed_20260202",
#                 "x_humanoid_lerobot_data_with_quality_16_01_26_processed_20260202",
#                 "x_humanoid_lerobot_data_with_quality_16_01_27_processed_20260202",
#                 "x_humanoid_lerobot_data_with_quality_16_01_28_processed_20260202",
# ]

# file_names = [
    # # "30_merged_20260128",
    #         #   "16_merged_20260128",
    #         # "41_merged_20260128"
    #         # "x_humanoid_lerobot_data_with_quality_16_01_13_processed_20260129",
    #         # "x_humanoid_lerobot_data_with_quality_16_01_14_processed_20260129",
    #         # "x_humanoid_lerobot_data_with_quality_16_01_15_processed_20260129",
    #         # "x_humanoid_lerobot_data_with_quality_16_01_17_processed_20260129",
    #         # "x_humanoid_lerobot_data_with_quality_16_01_18_processed_20260129",
    #         # "x_humanoid_lerobot_data_with_quality_16_01_19_processed_20260129",
    #         # "x_humanoid_lerobot_data_with_quality_16_01_20_processed_20260129",
    #         # "x_humanoid_lerobot_data_with_quality_16_01_21_processed_20260129",
    #         # "41_merged_20260128"
    #         "16_merged_20260129",
    #         "30_merged_20260129",
    #         "41_merged_20260129",
    #           ]

offline_dataset = MultiLeRobotDataset(
    repo_ids=file_names,
    root=data_dir,
)   
meta_data = offline_dataset._datasets[0].meta
print(meta_data.features)

print(len(offline_dataset))
for idx, data in enumerate(offline_dataset):
#     # print(data.keys())
    print(idx)
    # break
#     break
