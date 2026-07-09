import time
import numpy as np
from xrocs.common.data_type import Joints
from xrocs.core.config_loader import ConfigLoader
from xrocs.utils.logger.logger_loader import logger
from xrocs.core.station_loader import StationLoader
import cv2
import numpy as np
import torch
import cv2
import math

def decoder_image(camera_rgb_images, camera_depth_images, bgr2rgb=False):
    if type(camera_rgb_images[0]) is np.uint8:
        rgb = cv2.imdecode(camera_rgb_images, cv2.IMREAD_COLOR)
        if bgr2rgb and rgb is not None:
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
        if camera_depth_images is not None:
            depth_array = np.frombuffer(camera_depth_images, dtype=np.uint8)
            depth = cv2.imdecode(depth_array, cv2.IMREAD_UNCHANGED)
        else:
            depth = np.asarray([])
        return rgb, depth
    else:
        rgb_images = []
        depth_images = []
        # print(f"camera_rgb_images: {camera_rgb_images}")
        for idx, camera_rgb_image in enumerate(camera_rgb_images):
            # print(f"camera_rgb_image: {camera_rgb_image}")
            rgb = cv2.imdecode(camera_rgb_image, cv2.IMREAD_COLOR)
            if camera_depth_images is not None:
                depth_array = np.frombuffer(camera_depth_images[idx], dtype=np.uint8)
                depth = cv2.imdecode(depth_array, cv2.IMREAD_UNCHANGED)
            else:
                depth = np.asarray([])
            
            if bgr2rgb and rgb is not None:
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            rgb_images.append(rgb)
            depth_images.append(depth)
        rgb_images = np.asarray(rgb_images)
        depth_images = np.asarray(depth_images)
        return rgb_images, depth_images

class RealWorld_X_Humanoid_Env:
    def __init__(self, real_world_config_path = None, task_name = "Close the toolbox.", init_pose = None):
        if real_world_config_path == None:
            real_world_config_path = "./configuration.toml"
        cfg_loader = ConfigLoader(real_world_config_path)
        self.cfg_dict = cfg_loader.get_config()
        self.init_pose = init_pose
        if init_pose is not None:
            self.cfg_dict['robot']['arm']['home']['robot'] = init_pose
        station_loader = StationLoader(self.cfg_dict)
        
        self.robot_station = station_loader.generate_station_handle()
        self.robot_station.connect()
        self._traj_buffer = []
        self.task_name = task_name


        self.left_limits_rad = [(math.radians(-170.0), math.radians(170.0)),
                            (math.radians(-15.0), math.radians(150.0)),
                            (math.radians(-170.0), math.radians(170.0)),
                            (math.radians(-150.0), math.radians(15.0)),
                            (math.radians(-170.0), math.radians(170.0)),
                            (math.radians(-45.0), math.radians(60.0)),
                            (math.radians(-95.0), math.radians(75.0))]
        self.right_limits_rad =  [(math.radians(-170.0), math.radians(170.0)),
                            (math.radians(-150.0), math.radians(15.0)),
                            (math.radians(-170.0), math.radians(170.0)),
                            (math.radians(-150.0), math.radians(15.0)),
                            (math.radians(-170.0), math.radians(170.0)),
                            (math.radians(-45.0), math.radians(60.0)),
                            (math.radians(-75.0), math.radians(95.0))]

    def prepare(self):
        for name, _robot in self.robot_station.get_robot_handle().items():
            home = Joints(self.cfg_dict['robot']['arm']['home'][name],
                        num_of_dofs=len((self.cfg_dict['robot']['arm']['home'][name])))
            _robot.reach_target_joint(home)
        for gripper in self.robot_station.get_gripper_handle().values():
            gripper.open()
        time.sleep(5)
        logger.success('Resetting to home success!')

    def reset(self):
        for name, _robot in self.robot_station.get_robot_handle().items():
            home = Joints(self.cfg_dict['robot']['arm']['home'][name],
                        num_of_dofs=len((self.cfg_dict['robot']['arm']['home'][name])))
            _robot.reach_target_joint(home)
        for gripper in self.robot_station.get_gripper_handle().values():
            gripper.open()
        time.sleep(2)
        logger.success('Resetting to home success!')

    def post_process_obs(self, obs):
        # import pdb; pdb.set_trace()
        # print(f"obs: {obs}")
        left_state_joint_pos = obs['arm_joints']['left']
        left_hand_joint = obs['hand_joints']['left']
        right_state_joint_pos = obs['arm_joints']['right']
        right_hand_joint = obs['hand_joints']['right']
        rgb_image, _ = decoder_image(obs['images']['top'], None, bgr2rgb=True)
        left_image, _ = decoder_image(obs['images']['left'], None, bgr2rgb=True)
        right_image, _ = decoder_image(obs['images']['right'], None, bgr2rgb=True)
        cv2.imwrite("rl_envs_observation.jpg", rgb_image)
        # print("the rgb_image shape is:", rgb_image.shape)
        # print("the rgb_image shape is:", rgb_image)
        # cv2.imwrite("rl_envs_observation.jpg", rgb_image)
        total_state = np.concatenate([left_state_joint_pos, left_hand_joint, right_state_joint_pos, right_hand_joint], axis=-1)
        total_state = np.expand_dims(total_state, axis=0)
        head_frame_i = np.expand_dims(rgb_image, axis=0)
        left_frame_i = np.expand_dims(left_image, axis=0)
        right_frame_i = np.expand_dims(right_image, axis=0)
        batch = {
            "image": {
                "base_0_rgb": head_frame_i,
                "left_wrist_0_rgb": left_frame_i,
                "right_wrist_0_rgb": right_frame_i,
            },
            "state": total_state,
            "prompt": self.task_name,
            "image_mask": {
                "base_0_rgb": torch.tensor([True]),
                "left_wrist_0_rgb": torch.tensor([True]),
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "right_wrist_0_rgb": torch.tensor([True]),
            }
        }
        
        return batch
    
    def get_obs(self):
        obs = self.robot_station.get_obs()
        return self.post_process_obs(obs)
        # return obs


    def step(self, action):
        # Check the dual arm status. By Harvey Liu
        robot = self.robot_station._robot_dict.get("robot")
        dual_arm_status = robot.dual_arm_ctrler.get_status()
        if dual_arm_status is None:
            logger.error("Failed to get dual arm status. Please check the ROS 2 connection (unable to get the dual arm's joint status; please check the hardware or ROS 2 communication).")
            return False
        for name, status in dual_arm_status.items():
            if status["error"] != 0:
                logger.error(f"Dual arm joint {name} error: {status['error']}. Please check the dual arm. (Arm {name} joint error, error code is {status['error']}, please inspect the hardware according to the error message). Error code reference: 33072-device offline, 33073-joint position out of range, 1-joint motor overheating, 2-overcurrent, 3-undervoltage, 4-joint MOS overheating, 5-stall, 6-overvoltage, 7-phase loss, 8-encoder error")
                return False



        robot_targets = self.robot_station.decompose_action(action)
        # print(f"robot_targets: {robot_targets}")
        left_arm = robot_targets['arm']['position']['left']
        right_arm = robot_targets['arm']['position']['right']
        for i in range(len(left_arm)):
            if left_arm[i] < self.left_limits_rad[i][0] or left_arm[i] > self.left_limits_rad[i][1]:
                left_arm[i] = np.clip(left_arm[i], self.left_limits_rad[i][0], self.left_limits_rad[i][1])
        for i in range(len(right_arm)):
            if right_arm[i] < self.right_limits_rad[i][0] or right_arm[i] > self.right_limits_rad[i][1]:
                right_arm[i] = np.clip(right_arm[i], self.right_limits_rad[i][0], self.right_limits_rad[i][1])
                
        # import pdb; pdb.set_trace()
        obs = self.robot_station.step(robot_targets)
        return self.post_process_obs(obs)
