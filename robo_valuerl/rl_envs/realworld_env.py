import time
import numpy as np
from xrocs.common.data_type import Joints
from xrocs.core.config_loader import ConfigLoader
from xrocs.utils.logger.logger_loader import logger
from xrocs.core.station_loader import StationLoader
import cv2
import numpy as np
import torch

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

class RealWorldEnv:
    def __init__(self, real_world_config_path = None, agent = None, task_name = "Close the toolbox."):
        if real_world_config_path == None:
            real_world_config_path = "./configuration.toml"
        cfg_loader = ConfigLoader(real_world_config_path)
        self.cfg_dict = cfg_loader.get_config()
        station_loader = StationLoader(self.cfg_dict)
        self.robot_station = station_loader.generate_station_handle()
        self.robot_station.connect()
        self._traj_buffer = []
        self.task_name = task_name


    def prepare(self):
        for name, _robot in self.robot_station.get_robot_handle().items():
            home = Joints(self.cfg_dict['robot']['arm']['home'][name],
                          num_of_dofs=len((self.cfg_dict['robot']['arm']['home'][name])))
            _robot.reach_target_joint(home)
        for gripper in self.robot_station.get_gripper_handle().values():
            gripper.open()
        time.sleep(2)
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
        state_joint_pos = obs['arm_joints']['single']
        hand_joint = obs['hand_joints']['single']
        rgb_image, _ = decoder_image(obs['images']['front'], None, bgr2rgb=True)
        wrist_image, _ = decoder_image(obs['images']['wrist'], None, bgr2rgb=True)
        total_state = np.concatenate([state_joint_pos, hand_joint], axis=-1)
        total_state = np.expand_dims(total_state, axis=0)
        front_frame_i = np.expand_dims(rgb_image, axis=0)
        wrist_frame_i = np.expand_dims(wrist_image, axis=0)
        batch = {
            "image": {
                "base_0_rgb": front_frame_i,
                "left_wrist_0_rgb": wrist_frame_i,
                "right_wrist_0_rgb": np.zeros_like(front_frame_i),
            },
            "state": total_state,
            "prompt": self.task_name,
            "image_mask": {
                "base_0_rgb": torch.tensor([True]),
                "left_wrist_0_rgb": torch.tensor([True]),
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "right_wrist_0_rgb": torch.tensor([False]),
            }
        }
        
        return batch
    
    def get_obs(self):
        obs = self.robot_station.get_obs()
        return self.post_process_obs(obs)


    def step(self, action):
        robot_targets = self.robot_station.decompose_action(action)
        obs = self.robot_station.step(robot_targets)
        return self.post_process_obs(obs)
