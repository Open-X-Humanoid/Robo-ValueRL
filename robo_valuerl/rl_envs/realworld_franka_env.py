import time
import numpy as np
from xrocs.common.data_type import Joints
from xrocs.core.config_loader import ConfigLoader
from xrocs.utils.logger.logger_loader import logger
from xrocs.core.station_loader import StationLoader
import cv2
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
        for idx, camera_rgb_image in enumerate(camera_rgb_images):
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


class RealWorld_Franka_Env:
    def __init__(self, real_world_config_path=None, task_name="Close the toolbox.", init_pose=None):
        if real_world_config_path is None:
            real_world_config_path = "./franka_configuration.toml"
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

        self.joint_limits_rad = [
            (-2.8973, 2.8973),   # joint1
            (-1.7628, 1.7628),   # joint2
            (-2.8973, 2.8973),   # joint3
            (-3.0718, -0.0698),  # joint4
            (-2.8973, 2.8973),   # joint5
            (-0.0175, 3.7525),   # joint6
            (-2.8973, 2.8973),   # joint7
        ]

    def prepare(self):
        for name, _robot in self.robot_station.get_robot_handle().items():
            home = Joints(self.cfg_dict['robot']['arm']['home'][name],
                          num_of_dofs=len(self.cfg_dict['robot']['arm']['home'][name]))
            _robot.reach_target_joint(home)
        for gripper in self.robot_station.get_gripper_handle().values():
            gripper.open()
        time.sleep(5)
        logger.success('Resetting to home success!')

    def reset(self):
        for name, _robot in self.robot_station.get_robot_handle().items():
            home = Joints(self.cfg_dict['robot']['arm']['home'][name],
                          num_of_dofs=len(self.cfg_dict['robot']['arm']['home'][name]))
            _robot.reach_target_joint(home)
        for gripper in self.robot_station.get_gripper_handle().values():
            gripper.open()
        time.sleep(2)
        logger.success('Resetting to home success!')

    def post_process_obs(self, obs):
        arm_joint_pos = obs['arm_joints']['single']  # (8,): 7 arm + 1 gripper
        hand_joint = np.atleast_1d(np.asarray(obs['hand_joints']['single'], dtype=np.float32))  # (1,)
        ee = np.asarray(obs['arm_pose']['single'], dtype=np.float32)[:7]

        left_image, _ = decoder_image(obs['images']['left'], None, bgr2rgb=True)
        right_image, _ = decoder_image(obs['images']['right'], None, bgr2rgb=True)
        wrist_image, _ = decoder_image(obs['images']['wrist'], None, bgr2rgb=True)

        # Aligned with training: state = arm(8) + hand(1) + ee(7) = 16d
        total_state = np.concatenate([arm_joint_pos, hand_joint, ee], axis=-1).astype(np.float32)  # (16,)
        total_state = np.expand_dims(total_state, axis=0)  # (1, 16)

        # Key names aligned with the training wrapper (WenboFrankaLerobotOfflineWrapper):
        #   camera_left  → base_0_rgb
        #   camera_right → left_wrist_0_rgb
        #   camera_wrist → right_wrist_0_rgb
        batch = {
            "image": {
                "base_0_rgb": np.expand_dims(left_image, axis=0),
                "left_wrist_0_rgb": np.expand_dims(right_image, axis=0),
                "right_wrist_0_rgb": np.expand_dims(wrist_image, axis=0),
            },
            "state": total_state,  # shape (1, 16)
            "prompt": self.task_name,
            "image_mask": {
                "base_0_rgb": torch.tensor([True]),
                "left_wrist_0_rgb": torch.tensor([True]),
                "right_wrist_0_rgb": torch.tensor([True]),
            }
        }

        return batch

    def get_obs(self):
        # import pdb; pdb.set_trace()
        obs = self.robot_station.get_obs()
        # import pdb; pdb.set_trace()
        return self.post_process_obs(obs)

    def step(self, action):
        arm_joints = action[0:7].copy()
        gripper = 1.0  # always closed

        # Clip joints to Franka joint limits
        for i in range(7):
            arm_joints[i] = np.clip(arm_joints[i], self.joint_limits_rad[i][0], self.joint_limits_rad[i][1])

        robot_target = {
            "arm": {
                "position": {
                    "single": np.append(arm_joints, gripper),
                }
            }
        }
        obs = self.robot_station.step(robot_target)
        return self.post_process_obs(obs)
