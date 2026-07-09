"""Simplified Gym Interface for UR Robot Joint Control"""
import numpy as np
import cv2
import time
import gymnasium as gym

from xrocs.core.config_loader import ConfigLoader
from xrocs.core.station_loader import StationLoader
from xrocs.common.data_type import Joints
from xrocs.utils.logger.logger_loader import logger
def decoder_image(camera_rgb_images, camera_depth_images, bgr2rgb=False):
    """Decode camera images"""
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


class BaseEnv(gym.Env):
    """Simplified UR robot Gym environment, focused on joint control"""

    def __init__(self, config=None, fake_env=False):
        print('Initializing BaseEnv for UR robot joint control')

        self.config = config
        self.fake_env = fake_env
        self.joint_dim = 6  # UR robot has 6 joints
        self.gripper_dim = 1  # 1 gripper

        # Action space: 6 joint angles + 1 gripper
        self.action_space = gym.spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.joint_dim + self.gripper_dim,),
            dtype=np.float32
        )

        # Observation space: joint state + image
        self.observation_space = gym.spaces.Dict({
            "state": gym.spaces.Dict({
                "joints": gym.spaces.Box(-np.inf, np.inf, shape=(self.joint_dim,)),
                "gripper": gym.spaces.Box(0, 1, shape=(self.gripper_dim,)),
            }),
            "images": gym.spaces.Dict({
                "front": gym.spaces.Box(0, 255, shape=(480, 640, 3), dtype=np.uint8),
                "wrist": gym.spaces.Box(0, 255, shape=(480, 640, 3), dtype=np.uint8),
            })
        })
        
        if fake_env:
            print("Fake environment mode - not connecting to robot")
            return
            
        # Initialize the robot connection
        config_path = getattr(config, 'config_path', "/home/eai/Documents/configuration.toml")
        cfg_loader = ConfigLoader(config_path)
        self.cfg_dict = cfg_loader.get_config()
        station_loader = StationLoader(self.cfg_dict)
        self.robot_station = station_loader.generate_station_handle()
        self.robot_station.connect()
        logger.success('Robot station connected!')

        # Reset the counter
        self.episode_length = 0
        self.max_episode_length = getattr(config, 'max_episode_length', 1000)

    def step(self, action):
        """Execute one step of action"""
        if self.fake_env:
            return self._get_fake_obs(), 0.0, False, False, {}

        # Parse the action: first 6 dims are joint angles, last dim is the gripper
        joint_action = action[:self.joint_dim]
        gripper_action = np.clip(action[self.joint_dim], 0, 1)

        # Build the action target
        action_target = {
            "arm_joints": {"single": joint_action},
            "hand_joints": {"single": gripper_action}
        }

        # Execute the action
        obs = self.robot_station.step(action_target)

        # Update the counter
        self.episode_length += 1

        # Get the observation
        observation = self._get_obs(obs)

        # Simple reward and termination condition
        reward = 0.0
        terminated = False
        truncated = self.episode_length >= self.max_episode_length
        
        info = {
            "episode_length": self.episode_length,
            "joint_action": joint_action,
            "gripper_action": gripper_action
        }
        
        return observation, reward, terminated, truncated, info
    
    def reset(self, **kwargs):
        """Reset the environment"""
        if self.fake_env:
            return self._get_fake_obs(), {}

        # Reset the counter
        self.episode_length = 0

        # Reset to the home position
        self._reset_to_home()

        # Get the initial observation
        obs = self.robot_station.get_obs()
        observation = self._get_obs(obs)

        return observation, {"success": True}

    def _reset_to_home(self):
        """Reset the robot to the home position"""
        for name, _robot in self.robot_station.get_robot_handle().items():
            home = Joints(
                self.cfg_dict['robot']['arm']['home'][name],
                num_of_dofs=len(self.cfg_dict['robot']['arm']['home'][name])
            )
            _robot.reach_target_joint(home)
        
        for gripper in self.robot_station.get_gripper_handle().values():
            gripper.open()
        
        time.sleep(2)
        logger.success('Resetting to home success!')
    
    def _get_obs(self, obs=None):
        """Get the observation"""
        if obs is None:
            obs = self.robot_station.get_obs()

        # Extract state information
        joint_state = np.array(obs['arm_joints']['single'])
        gripper_state = np.array(obs['hand_joints']['single']).reshape(1)

        # Decode images
        images = {}
        for key in ['front', 'wrist']:
            if key in obs['images']:
                rgb, _ = decoder_image(obs['images'][key], None, bgr2rgb=True)
                images[key] = rgb
            else:
                # If this camera is not available, create a blank image
                images[key] = np.zeros((480, 640, 3), dtype=np.uint8)
        
        return {
            "state": {
                "joints": joint_state,
                "gripper": gripper_state,
            },
            "images": images
        }
    
    def _get_fake_obs(self):
        """Get a fake observation (used in fake_env mode)"""
        return {
            "state": {
                "joints": np.zeros(self.joint_dim),
                "gripper": np.array([0.0]),
            },
            "images": {
                "front": np.zeros((480, 640, 3), dtype=np.uint8),
                "wrist": np.zeros((480, 640, 3), dtype=np.uint8),
            }
        }
    
    def close(self):
        """Close the environment"""
        if not self.fake_env and hasattr(self, 'robot_station'):
            # Disconnect code can be added here
            pass
        return


