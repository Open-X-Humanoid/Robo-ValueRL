import dataclasses

import einops
import numpy as np

from openpi import transforms
from openpi.models import model as _model
import torch

from openpi.models.hl_gauss_distribution import HLGaussDistribution, DynamicHLGauss

def map_discounted_value_return_to_label(discounted_value_return, num_bin = 256):
    
    discounted_value_return = torch.clamp(discounted_value_return, -1, 0)
    label = (-discounted_value_return * num_bin).to(torch.int64).clamp(0, num_bin - 1)
    # to one hot encoding
    one_hot = torch.zeros(num_bin)
    one_hot[label] = 1
    return one_hot


def _parse_image(image) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:
        image = einops.rearrange(image, "c h w -> h w c")
    return image



@dataclasses.dataclass(frozen=True)
class X_Humanoid_Inputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """
    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        base_image = _parse_image(data["observation.image.image"])
        # wrist_image = _parse_image(data["observation.rgb_images.camera_wrist"])

        puppet_left_arm_position = data['observation.state.puppet_left_arm_position']
        puppet_right_arm_position = data['observation.state.puppet_right_arm_position']
        puppet_left_gripper_position = data['observation.state.puppet_left_gripper_position'].unsqueeze(0)
        puppet_right_gripper_position = data['observation.state.puppet_right_gripper_position'].unsqueeze(0)

        state = torch.cat([puppet_left_arm_position, puppet_left_gripper_position, puppet_right_arm_position, puppet_right_gripper_position], dim=-1)
        
        master_left_arm_position = data['observation.state.master_left_arm_position']
        master_right_arm_position = data['observation.state.master_right_arm_position']
        master_left_gripper_position = data['observation.state.master_left_gripper_position'].unsqueeze(-1)
        master_right_gripper_position = data['observation.state.master_right_gripper_position'].unsqueeze(-1)

        master_state = torch.cat([master_left_arm_position, master_left_gripper_position, master_right_arm_position, master_right_gripper_position], dim=-1)
        
        action_left_arm_position = data['action.left_arm_position']
        action_right_arm_position = data['action.right_arm_position']
        action_left_gripper_position = data['action.left_gripper_position'].unsqueeze(-1)
        action_right_gripper_position = data['action.right_gripper_position'].unsqueeze(-1)
        # print("action_left_arm_position: ", action_left_arm_position.shape)
        # print("action_left_gripper_position: ", action_left_gripper_position.shape)
        # print("action_right_arm_position: ", action_right_arm_position.shape)
        # print("action_right_gripper_position: ", action_right_gripper_position.shape)
        action = torch.cat([action_left_arm_position, action_left_gripper_position, action_right_arm_position, action_right_gripper_position], dim=-1)

        # print("the data.keys() is:", data.keys())
        if "discounted_value_return" in data.keys():
            value_gt = data['discounted_value_return']
            value_gt = (value_gt - 700) / (1200 + 700)
            value_gt = map_discounted_value_return_to_label(value_gt, num_bin=512)
            # value_gt = value_gt.numpy()
            # print("the value_gt is:", value_gt.shape)
            value_gt = value_gt.numpy()
        else:
            value_gt = None

        if "action_judgement" in data.keys():
            action_judgement = data['action_judgement']
            action_judgement = action_judgement.numpy()
        else:
            action_judgement = None

        # print("action: ", action.shape)
        # print("master_state: ", master_state.shape)
        # print("action: ", action.shape)

        state = state.numpy()
        master_state = master_state.numpy()
        # action = action.numpy()
        action = master_state
        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": state,
            # "master_state": master_state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": np.zeros_like(base_image),
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "right_wrist_0_rgb": np.False_,
            },
            "value_gt": value_gt,   
            "action_judgement": action_judgement,
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        inputs["actions"] = action

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]


        return inputs



@dataclasses.dataclass(frozen=True)
class X_Humanoid_Multi_Camera_Inputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """
    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        base_image = _parse_image(data["observation.image.image"])
        left_wrist_image = _parse_image(data["observation.image.left"])
        right_wrist_image = _parse_image(data["observation.image.right"])
        # wrist_image = _parse_image(data["observation.rgb_images.camera_wrist"])

        puppet_left_arm_position = data['observation.state.puppet_left_arm_position']
        puppet_right_arm_position = data['observation.state.puppet_right_arm_position']
        puppet_left_gripper_position = data['observation.state.puppet_left_gripper_position'].unsqueeze(0)
        puppet_right_gripper_position = data['observation.state.puppet_right_gripper_position'].unsqueeze(0)

        state = torch.cat([puppet_left_arm_position, puppet_left_gripper_position, puppet_right_arm_position, puppet_right_gripper_position], dim=-1)
        
        master_left_arm_position = data['observation.state.master_left_arm_position']
        master_right_arm_position = data['observation.state.master_right_arm_position']
        master_left_gripper_position = data['observation.state.master_left_gripper_position'].unsqueeze(-1)
        master_right_gripper_position = data['observation.state.master_right_gripper_position'].unsqueeze(-1)

        master_state = torch.cat([master_left_arm_position, master_left_gripper_position, master_right_arm_position, master_right_gripper_position], dim=-1)
        
        action_left_arm_position = data['action.left_arm_position']
        action_right_arm_position = data['action.right_arm_position']
        action_left_gripper_position = data['action.left_gripper_position'].unsqueeze(-1)
        action_right_gripper_position = data['action.right_gripper_position'].unsqueeze(-1)
        # print("action_left_arm_position: ", action_left_arm_position.shape)
        # print("action_left_gripper_position: ", action_left_gripper_position.shape)
        # print("action_right_arm_position: ", action_right_arm_position.shape)
        # print("action_right_gripper_position: ", action_right_gripper_position.shape)
        action = torch.cat([action_left_arm_position, action_left_gripper_position, action_right_arm_position, action_right_gripper_position], dim=-1)

        # print("the data.keys() is:", data.keys())
        if "discounted_value_return" in data.keys():
            value_gt = data['discounted_value_return']
            value_gt = (value_gt - 700) / (1200 + 700)
            value_gt = map_discounted_value_return_to_label(value_gt, num_bin=512)
            # value_gt = value_gt.numpy()
            # print("the value_gt is:", value_gt.shape)
            value_gt = value_gt.numpy()
        else:
            value_gt = None

        if "action_judgement" in data.keys():
            action_judgement = data['action_judgement']
            action_judgement = action_judgement.numpy()
        else:
            action_judgement = None

        # print("action: ", action.shape)
        # print("master_state: ", master_state.shape)
        # print("action: ", action.shape)

        state = state.numpy()
        master_state = master_state.numpy()
        # action = action.numpy()
        action = master_state
        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": state,
            # "master_state": master_state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "right_wrist_0_rgb": np.True_,
            },
            "value_gt": value_gt,   
            "action_judgement": action_judgement,
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        inputs["actions"] = action

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]


        return inputs


@dataclasses.dataclass(frozen=True)
class X_Humanoid_hl_gauss_Inputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """
    num_value_bins = 512
    sigma = 3.0
    bins = torch.linspace(-1200, 700, num_value_bins)
    hl_gauss_distribution = HLGaussDistribution(bins, sigma)
    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        base_image = _parse_image(data["observation.image.image"])
        # wrist_image = _parse_image(data["observation.rgb_images.camera_wrist"])

        puppet_left_arm_position = data['observation.state.puppet_left_arm_position']
        puppet_right_arm_position = data['observation.state.puppet_right_arm_position']
        puppet_left_gripper_position = data['observation.state.puppet_left_gripper_position'].unsqueeze(0)
        puppet_right_gripper_position = data['observation.state.puppet_right_gripper_position'].unsqueeze(0)

        state = torch.cat([puppet_left_arm_position, puppet_left_gripper_position, puppet_right_arm_position, puppet_right_gripper_position], dim=-1)
        
        master_left_arm_position = data['observation.state.master_left_arm_position']
        master_right_arm_position = data['observation.state.master_right_arm_position']
        master_left_gripper_position = data['observation.state.master_left_gripper_position'].unsqueeze(-1)
        master_right_gripper_position = data['observation.state.master_right_gripper_position'].unsqueeze(-1)

        master_state = torch.cat([master_left_arm_position, master_left_gripper_position, master_right_arm_position, master_right_gripper_position], dim=-1)
        
        action_left_arm_position = data['action.left_arm_position']
        action_right_arm_position = data['action.right_arm_position']
        action_left_gripper_position = data['action.left_gripper_position'].unsqueeze(-1)
        action_right_gripper_position = data['action.right_gripper_position'].unsqueeze(-1)
        # print("action_left_arm_position: ", action_left_arm_position.shape)
        # print("action_left_gripper_position: ", action_left_gripper_position.shape)
        # print("action_right_arm_position: ", action_right_arm_position.shape)
        # print("action_right_gripper_position: ", action_right_gripper_position.shape)
        action = torch.cat([action_left_arm_position, action_left_gripper_position, action_right_arm_position, action_right_gripper_position], dim=-1)

        # print("the data.keys() is:", data.keys())
        if "discounted_value_return" in data.keys():
            value_gt = data['discounted_value_return']
            # value_gt = value_gt / 300
            # value_gt = map_discounted_value_return_to_label(value_gt)
            # value_gt = value_gt.numpy()
            value_gt = self.hl_gauss_distribution.encode(value_gt)
            # print("the value_gt is:", value_gt.shape)
            value_gt = value_gt.numpy()
        else:
            value_gt = None

        if "action_judgement" in data.keys():
            action_judgement = data['action_judgement']
            action_judgement = action_judgement.numpy()
        else:
            action_judgement = None

        # print("action: ", action.shape)
        # print("master_state: ", master_state.shape)
        # print("action: ", action.shape)

        state = state.numpy()
        master_state = master_state.numpy()
        # action = action.numpy()
        action = master_state
        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": state,
            # "master_state": master_state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": np.zeros_like(base_image),
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "right_wrist_0_rgb": np.False_,
            },
            "value_gt": value_gt,   
            "action_judgement": action_judgement,
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        inputs["actions"] = action

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]


        return inputs



@dataclasses.dataclass(frozen=True)
class X_Humanoid_hl_gauss_RTG_Inputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """
    num_value_bins = 256
    sigma = 3.0
    bins = torch.linspace(0, 400, num_value_bins)
    discounted_hl_gauss_distribution = HLGaussDistribution(bins, sigma)
    time_bins = torch.linspace(0, 5000, num_value_bins)
    time_hl_gauss_distribution = HLGaussDistribution(time_bins, sigma)
    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        base_image = _parse_image(data["observation.image.image"])
        # wrist_image = _parse_image(data["observation.rgb_images.camera_wrist"])

        puppet_left_arm_position = data['observation.state.puppet_left_arm_position']
        puppet_right_arm_position = data['observation.state.puppet_right_arm_position']
        puppet_left_gripper_position = data['observation.state.puppet_left_gripper_position'].unsqueeze(0)
        puppet_right_gripper_position = data['observation.state.puppet_right_gripper_position'].unsqueeze(0)

        state = torch.cat([puppet_left_arm_position, puppet_left_gripper_position, puppet_right_arm_position, puppet_right_gripper_position], dim=-1)
        
        master_left_arm_position = data['observation.state.master_left_arm_position']
        master_right_arm_position = data['observation.state.master_right_arm_position']
        master_left_gripper_position = data['observation.state.master_left_gripper_position'].unsqueeze(-1)
        master_right_gripper_position = data['observation.state.master_right_gripper_position'].unsqueeze(-1)

        master_state = torch.cat([master_left_arm_position, master_left_gripper_position, master_right_arm_position, master_right_gripper_position], dim=-1)
        
        action_left_arm_position = data['action.left_arm_position']
        action_right_arm_position = data['action.right_arm_position']
        action_left_gripper_position = data['action.left_gripper_position'].unsqueeze(-1)
        action_right_gripper_position = data['action.right_gripper_position'].unsqueeze(-1)
        # print("action_left_arm_position: ", action_left_arm_position.shape)
        # print("action_left_gripper_position: ", action_left_gripper_position.shape)
        # print("action_right_arm_position: ", action_right_arm_position.shape)
        # print("action_right_gripper_position: ", action_right_gripper_position.shape)
        action = torch.cat([action_left_arm_position, action_left_gripper_position, action_right_arm_position, action_right_gripper_position], dim=-1)

        # print("the data.keys() is:", data.keys())
        if "discounted_value_return" in data.keys():
            value_gt = data['discounted_value_return']
            # value_gt = value_gt / 300
            # value_gt = map_discounted_value_return_to_label(value_gt)
            # value_gt = value_gt.numpy()
            value_gt = self.discounted_hl_gauss_distribution.encode(value_gt)
            # print("the value_gt is:", value_gt.shape)
            value_gt = value_gt.numpy()
        else:
            value_gt = None
        
        # print("the data keys are:", data.keys())
        # print("the data return_to_go_gt is:", data['return_to_go_gt'])
        # print("the remain_time is:", data['remain_time'])
        
        if 'return_to_go_gt' in data.keys():
            return_to_go_gt = data['return_to_go_gt']
            # print("the return_to_go_gt is:", data['return_to_go_gt'])
            return_to_go_gt = np.array((return_to_go_gt // 50), dtype=np.int32)
            # print("the return_to_go_gt is:", return_to_go_gt)
        else:
            return_to_go_gt = None
        
        # print("the data keys are:", data.keys())
        if 'frame_index' in data.keys():
            frame_index = data['frame_index']
            frame_index = np.array(frame_index, dtype=np.int32)
        else:
            frame_index = None
        
        if 'remain_time' in data.keys():
            remain_time_gt = data['remain_time']
            remain_time_gt = self.time_hl_gauss_distribution.encode(remain_time_gt)
            remain_time_gt = remain_time_gt.numpy()
        else:
            remain_time_gt = None
                
        if "action_judgement" in data.keys():
            action_judgement = data['action_judgement']
            action_judgement = action_judgement.numpy()
        else:
            action_judgement = None

        # print("action: ", action.shape)
        # print("master_state: ", master_state.shape)
        # print("action: ", action.shape)

        state = state.numpy()
        master_state = master_state.numpy()
        # action = action.numpy()
        action = master_state
        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": state,
            # "master_state": master_state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": np.zeros_like(base_image),
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "right_wrist_0_rgb": np.False_,
            },
            "value_gt": value_gt,   
            "return_to_go_gt": return_to_go_gt,
            "remain_time_gt": remain_time_gt,
            "action_judgement": action_judgement,
            "frame_index": frame_index,
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        inputs["actions"] = action

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]


        return inputs



@dataclasses.dataclass(frozen=True)
class X_Humanoid_hl_gauss_RTG_Multi_Camera_Inputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """
    num_value_bins = 256
    sigma = 3.0
    bins = torch.linspace(0, 400, num_value_bins)
    discounted_hl_gauss_distribution = HLGaussDistribution(bins, sigma)
    time_bins = torch.linspace(0, 5000, num_value_bins)
    time_hl_gauss_distribution = HLGaussDistribution(time_bins, sigma)
    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        base_image = _parse_image(data["observation.image.image"])
        left_wrist_image = _parse_image(data["observation.image.left"])
        right_wrist_image = _parse_image(data["observation.image.right"])

        puppet_left_arm_position = data['observation.state.puppet_left_arm_position']
        puppet_right_arm_position = data['observation.state.puppet_right_arm_position']
        puppet_left_gripper_position = data['observation.state.puppet_left_gripper_position'].unsqueeze(0)
        puppet_right_gripper_position = data['observation.state.puppet_right_gripper_position'].unsqueeze(0)

        state = torch.cat([puppet_left_arm_position, puppet_left_gripper_position, puppet_right_arm_position, puppet_right_gripper_position], dim=-1)
        
        master_left_arm_position = data['observation.state.master_left_arm_position']
        master_right_arm_position = data['observation.state.master_right_arm_position']
        master_left_gripper_position = data['observation.state.master_left_gripper_position'].unsqueeze(-1)
        master_right_gripper_position = data['observation.state.master_right_gripper_position'].unsqueeze(-1)

        master_state = torch.cat([master_left_arm_position, master_left_gripper_position, master_right_arm_position, master_right_gripper_position], dim=-1)
        
        action_left_arm_position = data['action.left_arm_position']
        action_right_arm_position = data['action.right_arm_position']
        action_left_gripper_position = data['action.left_gripper_position'].unsqueeze(-1)
        action_right_gripper_position = data['action.right_gripper_position'].unsqueeze(-1)
        # print("action_left_arm_position: ", action_left_arm_position.shape)
        # print("action_left_gripper_position: ", action_left_gripper_position.shape)
        # print("action_right_arm_position: ", action_right_arm_position.shape)
        # print("action_right_gripper_position: ", action_right_gripper_position.shape)
        action = torch.cat([action_left_arm_position, action_left_gripper_position, action_right_arm_position, action_right_gripper_position], dim=-1)

        # print("the data.keys() is:", data.keys())
        if "discounted_value_return" in data.keys():
            value_gt = data['discounted_value_return']
            # value_gt = value_gt / 300
            # value_gt = map_discounted_value_return_to_label(value_gt)
            # value_gt = value_gt.numpy()
            value_gt = self.discounted_hl_gauss_distribution.encode(value_gt)
            # print("the value_gt is:", value_gt.shape)
            value_gt = value_gt.numpy()
        else:
            value_gt = None
        
        # print("the data keys are:", data.keys())
        # print("the data return_to_go_gt is:", data['return_to_go_gt'])
        # print("the remain_time is:", data['remain_time'])
        
        if 'return_to_go_gt' in data.keys():
            return_to_go_gt = data['return_to_go_gt']
            # print("the return_to_go_gt is:", data['return_to_go_gt'])
            return_to_go_gt = np.array((return_to_go_gt // 50), dtype=np.int32)
            # print("the return_to_go_gt is:", return_to_go_gt)
        else:
            return_to_go_gt = None
        
        # print("the data keys are:", data.keys())
        if 'frame_index' in data.keys():
            frame_index = data['frame_index']
            frame_index = np.array(frame_index, dtype=np.int32)
        else:
            frame_index = None
        
        if 'remain_time' in data.keys():
            remain_time_gt = data['remain_time']
            remain_time_gt = self.time_hl_gauss_distribution.encode(remain_time_gt)
            remain_time_gt = remain_time_gt.numpy()
        else:
            remain_time_gt = None
                
        if "action_judgement" in data.keys():
            action_judgement = data['action_judgement']
            action_judgement = action_judgement.numpy()
        else:
            action_judgement = None

        # print("action: ", action.shape)
        # print("master_state: ", master_state.shape)
        # print("action: ", action.shape)

        state = state.numpy()
        master_state = master_state.numpy()
        # action = action.numpy()
        action = master_state
        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": state,
            # "master_state": master_state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "right_wrist_0_rgb": np.True_,
            },
            "value_gt": value_gt,   
            "return_to_go_gt": return_to_go_gt,
            "remain_time_gt": remain_time_gt,
            "action_judgement": action_judgement,
            "frame_index": frame_index,
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        inputs["actions"] = action

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]


        return inputs


@dataclasses.dataclass(frozen=True)
class X_Humanoid_hl_gauss_Multi_Camera_Inputs(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """
    num_value_bins = 512
    sigma = 3.0
    bins = torch.linspace(-1200, 700, num_value_bins)
    hl_gauss_distribution = HLGaussDistribution(bins, sigma)
    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        base_image = _parse_image(data["observation.image.image"])
        left_wrist_image = _parse_image(data["observation.image.left"])
        right_wrist_image = _parse_image(data["observation.image.right"])

        puppet_left_arm_position = data['observation.state.puppet_left_arm_position']
        puppet_right_arm_position = data['observation.state.puppet_right_arm_position']
        puppet_left_gripper_position = data['observation.state.puppet_left_gripper_position'].unsqueeze(0)
        puppet_right_gripper_position = data['observation.state.puppet_right_gripper_position'].unsqueeze(0)

        state = torch.cat([puppet_left_arm_position, puppet_left_gripper_position, puppet_right_arm_position, puppet_right_gripper_position], dim=-1)
        
        master_left_arm_position = data['observation.state.master_left_arm_position']
        master_right_arm_position = data['observation.state.master_right_arm_position']
        master_left_gripper_position = data['observation.state.master_left_gripper_position'].unsqueeze(-1)
        master_right_gripper_position = data['observation.state.master_right_gripper_position'].unsqueeze(-1)

        master_state = torch.cat([master_left_arm_position, master_left_gripper_position, master_right_arm_position, master_right_gripper_position], dim=-1)
        
        action_left_arm_position = data['action.left_arm_position']
        action_right_arm_position = data['action.right_arm_position']
        action_left_gripper_position = data['action.left_gripper_position'].unsqueeze(-1)
        action_right_gripper_position = data['action.right_gripper_position'].unsqueeze(-1)
        # print("action_left_arm_position: ", action_left_arm_position.shape)
        # print("action_left_gripper_position: ", action_left_gripper_position.shape)
        # print("action_right_arm_position: ", action_right_arm_position.shape)
        # print("action_right_gripper_position: ", action_right_gripper_position.shape)
        action = torch.cat([action_left_arm_position, action_left_gripper_position, action_right_arm_position, action_right_gripper_position], dim=-1)

        # print("the data.keys() is:", data.keys())
        if "discounted_value_return" in data.keys():
            value_gt = data['discounted_value_return']
            # value_gt = value_gt / 300
            # value_gt = map_discounted_value_return_to_label(value_gt)
            # value_gt = value_gt.numpy()
            value_gt = self.hl_gauss_distribution.encode(value_gt)
            # print("the value_gt is:", value_gt.shape)
            value_gt = value_gt.numpy()
        else:
            value_gt = None

        if "action_judgement" in data.keys():
            action_judgement = data['action_judgement']
            action_judgement = action_judgement.numpy()
        else:
            action_judgement = None

        # print("action: ", action.shape)
        # print("master_state: ", master_state.shape)
        # print("action: ", action.shape)

        state = state.numpy()
        master_state = master_state.numpy()
        # action = action.numpy()
        action = master_state
        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": state,
            # "master_state": master_state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": left_wrist_image,
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": right_wrist_image,
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.True_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "right_wrist_0_rgb": np.True_,
            },
            "value_gt": value_gt,   
            "action_judgement": action_judgement,
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        inputs["actions"] = action

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]


        return inputs




@dataclasses.dataclass(frozen=True)
class X_Humanoid_Inputs_with_dyna_hl_gauss(transforms.DataTransformFn):
    """
    This class is used to convert inputs to the model to the expected format. It is used for both training and inference.

    For your own dataset, you can copy this class and modify the keys based on the comments below to pipe
    the correct elements of your dataset into the model.
    """
    my_bins = torch.tensor(np.load("/mnt/dataset/toago6/workspace/code/hierarchical_rl_in_real_world/12_0114_all_lerobot_version_bins_without_failure_case.npy"))
    hl_gauss_distribution = DynamicHLGauss(bins=my_bins, sigma_scale=0.75)
    # Determines which model will be used.
    # Do not change this for your own dataset.
    model_type: _model.ModelType

    def __call__(self, data: dict) -> dict:
        # Possibly need to parse images to uint8 (H,W,C) since LeRobot automatically
        # stores as float32 (C,H,W), gets skipped for policy inference.
        # Keep this for your own dataset, but if your dataset stores the images
        # in a different key than "observation/image" or "observation/wrist_image",
        # you should change it below.
        # Pi0 models support three image inputs at the moment: one third-person view,
        # and two wrist views (left and right). If your dataset does not have a particular type
        # of image, e.g. wrist images, you can comment it out here and replace it with zeros like we do for the
        # right wrist image below.
        base_image = _parse_image(data["observation.image.image"])
        # wrist_image = _parse_image(data["observation.rgb_images.camera_wrist"])

        puppet_left_arm_position = data['observation.state.puppet_left_arm_position']
        puppet_right_arm_position = data['observation.state.puppet_right_arm_position']
        puppet_left_gripper_position = data['observation.state.puppet_left_gripper_position'].unsqueeze(0)
        puppet_right_gripper_position = data['observation.state.puppet_right_gripper_position'].unsqueeze(0)

        state = torch.cat([puppet_left_arm_position, puppet_left_gripper_position, puppet_right_arm_position, puppet_right_gripper_position], dim=-1)
        
        master_left_arm_position = data['observation.state.master_left_arm_position']
        master_right_arm_position = data['observation.state.master_right_arm_position']
        master_left_gripper_position = data['observation.state.master_left_gripper_position'].unsqueeze(-1)
        master_right_gripper_position = data['observation.state.master_right_gripper_position'].unsqueeze(-1)

        master_state = torch.cat([master_left_arm_position, master_left_gripper_position, master_right_arm_position, master_right_gripper_position], dim=-1)
        
        action_left_arm_position = data['action.left_arm_position']
        action_right_arm_position = data['action.right_arm_position']
        action_left_gripper_position = data['action.left_gripper_position'].unsqueeze(-1)
        action_right_gripper_position = data['action.right_gripper_position'].unsqueeze(-1)
        # print("action_left_arm_position: ", action_left_arm_position.shape)
        # print("action_left_gripper_position: ", action_left_gripper_position.shape)
        # print("action_right_arm_position: ", action_right_arm_position.shape)
        # print("action_right_gripper_position: ", action_right_gripper_position.shape)
        action = torch.cat([action_left_arm_position, action_left_gripper_position, action_right_arm_position, action_right_gripper_position], dim=-1)

        # print("the data.keys() is:", data.keys())
        if "discounted_value_return" in data.keys():
            value_gt = data['discounted_value_return']
            # value_gt = value_gt / 300
            # value_gt = map_discounted_value_return_to_label(value_gt)
            # value_gt = value_gt.numpy()
            value_gt = self.hl_gauss_distribution.encode(value_gt)
            # print("the value_gt is:", value_gt.shape)
            value_gt = value_gt.numpy()
        else:
            value_gt = None

        if "action_judgement" in data.keys():
            action_judgement = data['action_judgement']
            action_judgement = action_judgement.numpy()
        else:
            action_judgement = None

        # print("action: ", action.shape)
        # print("master_state: ", master_state.shape)
        # print("action: ", action.shape)

        state = state.numpy()
        master_state = master_state.numpy()
        # action = action.numpy()
        action = master_state
        # Create inputs dict. Do not change the keys in the dict below.
        inputs = {
            "state": state,
            # "master_state": master_state,
            "image": {
                "base_0_rgb": base_image,
                "left_wrist_0_rgb": np.zeros_like(base_image),
                # Pad any non-existent images with zero-arrays of the appropriate shape.
                "right_wrist_0_rgb": np.zeros_like(base_image),
            },
            "image_mask": {
                "base_0_rgb": np.True_,
                "left_wrist_0_rgb": np.False_,
                # We only mask padding images for pi0 model, not pi0-FAST. Do not change this for your own dataset.
                "right_wrist_0_rgb": np.False_,
            },
            "value_gt": value_gt,   
            "action_judgement": action_judgement,
        }

        # Pad actions to the model action dimension. Keep this for your own dataset.
        # Actions are only available during training.
        inputs["actions"] = action

        # Pass the prompt (aka language instruction) to the model.
        # Keep this for your own dataset (but modify the key if the instruction is not
        # stored in "prompt"; the output dict always needs to have the key "prompt").
        if "prompt" in data:
            inputs["prompt"] = data["prompt"]


        return inputs


@dataclasses.dataclass(frozen=True)
class X_Humanoid_Outputs(transforms.DataTransformFn):
    """
    This class is used to convert outputs from the model back the the dataset specific format. It is
    used for inference only.

    For your own dataset, you can copy this class and modify the action dimension based on the comments below.
    """

    def __call__(self, data: dict) -> dict:
        # Only return the first N actions -- since we padded actions above to fit the model action
        # dimension, we need to now parse out the correct number of actions in the return dict.
        # For Libero, we only return the first 7 actions (since the rest is padding).
        # For your own dataset, replace `7` with the action dimension of your dataset.

        master_left_arm_position = data['observation.state.master_left_arm_position']
        master_right_arm_position = data['observation.state.master_right_arm_position']
        master_left_gripper_position = data['observation.state.master_left_gripper_position'].unsqueeze(-1)
        master_right_gripper_position = data['observation.state.master_right_gripper_position'].unsqueeze(-1)

        master_state = torch.cat([master_left_arm_position, master_left_gripper_position, master_right_arm_position, master_right_gripper_position], dim=-1)
        
        # action_left_arm_position = data['action.left_arm_position']
        # action_right_arm_position = data['action.right_arm_position']
        # action_left_gripper_position = data['action.left_gripper_position']
        # action_right_gripper_position = data['action.right_gripper_position']
        # action = np.concatenate([action_left_arm_position, action_left_gripper_position, action_right_arm_position, action_right_gripper_position], dim=-1)
        
        return {"actions": master_state}
