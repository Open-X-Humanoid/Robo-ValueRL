
import dataclasses
import random
from typing import Protocol, Sequence, runtime_checkable

import numpy as np
import torch

import dataset.image_tools as image_tools
import transforms.array_typing as at
import transforms.tokenizer as _tokenizer
import typing

TypeAlias = typing.TypeAlias
DataDict: TypeAlias = at.PyTree


@runtime_checkable
class DataTransformFn(Protocol):
    def __call__(self, data: DataDict) -> DataDict:
        """Apply transformation to the data.

        Args:
            data: The data to apply the transform to. This is a possibly nested dictionary that contains
                unbatched data elements. Each leaf is expected to be a numpy array. Using JAX arrays is allowed
                but not recommended since it may result in extra GPU memory usage inside data loader worker
                processes.

        Returns:
            The transformed data. Could be the input `data` that was modified in place, or a new data structure.
        """

@dataclasses.dataclass(frozen=True)
class CompositeTransform(DataTransformFn):
    """A composite transform that applies a sequence of transforms in order."""

    transforms: Sequence[DataTransformFn]

    def __call__(self, data: DataDict) -> DataDict:
        for transform in self.transforms:
            data = transform(data)
        return data


def compose(transforms: Sequence[DataTransformFn]) -> DataTransformFn:
    """Compose a sequence of transforms into a single transform."""
    return CompositeTransform(transforms)


@dataclasses.dataclass(frozen=True)
class InjectDefaultPrompt(DataTransformFn):
    prompt: str | None

    def __call__(self, data: DataDict) -> DataDict:
        if self.prompt is not None and "prompt" not in data:
            data["prompt"] = np.asarray(self.prompt)
        return data
    
    
@dataclasses.dataclass(frozen=True)
class NormalizeImages(DataTransformFn):
    def __call__(self, data: DataDict) -> DataDict:
        for key in data.keys():
            if "rgb_images" in key or  key == "image" or key == "next_image":
                if isinstance(data[key], dict):
                    for sub_key in data[key].keys():
                        if data[key][sub_key].dtype == np.uint8:
                            data[key][sub_key] = data[key][sub_key].astype(np.float32) / 255.0 * 2 - 1
                        else:
                            # print("the data[key][sub_key] is:", data[key][sub_key].shape)
                            data[key][sub_key] = data[key][sub_key].astype(np.float32) * 2.0 - 1.0
                        if len(data[key][sub_key].shape) == 3:
                            if data[key][sub_key].shape[-1] == 3:
                                data[key][sub_key] = data[key][sub_key].transpose(2, 0, 1).copy()
                        elif len(data[key][sub_key].shape) == 4:
                            data[key][sub_key] = data[key][sub_key].transpose(0, 3, 1, 2).copy()
                else:
                    if data[key].dtype == np.uint8:
                        data[key] = data[key].astype(np.float32) / 255.0 * 2 - 1
                    else:
                        data[key] = data[key].astype(np.float32) * 2.0 - 1.0
                    if data[key].shape[-1] == 3:
                        data[key] = data[key].transpose(2, 0, 1).copy()
        return data



import numpy as np

@dataclasses.dataclass(frozen=True)
class ChangeGripperAction(DataTransformFn):
    def __call__(self, data: DataDict) -> DataDict:
        # Get state and actions
        state = data["state"]
        actions = data["actions"]


        target_indices = [7, -1]

        for idx in target_indices:
            # Process state
            # Boolean indexing: values above 0.05 become 1.0, otherwise 0.0
            state[..., idx] = (state[..., idx] > 0.05).astype(state.dtype)
            
            # Process actions
            actions[..., idx] = (actions[..., idx] > 0.05).astype(actions.dtype)

        # Write back into the data dict (in case it wasn't modified in place)
        data["state"] = state
        data["actions"] = actions
         
        return data

@dataclasses.dataclass(frozen=True)
class DeltaAction(DataTransformFn):
    mask: Sequence[bool] | None
    def __call__(self, data: DataDict) -> DataDict:
        # print("the data is:", data.keys())
        state, actions = data["state"], data["actions"]
        dims = actions.shape[-1]
        actions[..., :dims] -= np.expand_dims(np.where(self.mask, state, 0), axis=-2)
        data["actions"] = actions
        return data

@dataclasses.dataclass(frozen=True)
class NormalizeStatesActions(DataTransformFn):
    norm_stats: dict | None
    use_quantile_norm: bool = False
    def __call__(self, data: DataDict) -> DataDict:
        
        
        if self.norm_stats is not None:
            if self.use_quantile_norm:
                for key in self.norm_stats.keys():
                    if key in data.keys():
                        q01 = self.norm_stats[key].get("q01")
                        q99 = self.norm_stats[key].get("q99")
                        if q01 is None or q99 is None:
                            continue
                        data[key] = (data[key] - q01) / (q99 - q01 + 1e-6) * 2.0 - 1.0
            else:
                for key in self.norm_stats.keys():
                    if key in data.keys():
                        # print(f"the key is {key}, and the before norm is: {data[key]}")
                        # print("the mean is:", self.norm_stats[key]["mean"])
                        # print("the std is:", self.norm_stats[key]["std"])
                        data[key] = (data[key] - self.norm_stats[key]["mean"]) / (self.norm_stats[key]["std"] + 1e-6)
                        # print(f"the key is {key}, and the after norm is: {data[key]}")
                        # print("the max is:", np.max(data[key]))
                        # print("the min is:", np.min(data[key]))
        return data


@dataclasses.dataclass(frozen=True)
class NormalizeStatesActionsWithMaxMin(DataTransformFn):
    norm_stats: dict | None
    use_quantile_norm: bool = False
    def __call__(self, data: DataDict) -> DataDict:
        
        
        if self.norm_stats is not None:
            if "state" in data.keys():
                data["state"] = (data["state"] - self.norm_stats["state"]["min"]) / (self.norm_stats["state"]["max"] - self.norm_stats["state"]["min"] + 1e-6) * 2.0 - 1.0
            
            if "actions" in data.keys():
                data["actions"] = (data["actions"] - self.norm_stats["actions"]["mean"]) / (self.norm_stats["actions"]["std"] + 1e-6)

        return data


@dataclasses.dataclass(frozen=True)
class UnnormalizeStatesActions(DataTransformFn):
    norm_stats: dict | None
    use_quantile_norm: bool = False
    def __call__(self, data: DataDict) -> DataDict:
        
        if self.norm_stats is not None:
            if self.use_quantile_norm:
                for key in self.norm_stats.keys():
                    if key in data.keys():
                        q01 = self.norm_stats[key].get("q01")
                        q99 = self.norm_stats[key].get("q99")
                        if q01 is None or q99 is None:
                            continue
                        data[key] = (data[key] + 1.0) / 2.0 * (q99 - q01 + 1e-6) + q01
            else:
                for key in self.norm_stats.keys():
                    if key in data.keys():
                        data[key] = data[key] * (self.norm_stats[key]["std"] + 1e-6) + self.norm_stats[key]["mean"]
        return data


@dataclasses.dataclass(frozen=True)
class UnnormalizeStatesActionsWithMaxMin(DataTransformFn):
    norm_stats: dict | None
    use_quantile_norm: bool = False
    def __call__(self, data: DataDict) -> DataDict:
        
        if self.norm_stats is not None:
            if "state" in data.keys():
                data['state'] = data['state'] * (self.norm_stats['state']['max'] - self.norm_stats['state']['min'] + 1e-6) + self.norm_stats['state']['min']
            if "actions" in data.keys():
                data['actions'] = data['actions'] * (self.norm_stats['actions']['std'] + 1e-6) + self.norm_stats['actions']['mean']
        return data


@dataclasses.dataclass(frozen=True)
class ResizeImages(DataTransformFn):
    height: int
    width: int

    def __call__(self, data: DataDict) -> DataDict:

        for key in data.keys():
            if "rgb_images" in key or  key == "image" or key == "next_image" or "history_image" in key:
                # print("the key is:", key)
                if isinstance(data[key], dict):
                    for sub_key in data[key].keys():

                        data[key][sub_key] = image_tools.resize_with_pad(data[key][sub_key], self.height, self.width)

                else:
                    data[key] = image_tools.resize_with_pad(data[key], self.height, self.width)


        return data



@dataclasses.dataclass(frozen=True)
class TokenizePrompt(DataTransformFn):
    tokenizer: _tokenizer.PaligemmaTokenizer
    discrete_state_input: bool = False
    drop_rate: float = 0.3
    
    def __call__(self, data: DataDict) -> DataDict:
        if (prompt := data.pop("prompt", None)) is None:
            raise ValueError("Prompt is required")
        # import pdb; pdb.set_trace()
        if (action_quality := data.get("action_quality", None)) is not None:
            if isinstance(action_quality, torch.Tensor):
                if action_quality.shape[0] == 1:
                    action_quality = action_quality.item()
                else:
                    action_quality = action_quality[0].numpy()
                    
            elif isinstance(action_quality, np.ndarray):
                if action_quality.ndim == 0:
                    action_quality = action_quality.item()
                else:
                    action_quality = action_quality.flat[0]

        else:
            action_quality = None

        if action_quality is not None:
            if random.random() < self.drop_rate:
                action_quality = None

        if self.discrete_state_input:
            if (state := data.get("state", None)) is None:
                raise ValueError("State is required.")
        else:
            state = None

        if not isinstance(prompt, str):
            prompt = prompt.item()

        if isinstance(self.tokenizer, _tokenizer.PaligemmaTokenizerWithQuality):
            tokens, token_masks = self.tokenizer.tokenize(prompt, state, action_quality)
        elif isinstance(self.tokenizer, _tokenizer.PaligemmaTokenizer):
            tokens, token_masks = self.tokenizer.tokenize(prompt, state)
        return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks, "prompt": prompt}



# @dataclasses.dataclass(frozen=True)
# class TokenizePrompt(DataTransformFn):
#     tokenizer: _tokenizer.PaligemmaTokenizer

#     def __call__(self, data: DataDict) -> DataDict:

#         if (prompt := data.pop("prompt", None)) is None:
#             raise ValueError("Prompt is required")
#         # print("the prompt is:", prompt)
#         if not isinstance(prompt, str):
#             prompt = prompt.item()

#         tokens, token_masks = self.tokenizer.tokenize(prompt)
#         return {**data, "tokenized_prompt": tokens, "tokenized_prompt_mask": token_masks, "prompt": prompt}



@dataclasses.dataclass(frozen=True)
class PadStatesAndActions(DataTransformFn):
    """Zero-pads states and actions to the model action dimension."""

    model_action_dim: int

    def __call__(self, data: DataDict) -> DataDict:
        for key in data.keys():
            if "state" in key or "actions" in key:
                data[key] = pad_to_dim(data[key], self.model_action_dim, axis=-1)
        # print("the data is:", data)
        return data

def pad_to_dim(x: np.ndarray, target_dim: int, axis: int = -1, value: float = 0.0) -> np.ndarray:
    """Pad an array to the target dimension with zeros along the specified axis."""
    current_dim = x.shape[axis]
    if current_dim < target_dim:
        pad_width = [(0, 0)] * len(x.shape)
        pad_width[axis] = (0, target_dim - current_dim)
        return np.pad(x, pad_width, constant_values=value)
    return x
