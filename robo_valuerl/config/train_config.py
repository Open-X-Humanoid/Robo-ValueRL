"""See _CONFIGS for the list of available configs."""

import abc
from ast import List
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
from typing import Any, Literal, Protocol, TypeAlias, List, Sequence

import etils.epath as epath
import flax.nnx as nnx
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.tokenizer as _tokenizer
import openpi.policies.x_humaniod_policy as x_humanoid_policy

import openpi.policies.ur5_policy as ur5_policy
import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms
from config.model_config import ResNetDiffusionConfig
# from config.model_config import XHumanoidResNetConsistencyPolicyConfig
from config.model_config import XHumanoidResNetConRFTConsistencyPolicyConfig
from config.model_config import XHumanoidConsistencyConfig
ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter

UR5_DATA_LIST = ['ur_5e_singleArm-gripper-4cameras_1_press_stop_button_of_control_box_20250901', 'ur_wrist_place_on_golden_tray_20250425', 'ur_single_4_1_pour_screw_into_storage_box', 'ur_wrist_pick_blue_column_block_250422', 'ur_wrist_pick_red_block_250421', 'ur_5e_singleArm-gripper-4cameras_1_hang_lab_cloth_true', 'ur_5e_singleArm-gripper-4cameras__close_toolbox', 'ur_wrist_pick_pink_cup_20250428', 'ur_wrist_clean_plate_20250427', 'ur_5e_singleArm-gripper-4cameras__open_toolbox_20250827', 'ur_wrist_pour_water_into_cup_20250428', 'ur_wrist_clean_plate_20250428', 'ur_5e_singleArm-gripper-4cameras_1_pour_oil_for_gear', 'ur_5e_singleArm-gripper-4cameras_1_pour_water_into_glass_beaker', 'ur_5e_singleArm-gripper-4cameras__open_toolbox', 'ur_wrist_clean_plate_20250425', 'ur_wrist_pour_water_into_cup_20250425', 'ur_wrist_pick_red_block_250422', 'ur_5e_singleArm-gripper-4cameras_1_press_stop_button_of_control_box', 'ur_wrist_place_on_golden_tray_20250427', 'ur_wrist_pick_bread_20250425', 'ur_wrist_pick_pink_cup_20250425', 'ur_wrist_pour_water_into_cup_20250427', 'ur_5e_singleArm-gripper-4cameras_1_assemble_valve_thread', 'ur_wrist_insert_bread_into_bag_20250425', 'ur_wrist_pick_pink_cup_20250427', 'ur_wrist_place_on_golden_tray_20250428', 'ur_5e_singleArm-gripper-4cameras_1_hang_lab_cloth_true_20250903', 'ur_wrist_place_red_block_on_tray_250422']

# from openpi.training.config import LeRobot_X_HumanoidDataConfig, DataConfigFactory
@dataclasses.dataclass(frozen=True)
class AssetsConfig:
    """Determines the location of assets (e.g., norm stats) that will be used to set up the data pipeline.

    These assets will be replicated inside the checkpoint under the `assets/asset_id` directory.

    This can be used to load assets from a different checkpoint (e.g., base model checkpoint) or some other
    centralized location. For example, to load the norm stats for the Trossen robot from the base model checkpoint
    during fine-tuning, use:

    ```
    AssetsConfig(
        assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
        asset_id="trossen",
    )
    ```
    """

    # Assets directory. If not provided, the config assets_dirs will be used. This is useful to load assets from
    # a different checkpoint (e.g., base model checkpoint) or some other centralized location.
    assets_dir: str | None = None

    # Asset id. If not provided, the repo id will be used. This allows users to reference assets that describe
    # different robot platforms.
    asset_id: str | None = None


@dataclasses.dataclass(frozen=True)
class DataConfig:
    # LeRobot repo id. If None, fake data will be created.
    repo_id: str | None = None
    
    repo_ids: list[str] | None = None
    use_multi_repo: bool = False
    # The root directory of the LeRobot dataset.
    root: str | None = None
    # Directory within the assets directory containing the data assets.
    asset_id: str | None = None
    # Contains precomputed normalization stats. If None, normalization will not be performed.
    norm_stats: dict[str, _transforms.NormStats] | None = None

    # Used to adopt the inputs from a dataset specific format to a common format
    # which is expected by the data transforms.
    repack_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Data transforms, typically include robot specific transformations. Will be applied
    # before the data is normalized. See `model.Observation` and `model.Actions` to learn about the
    # normalized data.
    data_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # Model specific transforms. Will be applied after the data is normalized.
    model_transforms: _transforms.Group = dataclasses.field(default_factory=_transforms.Group)
    # If true, will use quantile normalization. Otherwise, normal z-score normalization will be used.
    use_quantile_norm: bool = False

    # Names of keys that will be used by the data loader to generate the action sequence. The length of the
    # sequence is defined by the `action_horizon` field in the model config. This should be adjusted if your
    # LeRobot dataset is using different keys to represent the action.
    action_sequence_keys: Sequence[str] = ("actions",)
    obs_sequence_keys: Sequence[str] = ("observation.state.arm_joint_position",
        "observation.state.hand_joint_position", "observation.rgb_images.camera_front", "observation.rgb_images.camera_wrist")
    # If true, will use the LeRobot dataset task to define the prompt.
    prompt_from_task: bool = False

    # Only used for RLDS data loader (ie currently only used for DROID).
    rlds_data_dir: str | None = None
    # Action space for DROID dataset.
    action_space: droid_rlds_dataset.DroidActionSpace | None = None
    # Path to the data filter file for DROID dataset
    filter_dict_path: str | None = None


class GroupFactory(Protocol):
    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        """Create a group."""


@dataclasses.dataclass(frozen=True)
class ModelTransformFactory(GroupFactory):
    """Creates model transforms for standard pi0 models."""

    # If provided, will determine the default prompt that be used by the model.
    default_prompt: str | None = None

    def __call__(self, model_config: _model.BaseModelConfig) -> _transforms.Group:
        match model_config.model_type:
            case _model.ModelType.PI0:
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI05:
                assert isinstance(model_config, pi0_config.Pi0Config)
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizePrompt(
                            _tokenizer.PaligemmaTokenizer(model_config.max_token_len),
                            discrete_state_input=model_config.discrete_state_input,
                        ),
                        _transforms.PadStatesAndActions(model_config.action_dim),
                    ],
                )
            case _model.ModelType.PI0_FAST:
                tokenizer_cls = (
                    _tokenizer.FASTTokenizer
                    if model_config.fast_model_tokenizer is None
                    else model_config.fast_model_tokenizer
                )
                tokenizer_kwargs = (
                    {} if model_config.fast_model_tokenizer_kwargs is None else model_config.fast_model_tokenizer_kwargs
                )
                return _transforms.Group(
                    inputs=[
                        _transforms.InjectDefaultPrompt(self.default_prompt),
                        _transforms.ResizeImages(224, 224),
                        _transforms.TokenizeFASTInputs(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                        ),
                    ],
                    outputs=[
                        _transforms.ExtractFASTActions(
                            tokenizer_cls(model_config.max_token_len, **tokenizer_kwargs),
                            action_horizon=model_config.action_horizon,
                            action_dim=model_config.action_dim,
                        )
                    ],
                )


@dataclasses.dataclass(frozen=True)
class DataConfigFactory(abc.ABC):
    # The LeRobot repo id.
    repo_id: str = tyro.MISSING
    
    repo_ids: list[str] | None = None
    use_multi_repo: bool = False
    # The root directory of the LeRobot dataset.
    root: str | None = None
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    # Base config that will be updated by the factory.
    base_config: tyro.conf.Suppress[DataConfig | None] = None

    action_sequence_keys: Sequence[str] = ("actions",)
    obs_sequence_keys: Sequence[str] = ("observation.state.arm_joint_position",
        "observation.state.hand_joint_position", "observation.rgb_images.camera_front", "observation.rgb_images.camera_wrist")

    @abc.abstractmethod
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        """Create a data config."""

    def create_base_config(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        repo_id = self.repo_id if self.repo_id is not tyro.MISSING else None
        root = self.root if self.root is not None else None
        asset_id = self.assets.asset_id or repo_id
        action_sequence_keys = self.action_sequence_keys
        obs_sequence_keys = self.obs_sequence_keys
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            root=root,
            repo_ids=self.repo_ids,
            use_multi_repo=self.use_multi_repo,
            asset_id=asset_id,
            action_sequence_keys=action_sequence_keys,
            obs_sequence_keys=obs_sequence_keys,
            norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            use_quantile_norm=model_config.model_type != ModelType.PI0,
        )

    def _load_norm_stats(self, assets_dir: epath.Path, asset_id: str | None) -> dict[str, _transforms.NormStats] | None:
        if asset_id is None:
            return None
        try:
            data_assets_dir = str(assets_dir / asset_id)
            norm_stats = _normalize.load(_download.maybe_download(data_assets_dir))
            logging.info(f"Loaded norm stats from {data_assets_dir}")
            return norm_stats
        except FileNotFoundError:
            logging.info(f"Norm stats not found in {data_assets_dir}, skipping.")
        return None


@dataclasses.dataclass(frozen=True)
class FakeDataConfig(DataConfigFactory):
    repo_id: str = "fake"

    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        return DataConfig(repo_id=self.repo_id)


@dataclasses.dataclass(frozen=True)
class LeRobotUR5DataConfig(DataConfigFactory):
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action.end_effector", "action.hand_joint_position", "action.arm_joint_position")
    obs_sequence_keys: Sequence[str] = ("observation.state.arm_joint_position",
        "observation.state.hand_joint_position", "observation.rgb_images.camera_front", "observation.rgb_images.camera_wrist")
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Boilerplate for remapping keys from the LeRobot dataset. We assume no renaming needed here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation.state.arm_joint_position": "observation.state.arm_joint_position",
                        "observation.state.hand_joint_position": "observation.state.hand_joint_position",
                        "observation.rgb_images.camera_front": "observation.rgb_images.camera_front",
                        "observation.rgb_images.camera_wrist": "observation.rgb_images.camera_wrist",
                        "prompt": "prompt",
                        "action.end_effector": "action.end_effector",
                        "action.arm_joint_position": "action.arm_joint_position",
                        "action.hand_joint_position": "action.hand_joint_position",
                        "return_to_go": "return_to_go",
                        "reward": "reward",
                    }
                )
            ]
        )
        
        # These transforms are the ones we wrote earlier.
        data_transforms = _transforms.Group(
            inputs=[ur5_policy.UR5Inputs(model_type=model_config.model_type)],
            outputs=[ur5_policy.UR5Outputs()],
        )

        # Convert absolute actions to delta actions.
        # By convention, we do not convert the gripper action (7th dimension).
        # delta_action_mask = _transforms.make_bool_mask(6, -1)
        # data_transforms = data_transforms.push(
        #     inputs=[_transforms.DeltaActions(delta_action_mask)],
        #     outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        # )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)
        # import pdb; pdb.set_trace()
        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms
        )


@dataclasses.dataclass(frozen=True)
class RL_LeRobotUR5DataConfig(DataConfigFactory):
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action.end_effector", "action.hand_joint_position")
    obs_sequence_keys: Sequence[str] = ("observation.state.arm_joint_position", "next_observation.state.arm_joint_position",
        "observation.state.hand_joint_position","next_observation.state.hand_joint_position",
        "observation.rgb_images.camera_front", "next_observation.rgb_images.camera_front",
        "observation.rgb_images.camera_wrist", "next_observation.rgb_images.camera_wrist")
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Boilerplate for remapping keys from the LeRobot dataset. We assume no renaming needed here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation.state.arm_joint_position": "observation.state.arm_joint_position",
                        "observation.state.hand_joint_position": "observation.state.hand_joint_position",
                        "observation.rgb_images.camera_front": "observation.rgb_images.camera_front",
                        "observation.rgb_images.camera_wrist": "observation.rgb_images.camera_wrist",
                        "next_observation.state.arm_joint_position": "next_observation.state.arm_joint_position",
                        "next_observation.state.hand_joint_position": "next_observation.state.hand_joint_position",
                        "next_observation.rgb_images.camera_front": "next_observation.rgb_images.camera_front",
                        "next_observation.rgb_images.camera_wrist": "next_observation.rgb_images.camera_wrist",
                        "prompt": "prompt",
                        "action.end_effector": "action.end_effector",
                        "action.hand_joint_position": "action.hand_joint_position",
                        "return_to_go": "return_to_go",
                        "reward": "reward",
                    }
                )
            ]
        )
        
        # These transforms are the ones we wrote earlier.
        data_transforms = _transforms.Group(
            inputs=[ur5_policy.RL_UR5Inputs(model_type=model_config.model_type)],
            outputs=[ur5_policy.RL_UR5Outputs()],
        )

        # Convert absolute actions to delta actions.
        # By convention, we do not convert the gripper action (7th dimension).
        # delta_action_mask = _transforms.make_bool_mask(6, -1)
        # data_transforms = data_transforms.push(
        #     inputs=[_transforms.DeltaActions(delta_action_mask)],
        #     outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        # )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)
        # import pdb; pdb.set_trace()
        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms
        )

@dataclasses.dataclass(frozen=True)
class LeRobot_X_HumanoidDataConfig(DataConfigFactory):
    # Action keys that will be used to read the action sequence from the dataset.
    action_sequence_keys: Sequence[str] = ("action.left_arm_position", "action.right_arm_position", "action.left_gripper_position", "action.right_gripper_position")
    obs_sequence_keys: Sequence[str] = ("observation.state.arm_joint_position",
        "observation.state.hand_joint_position", "observation.rgb_images.camera_front", "observation.rgb_images.camera_wrist")
    @override
    def create(self, assets_dirs: pathlib.Path, model_config: _model.BaseModelConfig) -> DataConfig:
        # Boilerplate for remapping keys from the LeRobot dataset. We assume no renaming needed here.
        repack_transform = _transforms.Group(
            inputs=[
                _transforms.RepackTransform(
                    {
                        "observation.state.puppet_left_arm_position": "observation.state.puppet_left_arm_position",
                        "observation.state.puppet_right_arm_position": "observation.state.puppet_right_arm_position",
                        "observation.state.puppet_left_gripper_position": "observation.state.puppet_left_gripper_position",
                        "observation.state.puppet_right_gripper_position": "observation.state.puppet_right_gripper_position",
                        "observation.state.master_left_arm_position": "observation.state.master_left_arm_position",
                        "observation.state.master_right_arm_position": "observation.state.master_right_arm_position",
                        "observation.state.master_left_gripper_position": "observation.state.master_left_gripper_position",
                        "observation.state.master_right_gripper_position": "observation.state.master_right_gripper_position",
                        "observation.image.image": "observation.image.image",
                        "prompt": "prompt",
                        "action.left_arm_position": "action.left_arm_position",
                        "action.right_arm_position": "action.right_arm_position",
                        "action.left_gripper_position": "action.left_gripper_position",
                        "action.right_gripper_position": "action.right_gripper_position",
                    }
                )
            ]
        )
        
        # These transforms are the ones we wrote earlier.
        data_transforms = _transforms.Group(
            inputs=[x_humanoid_policy.X_Humanoid_Inputs(model_type=model_config.model_type)],
            outputs=[x_humanoid_policy.X_Humanoid_Outputs()],
        )

        # Convert absolute actions to delta actions.
        # By convention, we do not convert the gripper action (7th dimension).
        # delta_action_mask = _transforms.make_bool_mask(6, -1)
        # data_transforms = data_transforms.push(
        #     inputs=[_transforms.DeltaActions(delta_action_mask)],
        #     outputs=[_transforms.AbsoluteActions(delta_action_mask)],
        # )

        # Model transforms include things like tokenizing the prompt and action targets
        # You do not need to change anything here for your own dataset.
        model_transforms = ModelTransformFactory()(model_config)
        # import pdb; pdb.set_trace()
        # We return all data transforms for training and inference. No need to change anything here.
        return dataclasses.replace(
            self.create_base_config(assets_dirs, model_config),
            repack_transforms=repack_transform,
            data_transforms=data_transforms,
            model_transforms=model_transforms
        )



@dataclasses.dataclass(frozen=True)
class TrainConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING
    chunk_size: int = 10

    host: str = "localhost"
    port: int = 8888

    action_sequence_keys: List[str] = dataclasses.field(
        default_factory=lambda: ["action.arm_joint_position", "action.hand_joint_position"]
    )
    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model:object = dataclasses.field(default_factory=pi0_config.Pi0Config)
    # define the algorithm config
    algorithm: object = None
    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # Precision for PyTorch training.
    pytorch_training_precision: Literal["bfloat16", "float32"] = "bfloat16"

    lr_schedule: _optimizer.LRScheduleConfig = dataclasses.field(default_factory=_optimizer.CosineDecaySchedule)
    optimizer: _optimizer.OptimizerConfig = dataclasses.field(default_factory=_optimizer.AdamW)
    ema_decay: float | None = 0.99

    # Specifies which weights should be frozen.
    freeze_filter: tyro.conf.Suppress[Filter] = dataclasses.field(default_factory=nnx.Nothing)

    # Determines the data to be trained on.
    data: DataConfigFactory = dataclasses.field(default_factory=FakeDataConfig)

    # Base directory for config assets (e.g., norm stats).
    assets_base_dir: str = "./assets"
    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    # Random seed that will be used by random generators during training.
    seed: int = 42
    # Global batch size.
    batch_size: int = 32
    sample_from_ratio: float = 1.0
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 50_000
    
    num_epochs: int = 100

    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 2
    # If set, any existing checkpoints matching step % keep_period == 0 will not be deleted.
    keep_period: int | None = 5000

    # If true, will overwrite the checkpoint directory if it already exists.
    overwrite: bool = False
    # If true, will resume training from the last checkpoint.
    resume: bool = False

    # If true, will enable wandb logging.
    wandb_enabled: bool = True

    # Used to pass metadata to the policy server.
    policy_metadata: dict[str, Any] | None = None

    # If the value is greater than 1, FSDP will be enabled and shard across number of specified devices; overall
    # device memory will be reduced but training could potentially be slower.
    # eg. if total device is 4 and fsdp devices is 2; then the model will shard to 2 devices and run
    # data parallel between 2 groups of devices.
    fsdp_devices: int = 1

    @property
    def assets_dirs(self) -> pathlib.Path:
        """Get the assets directory for this config."""
        return (pathlib.Path(self.assets_base_dir) / self.name).resolve()

    @property
    def checkpoint_dir(self) -> pathlib.Path:
        """Get the checkpoint directory for this config."""
        if not self.exp_name:
            raise ValueError("--exp_name must be set")
        return (pathlib.Path(self.checkpoint_base_dir) / self.name / self.exp_name).resolve()

    @property
    def trainable_filter(self) -> nnx.filterlib.Filter:
        """Get the filter for the trainable parameters."""
        return nnx.All(nnx.Param, nnx.Not(self.freeze_filter))

    def __post_init__(self) -> None:
        if self.resume and self.overwrite:
            raise ValueError("Cannot resume and overwrite at the same time.")


from config.model_config import ResNetMLPConfig, ResNetGMMConfig, CalQLConfig, XHumanoidResNetDiffusionConfig, XHumanoidMLPConfig, XHumanoidCalQLConfig
from config.algorithm_cfg import BCAlgorithmConfig, CalQLAlgorithmConfig, IQLAlgorithmConfig, TDAlgorithmConfig, ConsistencyAlgorithmConfig

# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    

    TrainConfig(
        name="pi0_joint_inference",
        model=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            regression=True,
            regression_rl=False,
            action_dim = 7
        ),
        batch_size= 1,
        data=LeRobotUR5DataConfig(
            repo_id=None,
            use_multi_repo=True,
            repo_ids=["ur_5e_singleArm-gripper-4cameras__close_toolbox"],
            # repo_ids=UR5_DATA_LIST,
            root="/home/eai/Dev/wink/openpi/ur_test_data",
            # This config lets us reload the UR5 normalization stats from the base model checkpoint.
            # Reloading normalization stats can help transfer pre-trained models to new environments.
            # See the [norm_stats.md](../docs/norm_stats.md) file for more details.
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="ur5e",
            ),
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. The recommended setting is True.
                prompt_from_task=True,
            ),
        ),
        # freeze_filter=pi0_config.Pi0Config(
        #     paligemma_variant="gemma_2b", action_expert_variant="gemma_300m"
        # ).get_freeze_filter(),
        # Load the pi0 base model checkpoint.
        # weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=50_000,
        save_interval = 10000,
        pytorch_weight_path = "/home/eai/Dev/wink/openpi/checkpoints/pi0_ur5_test_joint_action/pi0_ur5_test_joint_action/6000"
    ),

    TrainConfig(
        name="pi0_inference",
        model=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            regression=True,
            regression_rl=False,
            action_dim = 7
        ),
        batch_size= 1,
        data=LeRobotUR5DataConfig(
            repo_id=None,
            use_multi_repo=True,
            repo_ids=["ur_5e_singleArm-gripper-4cameras__close_toolbox"],
            # repo_ids=UR5_DATA_LIST,
            root="/home/eai/Dev/wink/openpi/ur_test_data",
            # This config lets us reload the UR5 normalization stats from the base model checkpoint.
            # Reloading normalization stats can help transfer pre-trained models to new environments.
            # See the [norm_stats.md](../docs/norm_stats.md) file for more details.
            assets=AssetsConfig(
                assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
                asset_id="ur5e",
            ),
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. The recommended setting is True.
                prompt_from_task=True,
            ),
        ),
        # freeze_filter=pi0_config.Pi0Config(
        #     paligemma_variant="gemma_2b", action_expert_variant="gemma_300m"
        # ).get_freeze_filter(),
        # Load the pi0 base model checkpoint.
        # weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=50_000,
        save_interval = 10000,
        pytorch_weight_path = "/home/eai/Dev/wink/hierarchical_rl/checkpoints/pick_pink_cup/2500"
    ),

    TrainConfig(
        name="resnet_mlp_agent",
        model=ResNetMLPConfig(
            action_dim = 7,
            chunk_size = 10,
            state_dim = 7,
            hidden_dim = 1024,
            resnet_model = "resnet18"
        ),
        algorithm=BCAlgorithmConfig(
            encoder_lr = 3e-4,
            policy_lr = 3e-4,
            scheduler_step_size = 1000,
            scheduler_gamma = 0.5
        ),
        batch_size= 1024,
        data=LeRobotUR5DataConfig(
            repo_id=None,
            use_multi_repo=True,
            repo_ids=["ur_wrist_pick_pink_cup_20250425",
                "ur_wrist_pick_pink_cup_20250427", "ur_wrist_pick_pink_cup_20250428"],
            # repo_ids=UR5_DATA_LIST,
            root="/mnt/dataset/toago6/workspace/code/ur_rl_with_reward_data",
            # This config lets us reload the UR5 normalization stats from the base model checkpoint.
            # Reloading normalization stats can help transfer pre-trained models to new environments.
            # See the [norm_stats.md](../docs/norm_stats.md) file for more details.
            # assets=AssetsConfig(
            #     assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            #     asset_id="ur5e",
            # ),
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. The recommended setting is True.
                prompt_from_task=True,
            ),
        ),
        # freeze_filter=pi0_config.Pi0Config(
        #     paligemma_variant="gemma_2b", action_expert_variant="gemma_300m"
        # ).get_freeze_filter(),
        # Load the pi0 base model checkpoint.
        # weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=100,
        num_epochs=500,
        save_interval = 30,
    ),

    TrainConfig(
        name="resnet_mlp_agent_eai",
        model=ResNetMLPConfig(
            action_dim = 7,
            chunk_size = 10,
            state_dim = 7,
            hidden_dim = 1024,
            resnet_model = "resnet18"
        ),
        algorithm=BCAlgorithmConfig(
            encoder_lr = 3e-4,
            policy_lr = 3e-4,
            scheduler_step_size = 1000,
            scheduler_gamma = 0.5
        ),
        batch_size= 1024,
        data=LeRobotUR5DataConfig(
            repo_id=None,
            use_multi_repo=True,
            repo_ids=["ur_wrist_pick_pink_cup_20250425",
                "ur_wrist_pick_pink_cup_20250427", "ur_wrist_pick_pink_cup_20250428"],
            # repo_ids=UR5_DATA_LIST,
            root="/home/eai/Dev/wink/ur_rl_with_reward_data",
            # This config lets us reload the UR5 normalization stats from the base model checkpoint.
            # Reloading normalization stats can help transfer pre-trained models to new environments.
            # See the [norm_stats.md](../docs/norm_stats.md) file for more details.
            # assets=AssetsConfig(
            #     assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            #     asset_id="ur5e",
            # ),
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. The recommended setting is True.
                prompt_from_task=True,
            ),
        ),
        # freeze_filter=pi0_config.Pi0Config(
        #     paligemma_variant="gemma_2b", action_expert_variant="gemma_300m"
        # ).get_freeze_filter(),
        # Load the pi0 base model checkpoint.
        # weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=100,
        num_epochs=500,
        save_interval = 30,
    ),
    
    TrainConfig(
        name="resnet_gmm_agent",
        model=ResNetGMMConfig(
            action_dim = 7,
            chunk_size = 10,
            state_dim = 7,
            num_components = 5,
            resnet_model = "resnet18"
        ),
        algorithm=BCAlgorithmConfig(
            encoder_lr = 3e-4,
            policy_lr = 3e-4,
            scheduler_step_size = 1000,
            scheduler_gamma = 0.5
        ),
        batch_size= 1,
        data=LeRobotUR5DataConfig(
            repo_id=None,
            use_multi_repo=True,
            repo_ids=["ur_wrist_pick_pink_cup_20250425"],
            # repo_ids=UR5_DATA_LIST,
            root="/mnt/dataset/toago6/workspace/code/ur_rl_with_reward_data",
            # This config lets us reload the UR5 normalization stats from the base model checkpoint.
            # Reloading normalization stats can help transfer pre-trained models to new environments.
            # See the [norm_stats.md](../docs/norm_stats.md) file for more details.
            # assets=AssetsConfig(
            #     assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            #     asset_id="ur5e",
            # ),
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. The recommended setting is True.
                prompt_from_task=True,
            ),
        ),
        # freeze_filter=pi0_config.Pi0Config(
        #     paligemma_variant="gemma_2b", action_expert_variant="gemma_300m"
        # ).get_freeze_filter(),
        # Load the pi0 base model checkpoint.
        # weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=50_000,
        save_interval = 10000,
    ),


    TrainConfig(
        name="calql_agent",
        model=CalQLConfig(
            action_dim = 7,
            chunk_size = 10,
            state_dim = 7,
            resnet_model = "resnet18"
        ),
        algorithm=CalQLAlgorithmConfig(
            actor_lr = 3e-4,
            critic_lr = 3e-4,
            scheduler_step_size = 1000,
            scheduler_gamma = 0.5
        ),
        batch_size= 16,
        data=LeRobotUR5DataConfig(
            repo_id=None,
            use_multi_repo=True,
            repo_ids=["ur_wrist_pick_pink_cup_20250425",
                "ur_wrist_pick_pink_cup_20250427", "ur_wrist_pick_pink_cup_20250428"],
            # repo_ids=UR5_DATA_LIST,
            root="/mnt/dataset/toago6/workspace/code/ur_rl_with_reward_data",
            # This config lets us reload the UR5 normalization stats from the base model checkpoint.
            # Reloading normalization stats can help transfer pre-trained models to new environments.
            # See the [norm_stats.md](../docs/norm_stats.md) file for more details.
            # assets=AssetsConfig(
            #     assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            #     asset_id="ur5e",
            # ),
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. The recommended setting is True.
                prompt_from_task=True,
            ),
        ),
        # freeze_filter=pi0_config.Pi0Config(
        #     paligemma_variant="gemma_2b", action_expert_variant="gemma_300m"
        # ).get_freeze_filter(),
        # Load the pi0 base model checkpoint.
        # weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=50_000,
        num_epochs=500,
        save_interval = 30,
    ),
    TrainConfig(
        name="resnet_mlp_inference",
        model=ResNetMLPConfig(
            action_dim = 7,
            chunk_size = 10,
            state_dim = 7,
            hidden_dim = 1024,
            resnet_model = "resnet18",
            pretrained_ckpt_path = "/home/eai/Dev/wink/hierarchical_rl/checkpoints/pick_cup_bc"
        ),
        algorithm=BCAlgorithmConfig(
            encoder_lr = 3e-4,
            policy_lr = 3e-4,
            scheduler_step_size = 1000,
            scheduler_gamma = 0.5
        ),
        batch_size= 1024,
        data=LeRobotUR5DataConfig(
            repo_id=None,
            use_multi_repo=True,
            repo_ids=["ur_wrist_pick_pink_cup_20250425",
                "ur_wrist_pick_pink_cup_20250427", "ur_wrist_pick_pink_cup_20250428"],
            # repo_ids=UR5_DATA_LIST,
            root="/mnt/dataset/toago6/workspace/code/ur_rl_with_reward_data",
            # This config lets us reload the UR5 normalization stats from the base model checkpoint.
            # Reloading normalization stats can help transfer pre-trained models to new environments.
            # See the [norm_stats.md](../docs/norm_stats.md) file for more details.
            # assets=AssetsConfig(
            #     assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            #     asset_id="ur5e",
            # ),
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. The recommended setting is True.
                prompt_from_task=True,
            ),
        ),
        # freeze_filter=pi0_config.Pi0Config(
        #     paligemma_variant="gemma_2b", action_expert_variant="gemma_300m"
        # ).get_freeze_filter(),
        # Load the pi0 base model checkpoint.
        # weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=100,
        num_epochs=500,
        save_interval = 30,
    ),

    TrainConfig(
        name="resnet_diffusion_agent_eai",
        model=ResNetDiffusionConfig(
            action_dim = 7,
            chunk_size = 10,
            state_dim = 7,
            hidden_dim = 1024,
            resnet_model = "resnet18",
            
        ),
        algorithm=BCAlgorithmConfig(
            encoder_lr = 3e-4,
            policy_lr = 3e-4,
            scheduler_step_size = 1000,
            scheduler_gamma = 0.5
        ),
        batch_size= 1024,
        
        
        data=LeRobotUR5DataConfig(
            repo_id=None,
            use_multi_repo=True,
            repo_ids=["ur_wrist_pick_pink_cup_20250425",
                "ur_wrist_pick_pink_cup_20250427", "ur_wrist_pick_pink_cup_20250428"],
            # repo_ids=UR5_DATA_LIST,
            root="/home/eai/Dev/wink/ur_rl_with_reward_data",
            # This config lets us reload the UR5 normalization stats from the base model checkpoint.
            # Reloading normalization stats can help transfer pre-trained models to new environments.
            # See the [norm_stats.md](../docs/norm_stats.md) file for more details.
            # assets=AssetsConfig(
            #     assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            #     asset_id="ur5e",
            # ),
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. The recommended setting is True.
                prompt_from_task=True,
            ),
        ),
        # freeze_filter=pi0_config.Pi0Config(
        #     paligemma_variant="gemma_2b", action_expert_variant="gemma_300m"
        # ).get_freeze_filter(),
        # Load the pi0 base model checkpoint.
        # weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=100,
        num_epochs=500,
        save_interval = 30,
    ),
    TrainConfig(
        name="resnet_diffusion_agent",
        model=ResNetDiffusionConfig(
            action_dim = 7,
            chunk_size = 10,
            state_dim = 7,
            hidden_dim = 1024,
            resnet_model = "resnet18",
            
        ),
        algorithm=BCAlgorithmConfig(
            encoder_lr = 3e-4,
            policy_lr = 3e-4,
            scheduler_step_size = 1000,
            scheduler_gamma = 0.5
        ),
        batch_size= 1024,
        data=LeRobotUR5DataConfig(
            repo_id=None,
            use_multi_repo=True,
            repo_ids=["ur_wrist_pick_pink_cup_20250425",
                "ur_wrist_pick_pink_cup_20250427", "ur_wrist_pick_pink_cup_20250428"],
            # repo_ids=UR5_DATA_LIST,
            root="/mnt/dataset/toago6/workspace/code/ur_rl_with_reward_data",
            # This config lets us reload the UR5 normalization stats from the base model checkpoint.
            # Reloading normalization stats can help transfer pre-trained models to new environments.
            # See the [norm_stats.md](../docs/norm_stats.md) file for more details.
            # assets=AssetsConfig(
            #     assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            #     asset_id="ur5e",
            # ),
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. The recommended setting is True.
                prompt_from_task=True,
            ),
        ),
        # freeze_filter=pi0_config.Pi0Config(
        #     paligemma_variant="gemma_2b", action_expert_variant="gemma_300m"
        # ).get_freeze_filter(),
        # Load the pi0 base model checkpoint.
        # weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=100,
        num_epochs=500,
        save_interval = 30,
    ),


    TrainConfig(
        name="x_humanoid_resnet_diffusion_agent",
        model=XHumanoidResNetDiffusionConfig(
            action_dim = 16,
            chunk_size = 10,
            state_dim = 16,
            hidden_dim = 1024,
            resnet_model = "resnet50",  
        ),
        algorithm=BCAlgorithmConfig(
            encoder_lr = 3e-4,
            policy_lr = 3e-4,
            scheduler_step_size = 1000,
            scheduler_gamma = 0.5
        ),
        action_sequence_keys = [
            "observation.state.master_left_arm_position",
            "observation.state.master_right_arm_position",
            "observation.state.master_left_gripper_position",
            "observation.state.master_right_gripper_position",
        ],
        batch_size= 64,
        data=LeRobotUR5DataConfig(
            repo_id=None,
            use_multi_repo=True,
            repo_ids=["lerobot_dataset_pick_place_apple_on_basket"],
            # repo_ids=UR5_DATA_LIST,
            root="/mnt/dataset/toago6/workspace/code/hdf5_data_load_convert/lerobot_dataset",
            # This config lets us reload the UR5 normalization stats from the base model checkpoint.
            # Reloading normalization stats can help transfer pre-trained models to new environments.
            # See the [norm_stats.md](../docs/norm_stats.md) file for more details.
            # assets=AssetsConfig(
            #     assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            #     asset_id="ur5e",
            # ),
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. The recommended setting is True.
                prompt_from_task=True,
            ),
        ),
        # freeze_filter=pi0_config.Pi0Config(
        #     paligemma_variant="gemma_2b", action_expert_variant="gemma_300m"
        # ).get_freeze_filter(),
        # Load the pi0 base model checkpoint.
        # weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=100,
        num_epochs=500,
        save_interval = 30,
    ),

    TrainConfig(
        name="x_humanoid_resnet_diffusion_agent_xserver",
        model=XHumanoidResNetDiffusionConfig(
            action_dim = 16,
            chunk_size = 10,
            state_dim = 16,
            hidden_dim = 1024,
            resnet_model = "resnet50",  
        ),
        algorithm=BCAlgorithmConfig(
            encoder_lr = 3e-4,
            policy_lr = 3e-4,
            scheduler_step_size = 1000,
            scheduler_gamma = 0.5
        ),
        action_sequence_keys = [
            "observation.state.master_left_arm_position",
            "observation.state.master_right_arm_position",
            "observation.state.master_left_gripper_position",
            "observation.state.master_right_gripper_position",
        ],
        batch_size= 256,
        data=LeRobotUR5DataConfig(
            repo_id=None,
            use_multi_repo=True,
            repo_ids=["lerobot_dataset_pick_place_apple_on_basket"],
            # repo_ids=UR5_DATA_LIST,
            root="/home/wink/Desktop/code/offline_rl_dataset",
            # This config lets us reload the UR5 normalization stats from the base model checkpoint.
            # Reloading normalization stats can help transfer pre-trained models to new environments.
            # See the [norm_stats.md](../docs/norm_stats.md) file for more details.
            # assets=AssetsConfig(
            #     assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            #     asset_id="ur5e",
            # ),
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. The recommended setting is True.
                prompt_from_task=True,
            ),
        ),
        # freeze_filter=pi0_config.Pi0Config(
        #     paligemma_variant="gemma_2b", action_expert_variant="gemma_300m"
        # ).get_freeze_filter(),
        # Load the pi0 base model checkpoint.
        # weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=100,
        num_epochs=100,
        save_interval = 15,
    ),


    TrainConfig(
        name="x_humanoid_resnet_diffusion_agent_xserver_inference",
        model=XHumanoidResNetDiffusionConfig(
            action_dim = 16,
            chunk_size = 10,
            state_dim = 16,
            hidden_dim = 1024,
            resnet_model = "resnet50",  
            pretrained_ckpt_path = "/home/wink/Desktop/code/hierarchical_rl_in_real_world/checkpoints/aug_apple/epoch_60"
        ),
        algorithm=BCAlgorithmConfig(
            encoder_lr = 3e-4,
            policy_lr = 3e-4,
            scheduler_step_size = 1000,
            scheduler_gamma = 0.5
        ),
        action_sequence_keys = [
            "observation.state.master_left_arm_position",
            "observation.state.master_right_arm_position",
            "observation.state.master_left_gripper_position",
            "observation.state.master_right_gripper_position",
        ],
        batch_size= 64,
        data=LeRobotUR5DataConfig(
            repo_id=None,
            use_multi_repo=True,
            repo_ids=["lerobot_dataset_pick_place_apple_on_basket"],
            # repo_ids=UR5_DATA_LIST,
            root="/home/wink/Desktop/code/datasets",
            # This config lets us reload the UR5 normalization stats from the base model checkpoint.
            # Reloading normalization stats can help transfer pre-trained models to new environments.
            # See the [norm_stats.md](../docs/norm_stats.md) file for more details.
            # assets=AssetsConfig(
            #     assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            #     asset_id="ur5e",
            # ),
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. The recommended setting is True.
                prompt_from_task=True,
            ),
        ),
        # freeze_filter=pi0_config.Pi0Config(
        #     paligemma_variant="gemma_2b", action_expert_variant="gemma_300m"
        # ).get_freeze_filter(),
        # Load the pi0 base model checkpoint.
        # weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=100,
        num_epochs=500,
        save_interval = 30,
    ),

    TrainConfig(
        name="calql_agent_xserver",
        model=CalQLConfig(
            action_dim = 7,
            chunk_size = 10,
            state_dim = 7,
            resnet_model = "resnet18"
        ),
        algorithm=CalQLAlgorithmConfig(
            actor_lr = 3e-4,
            critic_lr = 3e-4,
            scheduler_step_size = 1000,
            scheduler_gamma = 0.5
        ),
        batch_size= 16,
        data=LeRobotUR5DataConfig(
            repo_id=None,
            use_multi_repo=True,
            repo_ids=["ur_wrist_pick_pink_cup_20250427", "ur_wrist_pick_pink_cup_20250428"],
            # repo_ids=UR5_DATA_LIST,
            root="/home/wink/Desktop/code/datasets",
            # This config lets us reload the UR5 normalization stats from the base model checkpoint.
            # Reloading normalization stats can help transfer pre-trained models to new environments.
            # See the [norm_stats.md](../docs/norm_stats.md) file for more details.
            # assets=AssetsConfig(
            #     assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            #     asset_id="ur5e",
            # ),
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. The recommended setting is True.
                prompt_from_task=True,
            ),
        ),
        # freeze_filter=pi0_config.Pi0Config(
        #     paligemma_variant="gemma_2b", action_expert_variant="gemma_300m"
        # ).get_freeze_filter(),
        # Load the pi0 base model checkpoint.
        # weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=50_000,
        num_epochs=500,
        save_interval = 30,
    ),  
    

    TrainConfig(
        name="x_humanoid_calql_agent_xserver",
        model=XHumanoidCalQLConfig(
            action_dim = 16,
            chunk_size = 10,
            state_dim = 16,
            resnet_model = "resnet50",
            num_diffusion_timesteps = 10,
            num_inference_steps = 10,
            pretrained_ckpt_path= "/home/wink/Desktop/code/hierarchical_rl_in_real_world/checkpoints/x_humanoid_resnet_diffusion_agent_xserver/test/20251103_000228/epoch_30"
        ),
        action_sequence_keys = [
            "observation.state.master_left_arm_position",
            "observation.state.master_right_arm_position",
            "observation.state.master_left_gripper_position",
            "observation.state.master_right_gripper_position",
        ],
        algorithm=CalQLAlgorithmConfig(
            actor_lr = 3e-4,
            critic_lr = 3e-4,
            scheduler_step_size = 1000,
            scheduler_gamma = 0.5
        ),
        batch_size= 128,
        data=LeRobotUR5DataConfig(
            repo_id=None,
            use_multi_repo=True,
            repo_ids=["lerobot_dataset_pick_place_apple_on_basket"],
            # repo_ids=UR5_DATA_LIST,
            root="/home/wink/Desktop/code/offline_rl_dataset",
            # This config lets us reload the UR5 normalization stats from the base model checkpoint.
            # Reloading normalization stats can help transfer pre-trained models to new environments.
            # See the [norm_stats.md](../docs/norm_stats.md) file for more details.
            # assets=AssetsConfig(
            #     assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            #     asset_id="ur5e",
            # ),
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. The recommended setting is True.
                prompt_from_task=True,
            ),
        ),
        # freeze_filter=pi0_config.Pi0Config(
        #     paligemma_variant="gemma_2b", action_expert_variant="gemma_300m"
        # ).get_freeze_filter(),
        # Load the pi0 base model checkpoint.
        # weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=50_000,
        num_epochs=500,
        save_interval = 30,
    ),


    TrainConfig(
        name="pi0_x_humanoid_single_apple_inference_diffusion",
        model=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            regression=False,
            action_dim = 32
        ),
        batch_size= 512,
        data=LeRobot_X_HumanoidDataConfig(
            repo_id=None,
            use_multi_repo=True,
            repo_ids=["lerobot_dataset_pick_place_apple_on_basket"],
            # repo_ids=UR5_DATA_LIST,
            root="/home/wink/Desktop/code/datasets",
            # This config lets us reload the UR5 normalization stats from the base model checkpoint.
            # Reloading normalization stats can help transfer pre-trained models to new environments.
            # See the [norm_stats.md](../docs/norm_stats.md) file for more details.
            # assets=AssetsConfig(
            #     assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            #     asset_id="ur5e",
            # ),
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. The recommended setting is True.
                prompt_from_task=True,
            ),
        ),
        # freeze_filter=pi0_config.Pi0Config(
        #     paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        # ).get_freeze_filter(),
        # Load the pi0 base model checkpoint.
        # weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        save_interval = 5000,
        pytorch_weight_path = "/home/wink/Desktop/code/hierarchical_rl_in_real_world/checkpoints/pi0_x_humanoid_single_apple/diffusion_version"
    ),

    TrainConfig(
        name="pi0_x_humanoid_single_banana_inference_diffusion",
        model=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            regression=False,
            action_dim = 32
        ),
        batch_size= 512,
        data=LeRobot_X_HumanoidDataConfig(
            repo_id=None,
            use_multi_repo=True,
            repo_ids=["lerobot_dataset_pick_place_banana_on_basket"],
            # repo_ids=UR5_DATA_LIST,
            root="/home/wink/Desktop/code/datasets",
            # This config lets us reload the UR5 normalization stats from the base model checkpoint.
            # Reloading normalization stats can help transfer pre-trained models to new environments.
            # See the [norm_stats.md](../docs/norm_stats.md) file for more details.
            # assets=AssetsConfig(
            #     assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            #     asset_id="ur5e",
            # ),
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. The recommended setting is True.
                prompt_from_task=True,
            ),
        ),
        # freeze_filter=pi0_config.Pi0Config(
        #     paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        # ).get_freeze_filter(),
        # Load the pi0 base model checkpoint.
        # weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        save_interval = 5000,
        pytorch_weight_path = "/home/wink/Desktop/code/hierarchical_rl_in_real_world/checkpoints/banana_diffusion_version"
    ),



    TrainConfig(
        name="x_humanoid_calql_agent_xserver_baidu",
        model=XHumanoidResNetDiffusionConfig(
            action_dim = 16,
            chunk_size = 10,
            state_dim = 16,
            resnet_model = "resnet50",
            num_diffusion_timesteps = 10,
            num_inference_steps = 10,
            pretrained_ckpt_path= "/mnt/dataset/toago6/workspace/code/hierarchical_rl_in_real_world/checkpoints/apple_diffusion/epoch_30"
        ),
        action_sequence_keys = [
            "observation.state.master_left_arm_position",
            "observation.state.master_right_arm_position",
            "observation.state.master_left_gripper_position",
            "observation.state.master_right_gripper_position",
        ],
        algorithm=CalQLAlgorithmConfig(
            actor_lr = 3e-4,
            critic_lr = 3e-4,
            scheduler_step_size = 1000,
            scheduler_gamma = 0.5,
            cql_alpha = 0.01
        ),
        batch_size= 512,
        data=LeRobotUR5DataConfig(
            repo_id=None,
            use_multi_repo=True,
            repo_ids=["lerobot_dataset_pick_place_apple_on_basket"],
            # repo_ids=UR5_DATA_LIST,
            root="/mnt/dataset/toago6/x_humanoid_offline_rl_dataset/",
            # This config lets us reload the UR5 normalization stats from the base model checkpoint.
            # Reloading normalization stats can help transfer pre-trained models to new environments.
            # See the [norm_stats.md](../docs/norm_stats.md) file for more details.
            # assets=AssetsConfig(
            #     assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            #     asset_id="ur5e",
            # ),
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. The recommended setting is True.
                prompt_from_task=True,
            ),
        ),
        # freeze_filter=pi0_config.Pi0Config(
        #     paligemma_variant="gemma_2b", action_expert_variant="gemma_300m"
        # ).get_freeze_filter(),
        # Load the pi0 base model checkpoint.
        # weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=50_000,
        num_epochs=500,
        save_interval = 30,
    ),

    TrainConfig(
        name="x_humanoid_resnet_diffusion_agent_xserver_baidu",
        model=XHumanoidResNetDiffusionConfig(
            action_dim = 16,
            chunk_size = 10,
            state_dim = 16,
            hidden_dim = 1024,
            resnet_model = "resnet50",  
        ),
        algorithm=BCAlgorithmConfig(
            encoder_lr = 3e-4,
            policy_lr = 3e-4,
            scheduler_step_size = 1000,
            scheduler_gamma = 0.5
        ),
        action_sequence_keys = [
            "observation.state.master_left_arm_position",
            "observation.state.master_right_arm_position",
            "observation.state.master_left_gripper_position",
            "observation.state.master_right_gripper_position",
        ],
        batch_size= 512,
        data=LeRobotUR5DataConfig(
            repo_id=None,
            use_multi_repo=True,
            repo_ids=["lerobot_dataset_pick_place_apple_on_basket"],
            # repo_ids=UR5_DATA_LIST,
            root="/mnt/dataset/toago6/x_humanoid_offline_rl_dataset",
            # This config lets us reload the UR5 normalization stats from the base model checkpoint.
            # Reloading normalization stats can help transfer pre-trained models to new environments.
            # See the [norm_stats.md](../docs/norm_stats.md) file for more details.
            # assets=AssetsConfig(
            #     assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            #     asset_id="ur5e",
            # ),
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. The recommended setting is True.
                prompt_from_task=True,
            ),
        ),
        # freeze_filter=pi0_config.Pi0Config(
        #     paligemma_variant="gemma_2b", action_expert_variant="gemma_300m"
        # ).get_freeze_filter(),
        # Load the pi0 base model checkpoint.
        # weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=100,
        num_epochs=100,
        save_interval = 10,
    ),

    TrainConfig(
        name="x_humanoid_resnet_diffusion_agent_xserver_baidu_value_dataset",
        model=XHumanoidResNetDiffusionConfig(
            action_dim = 16,
            chunk_size = 10,
            state_dim = 16,
            hidden_dim = 1024,
            resnet_model = "resnet50",  
        ),
        algorithm=BCAlgorithmConfig(
            encoder_lr = 3e-4,
            policy_lr = 3e-4,
            scheduler_step_size = 1000,
            scheduler_gamma = 0.5
        ),
        action_sequence_keys = [
            "observation.state.master_left_arm_position",
            "observation.state.master_right_arm_position",
            "observation.state.master_left_gripper_position",
            "observation.state.master_right_gripper_position",
        ],
        batch_size= 512,
        data=LeRobotUR5DataConfig(
            repo_id=None,
            use_multi_repo=True,
            repo_ids=["lerobot_dataset_pick_place_apple_on_basket"],
            # repo_ids=UR5_DATA_LIST,
            root="/mnt/dataset/toago6/workspace/code/x_humanoid_value_training",
            # This config lets us reload the UR5 normalization stats from the base model checkpoint.
            # Reloading normalization stats can help transfer pre-trained models to new environments.
            # See the [norm_stats.md](../docs/norm_stats.md) file for more details.
            # assets=AssetsConfig(
            #     assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            #     asset_id="ur5e",
            # ),
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. The recommended setting is True.
                prompt_from_task=True,
            ),
        ),
        # freeze_filter=pi0_config.Pi0Config(
        #     paligemma_variant="gemma_2b", action_expert_variant="gemma_300m"
        # ).get_freeze_filter(),
        # Load the pi0 base model checkpoint.
        # weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=100,
        num_epochs=100,
        save_interval = 5,
    ),

    TrainConfig(
        name="x_humanoid_mlp_agent_value_dataset",
        model=XHumanoidMLPConfig(
            action_dim = 16,
            chunk_size = 10,
            state_dim = 16,
            hidden_dim = 1024,
            resnet_model = "resnet50",  
        ),
        algorithm=BCAlgorithmConfig(
            encoder_lr = 3e-4,
            policy_lr = 3e-4,
            scheduler_step_size = 1000,
            scheduler_gamma = 0.5
        ),
        action_sequence_keys = [
            "observation.state.master_left_arm_position",
            "observation.state.master_right_arm_position",
            "observation.state.master_left_gripper_position",
            "observation.state.master_right_gripper_position",
        ],
        batch_size= 512,
        data=LeRobotUR5DataConfig(
            repo_id=None,
            use_multi_repo=True,
            repo_ids=["lerobot_dataset_pick_place_apple_on_basket"],
            # repo_ids=UR5_DATA_LIST,
            root="/mnt/dataset/toago6/workspace/code/x_humanoid_value_training",
            # This config lets us reload the UR5 normalization stats from the base model checkpoint.
            # Reloading normalization stats can help transfer pre-trained models to new environments.
            # See the [norm_stats.md](../docs/norm_stats.md) file for more details.
            # assets=AssetsConfig(
            #     assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            #     asset_id="ur5e",
            # ),
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. The recommended setting is True.
                prompt_from_task=True,
            ),
        ),
        # freeze_filter=pi0_config.Pi0Config(
        #     paligemma_variant="gemma_2b", action_expert_variant="gemma_300m"
        # ).get_freeze_filter(),
        # Load the pi0 base model checkpoint.
        # weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=100,
        num_epochs=100,
        save_interval = 10,
    ),

    TrainConfig(
        name="x_humanoid_resnet_diffusion_agent_xserver_baidu_inference",
        model=XHumanoidResNetDiffusionConfig(
            action_dim = 16,
            chunk_size = 10,
            state_dim = 16,
            hidden_dim = 1024,
            resnet_model = "resnet50",  
            pretrained_ckpt_path = "/mnt/dataset/toago6/workspace/code/hierarchical_rl_in_real_world/checkpoints/x_humanoid_resnet_diffusion_agent_xserver_baidu/x_humanoid_resnet_diffusion_agent_xserver_baidu/20251124_154059/epoch_10",
        ),
        algorithm=BCAlgorithmConfig(
            encoder_lr = 3e-4,
            policy_lr = 3e-4,
            scheduler_step_size = 1000,
            scheduler_gamma = 0.5
        ),
        action_sequence_keys = [
            "observation.state.master_left_arm_position",
            "observation.state.master_right_arm_position",
            "observation.state.master_left_gripper_position",
            "observation.state.master_right_gripper_position",
        ],
        batch_size= 512,
        data=LeRobotUR5DataConfig(
            repo_id=None,
            use_multi_repo=True,
            repo_ids=["lerobot_dataset_pick_place_apple_on_basket"],
            # repo_ids=UR5_DATA_LIST,
            root="/mnt/dataset/toago6/x_humanoid_offline_rl_dataset",
            # This config lets us reload the UR5 normalization stats from the base model checkpoint.
            # Reloading normalization stats can help transfer pre-trained models to new environments.
            # See the [norm_stats.md](../docs/norm_stats.md) file for more details.
            # assets=AssetsConfig(
            #     assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            #     asset_id="ur5e",
            # ),
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. The recommended setting is True.
                prompt_from_task=True,
            ),
        ),
        # freeze_filter=pi0_config.Pi0Config(
        #     paligemma_variant="gemma_2b", action_expert_variant="gemma_300m"
        # ).get_freeze_filter(),
        # Load the pi0 base model checkpoint.
        # weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=100,
        num_epochs=100,
        save_interval = 10,
    ),

    TrainConfig(
        name="pi0_x_humanoid_single_apple_inference_diffusion_baidu",
        model=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            regression=False,
            action_dim = 32,
            action_horizon = 10
        ),
        action_sequence_keys = [
            "observation.state.master_left_arm_position",
            "observation.state.master_right_arm_position",
            "observation.state.master_left_gripper_position",
            "observation.state.master_right_gripper_position",
        ],
        algorithm=CalQLAlgorithmConfig(
            actor_lr = 3e-4,
            critic_lr = 3e-4,
            scheduler_step_size = 1000,
            scheduler_gamma = 0.5
        ),
        chunk_size = 10,
        batch_size= 8,
        data=LeRobot_X_HumanoidDataConfig(
            repo_id=None,
            use_multi_repo=True,
            repo_ids=["lerobot_dataset_pick_place_apple_on_basket"],
            # repo_ids=UR5_DATA_LIST,
            root="/mnt/dataset/toago6/x_humanoid_offline_rl_dataset/",
            # This config lets us reload the UR5 normalization stats from the base model checkpoint.
            # Reloading normalization stats can help transfer pre-trained models to new environments.
            # See the [norm_stats.md](../docs/norm_stats.md) file for more details.
            # assets=AssetsConfig(
            #     assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            #     asset_id="ur5e",
            # ),
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. The recommended setting is True.
                prompt_from_task=True,
            ),
        ),
        # freeze_filter=pi0_config.Pi0Config(
        #     paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        # ).get_freeze_filter(),
        # Load the pi0 base model checkpoint.
        # weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        save_interval = 5000,
        # pytorch_weight_path = "/home/wink/Desktop/code/hierarchical_rl_in_real_world/checkpoints/pi0_x_humanoid_single_apple/diffusion_version"
    ),
    TrainConfig(
        name="x_humanoid_iql_agent_xserver_baidu",
        model=XHumanoidCalQLConfig(
            action_dim = 16,
            chunk_size = 10,
            state_dim = 16,
            resnet_model = "resnet50",
            num_diffusion_timesteps = 10,
            num_inference_steps = 10,
            pretrained_ckpt_path= "/mnt/dataset/toago6/workspace/code/hierarchical_rl_in_real_world/checkpoints/apple_diffusion/epoch_30"
        ),
        action_sequence_keys = [
            "observation.state.master_left_arm_position",
            "observation.state.master_right_arm_position",
            "observation.state.master_left_gripper_position",
            "observation.state.master_right_gripper_position",
        ],
        algorithm=IQLAlgorithmConfig(
            actor_lr = 3e-4,
            critic_lr = 3e-4,
            scheduler_step_size = 1000,
            scheduler_gamma = 0.5
        ),
        batch_size= 16,
        data=LeRobotUR5DataConfig(
            repo_id=None,
            use_multi_repo=True,
            repo_ids=["lerobot_dataset_pick_place_apple_on_basket"],
            # repo_ids=UR5_DATA_LIST,
            root="/mnt/dataset/toago6/x_humanoid_offline_rl_dataset/",
            # This config lets us reload the UR5 normalization stats from the base model checkpoint.
            # Reloading normalization stats can help transfer pre-trained models to new environments.
            # See the [norm_stats.md](../docs/norm_stats.md) file for more details.
            # assets=AssetsConfig(
            #     assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            #     asset_id="ur5e",
            # ),
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. The recommended setting is True.
                prompt_from_task=True,
            ),
        ),
        # freeze_filter=pi0_config.Pi0Config(
        #     paligemma_variant="gemma_2b", action_expert_variant="gemma_300m"
        # ).get_freeze_filter(),
        # Load the pi0 base model checkpoint.
        # weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=50_000,
        num_epochs=500,
        save_interval = 30,
    ),

    TrainConfig(
        name="x_humanoid_td_agent_xserver_baidu",
        model=XHumanoidCalQLConfig(
            action_dim = 16,
            chunk_size = 10,
            state_dim = 16,
            resnet_model = "resnet50",
            num_diffusion_timesteps = 10,
            num_inference_steps = 10,
            pretrained_ckpt_path= "/mnt/dataset/toago6/workspace/code/hierarchical_rl_in_real_world/checkpoints/apple_diffusion/epoch_30"
        ),
        action_sequence_keys = [
            "observation.state.master_left_arm_position",
            "observation.state.master_right_arm_position",
            "observation.state.master_left_gripper_position",
            "observation.state.master_right_gripper_position",
        ],
        algorithm=TDAlgorithmConfig(
            actor_lr = 3e-4,
            critic_lr = 3e-4,
            scheduler_step_size = 1000,
            scheduler_gamma = 0.5
        ),
        batch_size= 512,
        data=LeRobotUR5DataConfig(
            repo_id=None,
            use_multi_repo=True,
            repo_ids=["lerobot_dataset_pick_place_apple_on_basket"],
            # repo_ids=UR5_DATA_LIST,
            root="/mnt/dataset/toago6/x_humanoid_offline_rl_dataset/",
            # This config lets us reload the UR5 normalization stats from the base model checkpoint.
            # Reloading normalization stats can help transfer pre-trained models to new environments.
            # See the [norm_stats.md](../docs/norm_stats.md) file for more details.
            # assets=AssetsConfig(
            #     assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            #     asset_id="ur5e",
            # ),
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. The recommended setting is True.
                prompt_from_task=True,
            ),
        ),
        # freeze_filter=pi0_config.Pi0Config(
        #     paligemma_variant="gemma_2b", action_expert_variant="gemma_300m"
        # ).get_freeze_filter(),
        # Load the pi0 base model checkpoint.
        # weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=50_000,
        num_epochs=500,
        save_interval = 30,
    ),

    # TrainConfig(
    #     name="x_humanoid_resnet_consistency_policy_agent",
    #     model=XHumanoidResNetConsistencyPolicyConfig(
    #         action_dim = 16,
    #         chunk_size = 10,
    #         state_dim = 16,
    #         hidden_dim = 1024,
    #         resnet_model = "resnet50",  
    #         num_diffusion_timesteps = 20,
    #         num_inference_steps = 1,
    #         ema_decay = 0.999,
    #         loss_ctm_weight = 1
    #     ),
    #     algorithm=BCAlgorithmConfig(
    #         encoder_lr = 3e-4,
    #         policy_lr = 3e-4,
    #         scheduler_step_size = 1000,
    #         scheduler_gamma = 0.5
    #     ),
    #     action_sequence_keys = [
    #         "observation.state.master_left_arm_position",
    #         "observation.state.master_right_arm_position",
    #         "observation.state.master_left_gripper_position",
    #         "observation.state.master_right_gripper_position",
    #     ],
    #     batch_size= 512,
    #     data=LeRobotUR5DataConfig(
    #         repo_id=None,
    #         use_multi_repo=True,
    #         repo_ids=["lerobot_dataset_pick_place_apple_on_basket"],
    #         # repo_ids=UR5_DATA_LIST,
    #         root="/mnt/dataset/toago6/x_humanoid_offline_rl_dataset/",
    #         # This config lets us reload the UR5 normalization stats from the base model checkpoint.
    #         # Reloading normalization stats can help transfer pre-trained models to new environments.
    #         # See the [norm_stats.md](../docs/norm_stats.md) file for more details.
    #         # assets=AssetsConfig(
    #         #     assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
    #         #     asset_id="ur5e",
    #         # ),
    #         base_config=DataConfig(
    #             # This flag determines whether we load the prompt (i.e. the task instruction) from the
    #             # ``task`` field in the LeRobot dataset. The recommended setting is True.
    #             prompt_from_task=True,
    #         ),
    #     ),
    #     # freeze_filter=pi0_config.Pi0Config(
    #     #     paligemma_variant="gemma_2b", action_expert_variant="gemma_300m"
    #     # ).get_freeze_filter(),
    #     # Load the pi0 base model checkpoint.
    #     # weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
    #     # num_train_steps=100,
    #     num_epochs=40,
    #     save_interval = 10,
    # ),

    TrainConfig(
        name="x_humanoid_resnet_conrft_consistency_policy_agent",
            model=XHumanoidResNetConRFTConsistencyPolicyConfig(
            action_dim = 16,
            chunk_size = 10,
            state_dim = 16,
            hidden_dim = 1024,
            resnet_model = "resnet50",  
            sigma_data = 0.5,
            sigma_min = 0.002,
            sigma_max = 80.0,
            rho = 7.0,
            steps = 40,
            clip_denoised = False,
            ema_decay = 0.999,
            pretrained_ckpt_path = None,
        ),
        algorithm=BCAlgorithmConfig(
            encoder_lr = 3e-4,
            policy_lr = 3e-4,
            scheduler_step_size = 1000,
            scheduler_gamma = 0.5
        ),
        action_sequence_keys = [
            "observation.state.master_left_arm_position",
            "observation.state.master_right_arm_position",
            "observation.state.master_left_gripper_position",
            "observation.state.master_right_gripper_position",
        ],
        batch_size= 512,
        data=LeRobotUR5DataConfig(
            repo_id=None,
            use_multi_repo=True,
            repo_ids=["lerobot_dataset_pick_place_apple_on_basket"],
            # repo_ids=UR5_DATA_LIST,
            root="/mnt/dataset/toago6/x_humanoid_offline_rl_dataset/",
            # This config lets us reload the UR5 normalization stats from the base model checkpoint.
            # Reloading normalization stats can help transfer pre-trained models to new environments.
            # See the [norm_stats.md](../docs/norm_stats.md) file for more details.
            # assets=AssetsConfig(
            #     assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            #     asset_id="ur5e",
            # ),
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. The recommended setting is True.
                prompt_from_task=True,
            ),
        ),
        # freeze_filter=pi0_config.Pi0Config(
        #     paligemma_variant="gemma_2b", action_expert_variant="gemma_300m"
        # ).get_freeze_filter(),
        # Load the pi0 base model checkpoint.
        # weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=100,
        num_epochs=40,
        save_interval = 10,
    ),


    TrainConfig(
        name="x_humanoid_resnet_conrft_consistency_policy_agent_inference",
            model=XHumanoidResNetConRFTConsistencyPolicyConfig(
            action_dim = 16,
            chunk_size = 10,
            state_dim = 16,
            hidden_dim = 1024,
            resnet_model = "resnet50",  
            sigma_data = 0.5,
            sigma_min = 0.002,
            sigma_max = 80.0,
            rho = 7.0,
            steps = 40,
            clip_denoised = False,
            ema_decay = 0.999,
            pretrained_ckpt_path = "/home/wink/Desktop/code/hierarchical_rl_in_real_world/checkpoints/consistency",
        ),
        algorithm=BCAlgorithmConfig(
            encoder_lr = 3e-4,
            policy_lr = 3e-4,
            scheduler_step_size = 1000,
            scheduler_gamma = 0.5
        ),
        action_sequence_keys = [
            "observation.state.master_left_arm_position",
            "observation.state.master_right_arm_position",
            "observation.state.master_left_gripper_position",
            "observation.state.master_right_gripper_position",
        ],
        batch_size= 512,
        data=LeRobotUR5DataConfig(
            repo_id=None,
            use_multi_repo=True,
            repo_ids=["lerobot_dataset_pick_place_apple_on_basket"],
            # repo_ids=UR5_DATA_LIST,
            root="/mnt/dataset/toago6/x_humanoid_offline_rl_dataset/",
            # This config lets us reload the UR5 normalization stats from the base model checkpoint.
            # Reloading normalization stats can help transfer pre-trained models to new environments.
            # See the [norm_stats.md](../docs/norm_stats.md) file for more details.
            # assets=AssetsConfig(
            #     assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            #     asset_id="ur5e",
            # ),
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. The recommended setting is True.
                prompt_from_task=True,
            ),
        ),
        # freeze_filter=pi0_config.Pi0Config(
        #     paligemma_variant="gemma_2b", action_expert_variant="gemma_300m"
        # ).get_freeze_filter(),
        # Load the pi0 base model checkpoint.
        # weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=100,
        num_epochs=40,
        save_interval = 10,
    ),


    TrainConfig(
        name="x_humanoid_consistency_agent",
            model=XHumanoidConsistencyConfig(
            action_dim = 16,
            chunk_size = 10,
            state_dim = 16,
            hidden_dim = 1024,
            resnet_model = "resnet50",  
            sigma_data = 0.5,
            ema_decay = 0.999,
            pretrained_ckpt_path = None,
            pretrained_encoder = False,
            weight_schedule = "uniform"
        ),
        algorithm=ConsistencyAlgorithmConfig(
            encoder_lr = 3e-4,
            policy_lr = 3e-4,
            steps_per_epoch = 100,
            num_epochs = 200
        ),
        action_sequence_keys = [
            "observation.state.master_left_arm_position",
            "observation.state.master_right_arm_position",
            "observation.state.master_left_gripper_position",
            "observation.state.master_right_gripper_position",
        ],
        batch_size= 512,
        data=LeRobotUR5DataConfig(
            repo_id=None,
            use_multi_repo=True,
            repo_ids=["lerobot_dataset_pick_place_apple_on_basket"],
            # repo_ids=UR5_DATA_LIST,
            root="/mnt/dataset/toago6/workspace/code/x_humanoid_value_training",
            # This config lets us reload the UR5 normalization stats from the base model checkpoint.
            # Reloading normalization stats can help transfer pre-trained models to new environments.
            # See the [norm_stats.md](../docs/norm_stats.md) file for more details.
            # assets=AssetsConfig(
            #     assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            #     asset_id="ur5e",
            # ),
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. The recommended setting is True.
                prompt_from_task=True,
            ),
        ),
        # freeze_filter=pi0_config.Pi0Config(
        #     paligemma_variant="gemma_2b", action_expert_variant="gemma_300m"
        # ).get_freeze_filter(),
        # Load the pi0 base model checkpoint.
        # weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=100,
        num_epochs=100,
        save_interval = 10,
    ),

    TrainConfig(
        name="x_humanoid_consistency_agent_origin_dataset_truncated_snr_weight_schedule",
            model=XHumanoidConsistencyConfig(
            action_dim = 16,
            chunk_size = 10,
            state_dim = 16,
            hidden_dim = 1024,
            resnet_model = "resnet50",  
            sigma_data = 0.5,
            ema_decay = 0.999,
            pretrained_ckpt_path = None,
            pretrained_encoder = False, 
            weight_schedule = "truncated-snr"
        ),
        algorithm=ConsistencyAlgorithmConfig(
            encoder_lr = 3e-4,
            policy_lr = 3e-4,
            steps_per_epoch = 100,
            num_epochs = 200
        ),
        action_sequence_keys = [
            "observation.state.master_left_arm_position",
            "observation.state.master_right_arm_position",
            "observation.state.master_left_gripper_position",
            "observation.state.master_right_gripper_position",
        ],
        batch_size= 512,
        data=LeRobotUR5DataConfig(
            repo_id=None,
            use_multi_repo=True,
            repo_ids=["lerobot_dataset_pick_place_apple_on_basket"],
            # repo_ids=UR5_DATA_LIST,
            root="/mnt/dataset/toago6/workspace/code/hdf5_data_load_convert/lerobot_datasets",
            # This config lets us reload the UR5 normalization stats from the base model checkpoint.
            # Reloading normalization stats can help transfer pre-trained models to new environments.
            # See the [norm_stats.md](../docs/norm_stats.md) file for more details.
            # assets=AssetsConfig(
            #     assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            #     asset_id="ur5e",
            # ),
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. The recommended setting is True.
                prompt_from_task=True,
            ),
        ),
        # freeze_filter=pi0_config.Pi0Config(
        #     paligemma_variant="gemma_2b", action_expert_variant="gemma_300m"
        # ).get_freeze_filter(),
        # Load the pi0 base model checkpoint.
        # weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=100,
        num_epochs=100,
        save_interval = 10,
    ),


    TrainConfig(
        name="x_humanoid_consistency_agent_origin_dataset_uniform_weight_schedule",
            model=XHumanoidConsistencyConfig(
            action_dim = 16,
            chunk_size = 10,
            state_dim = 16,
            hidden_dim = 1024,
            resnet_model = "resnet50",  
            sigma_data = 0.5,
            ema_decay = 0.999,
            pretrained_ckpt_path = None,
            pretrained_encoder = False, 
            weight_schedule = "uniform"
        ),
        algorithm=ConsistencyAlgorithmConfig(
            encoder_lr = 3e-4,
            policy_lr = 3e-4,
            steps_per_epoch = 100,
            num_epochs = 200
        ),
        action_sequence_keys = [
            "observation.state.master_left_arm_position",
            "observation.state.master_right_arm_position",
            "observation.state.master_left_gripper_position",
            "observation.state.master_right_gripper_position",
        ],
        batch_size= 512,
        data=LeRobotUR5DataConfig(
            repo_id=None,
            use_multi_repo=True,
            repo_ids=["lerobot_dataset_pick_place_apple_on_basket"],
            # repo_ids=UR5_DATA_LIST,
            root="/mnt/dataset/toago6/workspace/code/hdf5_data_load_convert/lerobot_datasets",
            # This config lets us reload the UR5 normalization stats from the base model checkpoint.
            # Reloading normalization stats can help transfer pre-trained models to new environments.
            # See the [norm_stats.md](../docs/norm_stats.md) file for more details.
            # assets=AssetsConfig(
            #     assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            #     asset_id="ur5e",
            # ),
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. The recommended setting is True.
                prompt_from_task=True,
            ),
        ),
        # freeze_filter=pi0_config.Pi0Config(
        #     paligemma_variant="gemma_2b", action_expert_variant="gemma_300m"
        # ).get_freeze_filter(),
        # Load the pi0 base model checkpoint.
        # weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=100,
        num_epochs=100,
        save_interval = 10,
    ),

    TrainConfig(
        name="x_humanoid_consistency_agent_without_state",
            model=XHumanoidConsistencyConfig(
            action_dim = 16,
            chunk_size = 10,
            state_dim = 16,
            hidden_dim = 1024,
            resnet_model = "resnet50",  
            sigma_data = 0.5,
            ema_decay = 0.999,
            pretrained_ckpt_path = None,
            pretrained_encoder = False,
            use_state = False
        ),
        algorithm=ConsistencyAlgorithmConfig(
            encoder_lr = 3e-4,
            policy_lr = 3e-4,
            steps_per_epoch = 100,
            num_epochs = 100
        ),
        action_sequence_keys = [
            "observation.state.master_left_arm_position",
            "observation.state.master_right_arm_position",
            "observation.state.master_left_gripper_position",
            "observation.state.master_right_gripper_position",
        ],
        batch_size= 512,
        data=LeRobotUR5DataConfig(
            repo_id=None,
            use_multi_repo=True,
            repo_ids=["lerobot_dataset_pick_place_apple_on_basket"],
            # repo_ids=UR5_DATA_LIST,
            root="/mnt/dataset/toago6/workspace/code/x_humanoid_value_training",
            # This config lets us reload the UR5 normalization stats from the base model checkpoint.
            # Reloading normalization stats can help transfer pre-trained models to new environments.
            # See the [norm_stats.md](../docs/norm_stats.md) file for more details.
            # assets=AssetsConfig(
            #     assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            #     asset_id="ur5e",
            # ),
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. The recommended setting is True.
                prompt_from_task=True,
            ),
        ),
        # freeze_filter=pi0_config.Pi0Config(
        #     paligemma_variant="gemma_2b", action_expert_variant="gemma_300m"
        # ).get_freeze_filter(),
        # Load the pi0 base model checkpoint.
        # weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=100,
        num_epochs=100,
        save_interval = 5,
    ),

    TrainConfig(
        name="x_humanoid_consistency_agent_without_pretrained",
            model=XHumanoidConsistencyConfig(
            action_dim = 16,
            chunk_size = 10,
            state_dim = 16,
            hidden_dim = 1024,
            resnet_model = "resnet50",  
            sigma_data = 0.5,
            ema_decay = 0.999,
            pretrained_ckpt_path = None,
            pretrained_encoder = False
        ),
        algorithm=ConsistencyAlgorithmConfig(
            encoder_lr = 3e-4,
            policy_lr = 3e-4,
            steps_per_epoch = 100,
            num_epochs = 200
        ),
        action_sequence_keys = [
            "observation.state.master_left_arm_position",
            "observation.state.master_right_arm_position",
            "observation.state.master_left_gripper_position",
            "observation.state.master_right_gripper_position",
        ],
        batch_size= 512,
        data=LeRobotUR5DataConfig(
            repo_id=None,
            use_multi_repo=True,
            repo_ids=["lerobot_dataset_pick_place_apple_on_basket"],
            # repo_ids=UR5_DATA_LIST,
            root="/mnt/dataset/toago6/workspace/code/x_humanoid_value_training",
            # This config lets us reload the UR5 normalization stats from the base model checkpoint.
            # Reloading normalization stats can help transfer pre-trained models to new environments.
            # See the [norm_stats.md](../docs/norm_stats.md) file for more details.
            # assets=AssetsConfig(
            #     assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            #     asset_id="ur5e",
            # ),
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. The recommended setting is True.
                prompt_from_task=True,
            ),
        ),
        # freeze_filter=pi0_config.Pi0Config(
        #     paligemma_variant="gemma_2b", action_expert_variant="gemma_300m"
        # ).get_freeze_filter(),
        # Load the pi0 base model checkpoint.
        # weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=100,
        num_epochs=200,
        save_interval = 10,
    ),

    TrainConfig(
        name="x_humanoid_consistency_agent_offline_dataset",
            model=XHumanoidConsistencyConfig(
            action_dim = 16,
            chunk_size = 10,
            state_dim = 16,
            hidden_dim = 1024,
            resnet_model = "resnet50",  
            sigma_data = 0.5,
            ema_decay = 0.999,
            pretrained_ckpt_path = None,
            pretrained_encoder = False
        ),
        algorithm=ConsistencyAlgorithmConfig(
            encoder_lr = 3e-4,
            policy_lr = 3e-4,
            steps_per_epoch = 100,
            num_epochs = 100
        ),
        action_sequence_keys = [
            "observation.state.master_left_arm_position",
            "observation.state.master_right_arm_position",
            "observation.state.master_left_gripper_position",
            "observation.state.master_right_gripper_position",
        ],
        batch_size= 512,
        data=LeRobotUR5DataConfig(
            repo_id=None,
            use_multi_repo=True,
            repo_ids=["lerobot_dataset_pick_place_apple_on_basket"],
            # repo_ids=UR5_DATA_LIST,
            root="/mnt/dataset/toago6/workspace/code/x_humanoid_value_training",
            # This config lets us reload the UR5 normalization stats from the base model checkpoint.
            # Reloading normalization stats can help transfer pre-trained models to new environments.
            # See the [norm_stats.md](../docs/norm_stats.md) file for more details.
            # assets=AssetsConfig(
            #     assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            #     asset_id="ur5e",
            # ),
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. The recommended setting is True.
                prompt_from_task=True,
            ),
        ),
        # freeze_filter=pi0_config.Pi0Config(
        #     paligemma_variant="gemma_2b", action_expert_variant="gemma_300m"
        # ).get_freeze_filter(),
        # Load the pi0 base model checkpoint.
        # weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=100,
        num_epochs=100,
        save_interval = 5,
    ),

    TrainConfig(
        name="x_humanoid_consistency_agent_inference",
            model=XHumanoidConsistencyConfig(
            action_dim = 16,
            chunk_size = 10,
            state_dim = 16,
            hidden_dim = 1024,
            resnet_model = "resnet50",  
            sigma_data = 0.5,
            ema_decay = 0.999,
            pretrained_ckpt_path = "/mnt/dataset/toago6/workspace/code/hierarchical_rl_in_real_world/checkpoints/x_humanoid_consistency_agent/x_humanoid_consistency_agent_norm_with_scheduler/20251125_044026/epoch_60",
        ),
        algorithm=ConsistencyAlgorithmConfig(
            encoder_lr = 3e-4,
            policy_lr = 3e-4,
            steps_per_epoch = 100,
            num_epochs = 200
        ),
        action_sequence_keys = [
            "observation.state.master_left_arm_position",
            "observation.state.master_right_arm_position",
            "observation.state.master_left_gripper_position",
            "observation.state.master_right_gripper_position",
        ],
        batch_size= 512,
        data=LeRobotUR5DataConfig(
            repo_id=None,
            use_multi_repo=True,
            repo_ids=["lerobot_dataset_pick_place_apple_on_basket"],
            # repo_ids=UR5_DATA_LIST,
            root="/mnt/dataset/toago6/x_humanoid_offline_rl_dataset/",
            # This config lets us reload the UR5 normalization stats from the base model checkpoint.
            # Reloading normalization stats can help transfer pre-trained models to new environments.
            # See the [norm_stats.md](../docs/norm_stats.md) file for more details.
            # assets=AssetsConfig(
            #     assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            #     asset_id="ur5e",
            # ),
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. The recommended setting is True.
                prompt_from_task=True,
            ),
        ),
        # freeze_filter=pi0_config.Pi0Config(
        #     paligemma_variant="gemma_2b", action_expert_variant="gemma_300m"
        # ).get_freeze_filter(),
        # Load the pi0 base model checkpoint.
        # weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=100,
        num_epochs=200,
        save_interval = 10,
    ),

    TrainConfig(
        name="pi0_x_humanoid_single_apple_consistency_no_consistency_loss_inference",
        model=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            consistency=True,
            action_dim = 32,
            freeze_paligemma = True,
            use_consistency_loss = False
        ),
        batch_size= 512,
        data=LeRobot_X_HumanoidDataConfig(
            repo_id=None,
            use_multi_repo=True,
            repo_ids=["lerobot_dataset_pick_place_apple_on_basket"],
            # repo_ids=UR5_DATA_LIST,
            root="/mnt/dataset/toago6/workspace/code/hdf5_data_load_convert/lerobot_datasets",
            # This config lets us reload the UR5 normalization stats from the base model checkpoint.
            # Reloading normalization stats can help transfer pre-trained models to new environments.
            # See the [norm_stats.md](../docs/norm_stats.md) file for more details.
            # assets=AssetsConfig(
            #     assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            #     asset_id="ur5e",
            # ),
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. The recommended setting is True.
                prompt_from_task=True,
            ),
        ),
        # freeze_filter=pi0_config.Pi0Config(
        #     paligemma_variant="gemma_2b_lora"
        # ).get_freeze_filter(),
        # Load the pi0 base model checkpoint.
        # weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=10_000,
        save_interval = 1000,
        pytorch_weight_path = "/home/wink/Desktop/code/openpi_x_humanoid/checkpoints/consistency_no_loss"
    ),


    TrainConfig(
        name="x_humanoid_calql_consistency_agent_xserver_baidu",
        model=XHumanoidConsistencyConfig(
            action_dim = 16,
            chunk_size = 10,
            state_dim = 16,
            hidden_dim = 1024,
            resnet_model = "resnet50",  
            sigma_data = 0.5,
            ema_decay = 0.999,
            pretrained_ckpt_path = "/mnt/dataset/toago6/workspace/code/hierarchical_rl_in_real_world/checkpoints/x_humanoid_consistency_agent_origin_dataset_uniform_weight_schedule/x_humanoid_consistency_agent_origin_dataset_uniform_weight_schedule/20251207_010027/epoch_30",
            pretrained_encoder = False,
            weight_schedule = "uniform"
        ),
        action_sequence_keys = [
            "observation.state.master_left_arm_position",
            "observation.state.master_right_arm_position",
            "observation.state.master_left_gripper_position",
            "observation.state.master_right_gripper_position",
        ],
        algorithm=CalQLAlgorithmConfig(
            actor_lr = 3e-4,
            critic_lr = 3e-4,
            scheduler_step_size = 1000,
            scheduler_gamma = 0.5,
            cql_alpha = 0.01,
            pretrained_ckpt_path = "/mnt/dataset/toago6/workspace/code/hierarchical_rl_in_real_world/checkpoints/x_humanoid_calql_consistency_agent_xserver_baidu/x_humanoid_calql_consistency_agent_xserver_baidu/20251215_112957/epoch_0"
        ),
        batch_size= 512,
        data=LeRobotUR5DataConfig(
            repo_id=None,
            use_multi_repo=True,
            repo_ids=["lerobot_dataset_pick_place_apple_on_basket"],
            # repo_ids=UR5_DATA_LIST,
            root="/mnt/dataset/toago6/x_humanoid_offline_rl_dataset/",
            # This config lets us reload the UR5 normalization stats from the base model checkpoint.
            # Reloading normalization stats can help transfer pre-trained models to new environments.
            # See the [norm_stats.md](../docs/norm_stats.md) file for more details.
            # assets=AssetsConfig(
            #     assets_dir="gs://openpi-assets/checkpoints/pi0_base/assets",
            #     asset_id="ur5e",
            # ),
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. The recommended setting is True.
                prompt_from_task=True,
            ),
        ),
        # freeze_filter=pi0_config.Pi0Config(
        #     paligemma_variant="gemma_2b", action_expert_variant="gemma_300m"
        # ).get_freeze_filter(),
        # Load the pi0 base model checkpoint.
        # weight_loader=weight_loaders.CheckpointWeightLoader("gs://openpi-assets/checkpoints/pi0_base/params"),
        # num_train_steps=50_000,
        num_epochs=500,
        save_interval = 30,
    ),

    #
    # RoboArena configs.
    #
    *roboarena_config.get_roboarena_configs(),
]

if len({config.name for config in _CONFIGS}) != len(_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _CONFIGS}


def cli() -> TrainConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})



def get_config(config_name: str) -> TrainConfig:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
