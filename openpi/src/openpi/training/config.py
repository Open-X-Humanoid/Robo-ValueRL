"""See _CONFIGS for the list of available configs."""

import abc
from collections.abc import Sequence
import dataclasses
import difflib
import logging
import pathlib
# from typing import Any, Literal, Protocol, TypeAlias
from typing import Any, Literal, Protocol, TypeAlias, List, Sequence
import etils.epath as epath
import flax.nnx as nnx
import json
from typing_extensions import override
import tyro

import openpi.models.model as _model
import openpi.models.pi0_config as pi0_config
import openpi.models.tokenizer as _tokenizer

import openpi.shared.download as _download
import openpi.shared.normalize as _normalize
import openpi.training.droid_rlds_dataset as droid_rlds_dataset
# import openpi.training.misc.roboarena_config as roboarena_config
import openpi.training.optimizer as _optimizer
import openpi.training.paths as paths
import openpi.training.weight_loaders as weight_loaders
import openpi.transforms as _transforms
import openpi.policies.x_humaniod_policy as x_humanoid_policy

ModelType: TypeAlias = _model.ModelType
# Work around a tyro issue with using nnx.filterlib.Filter directly.
Filter: TypeAlias = nnx.filterlib.Filter

UR5_DATA_LIST = ['ur_5e_singleArm-gripper-4cameras_1_press_stop_button_of_control_box_20250901', 'ur_wrist_place_on_golden_tray_20250425', 'ur_single_4_1_pour_screw_into_storage_box', 'ur_wrist_pick_blue_column_block_250422', 'ur_wrist_pick_red_block_250421', 'ur_5e_singleArm-gripper-4cameras_1_hang_lab_cloth_true', 'ur_5e_singleArm-gripper-4cameras__close_toolbox', 'ur_wrist_pick_pink_cup_20250428', 'ur_wrist_clean_plate_20250427', 'ur_5e_singleArm-gripper-4cameras__open_toolbox_20250827', 'ur_wrist_pour_water_into_cup_20250428', 'ur_wrist_clean_plate_20250428', 'ur_5e_singleArm-gripper-4cameras_1_pour_oil_for_gear', 'ur_5e_singleArm-gripper-4cameras_1_pour_water_into_glass_beaker', 'ur_5e_singleArm-gripper-4cameras__open_toolbox', 'ur_wrist_clean_plate_20250425', 'ur_wrist_pour_water_into_cup_20250425', 'ur_wrist_pick_red_block_250422', 'ur_5e_singleArm-gripper-4cameras_1_press_stop_button_of_control_box', 'ur_wrist_place_on_golden_tray_20250427', 'ur_wrist_pick_bread_20250425', 'ur_wrist_pick_pink_cup_20250425', 'ur_wrist_pour_water_into_cup_20250427', 'ur_5e_singleArm-gripper-4cameras_1_assemble_valve_thread', 'ur_wrist_insert_bread_into_bag_20250425', 'ur_wrist_pick_pink_cup_20250427', 'ur_wrist_place_on_golden_tray_20250428', 'ur_5e_singleArm-gripper-4cameras_1_hang_lab_cloth_true_20250903', 'ur_wrist_place_red_block_on_tray_250422']


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
    filtered_bc_root: str | None = None
    filtered_bc_repo_ids: list[str] | None = None
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
    
    filtered_bc_root: str | None = None
    filtered_bc_repo_ids: list[str] | None = None

    openx_root: str | None = None
    openx_repo_ids: list[str] | None = None
    
    rollout_root: str | None = None
    rollout_repo_ids: list[str] | None = None
    
    # Determines how the assets will be loaded.
    assets: AssetsConfig = dataclasses.field(default_factory=AssetsConfig)
    norm_stats_path: str | None = None
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
        
        filtered_bc_root = self.filtered_bc_root if self.filtered_bc_root is not None else None
        filtered_bc_repo_ids = self.filtered_bc_repo_ids if self.filtered_bc_repo_ids is not None else None
        
        if self.norm_stats_path is not None:
            norm_stats=json.load(open(self.norm_stats_path))
            for key in norm_stats:
                norm_stats[key] = _transforms.NormStats(**norm_stats[key])
        else:
            norm_stats=None
        print("the base config norm stats is", norm_stats)
        print("use_quantile_norm is:", model_config.model_type != ModelType.PI0)
        return dataclasses.replace(
            self.base_config or DataConfig(),
            repo_id=repo_id,
            root=root,
            filtered_bc_root=filtered_bc_root,
            filtered_bc_repo_ids=filtered_bc_repo_ids,
            repo_ids=self.repo_ids,
            use_multi_repo=self.use_multi_repo,
            asset_id=asset_id,
            action_sequence_keys=action_sequence_keys,
            obs_sequence_keys=obs_sequence_keys,
            # norm_stats=self._load_norm_stats(epath.Path(self.assets.assets_dir or assets_dirs), asset_id),
            norm_stats=norm_stats,
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
class LeRobot_X_Humanoid_Multi_Camera_DataConfig(DataConfigFactory):
    # Action keys that will be used to read the action sequence from the dataset.
    # action_sequence_keys: Sequence[str] = ("action.left_arm_position", "action.right_arm_position", "action.left_gripper_position", "action.right_gripper_position")
    action_sequence_keys: Sequence[str] = ("observation.state.master_left_arm_position", "observation.state.master_right_arm_position", "observation.state.master_left_gripper_position", "observation.state.master_right_gripper_position")
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
                        "observation.image.left": "observation.image.left",
                        "observation.image.right": "observation.image.right",
                        # "discounted_value_return": "discounted_value_return",
                        # "action_judgement": "action_judgement",
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
            inputs=[x_humanoid_policy.X_Humanoid_Multi_Camera_Inputs(model_type=model_config.model_type)],
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
class LeRobot_X_Humanoid_hl_gauss_Value_RTG_Multi_Camera_DataConfig(DataConfigFactory):
    # Action keys that will be used to read the action sequence from the dataset.
    # action_sequence_keys: Sequence[str] = ("action.left_arm_position", "action.right_arm_position", "action.left_gripper_position", "action.right_gripper_position")
    action_sequence_keys: Sequence[str] = ("observation.state.master_left_arm_position", "observation.state.master_right_arm_position", "observation.state.master_left_gripper_position", "observation.state.master_right_gripper_position")
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
                        "observation.image.left": "observation.image.left",
                        "observation.image.right": "observation.image.right",
                        "discounted_value_return": "discounted_value_return",
                        "return_to_go_gt": "return_to_go",
                        "remain_time": "remain_time",
                        "frame_index": "frame_index",
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
            inputs=[x_humanoid_policy.X_Humanoid_hl_gauss_RTG_Multi_Camera_Inputs(model_type=model_config.model_type)],
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
    chunk_size: int = 50
    filtered_weight: float = 1.0
    
    
    vis_dir: str | None = None
    vis_data_dir: str | None = None
    rollout_ratio:  float = 2.0
    lr: float = 2e-4
    decay_steps: int = 50000
    annotation_dir_root: str | None = None
    annotate_total: int = 5
    annotate_count: int = 0
    
    history_length: int = 5
    
    action_sequence_keys: List[str] = dataclasses.field(
        default_factory=lambda: [            "observation.state.master_left_arm_position",
            "observation.state.master_right_arm_position",
            "observation.state.master_left_gripper_position",
            "observation.state.master_right_gripper_position",]
    )
    # Defines the model config. Some attributes (action_dim, action_horizon, and max_token_len) are shared by all models
    # -- see BaseModelConfig. Specific model implementations (e.g., Pi0Config) inherit from BaseModelConfig and may
    # define additional attributes.
    model: _model.BaseModelConfig = dataclasses.field(default_factory=pi0_config.Pi0Config)

    skip_norm_stats: bool =  False
    only_load_paligemma: bool = False
    sample_from_ratio: float = 1

    # A weight loader can optionally load (possibly partial) weights from disk after the model is initialized.
    weight_loader: weight_loaders.WeightLoader = dataclasses.field(default_factory=weight_loaders.NoOpWeightLoader)

    # Optional path to a PyTorch checkpoint to load weights from.
    pytorch_weight_path: str | None = None

    # adaptive_layer_ckpt_path
    adaptive_layer_ckpt_path: str | None = None
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
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 2
    # Number of train steps (batches) to run.
    num_train_steps: int = 50_000

    num_epochs: int = 5000
    # How often (in steps) to log training metrics.
    log_interval: int = 100
    # How often (in steps) to save checkpoints.
    save_interval: int = 10000
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


# Use `get_config` if you need to get a config by name in your code.
_CONFIGS = [
    TrainConfig(
        name="stack_all_hours_data_pretraining",
        model=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            consistency=True,
            action_dim = 16,
            freeze_paligemma = False,
            num_value_bins = 256,
            use_self_preprocess = False,
            use_consistency_loss = True,
            consistency_loss_weight = 0.1,
        ),
        sample_from_ratio=1.0,
        batch_size= 512,
        decay_steps=50000,
        data=LeRobot_X_Humanoid_Multi_Camera_DataConfig(
            repo_id=None,
            use_multi_repo=True,
            repo_ids=[
            'x_humanoid_lerobot_data_with_quality_16_01_21_processed_20260129_processed_20260202', 
            ],
            root=paths.data("block_dataset"),
            norm_stats_path=paths.NORM_STATS_PATH,
            base_config=DataConfig(
                # This flag determines whether we load the prompt (i.e. the task instruction) from the
                # ``task`` field in the LeRobot dataset. The recommended setting is True.
                prompt_from_task=True,
            ),
        ),
        num_train_steps=20_000,
        save_interval = 2000,
        pytorch_weight_path = paths.ckpt("pytorch_pi0_base")
    ),

    TrainConfig(
        name="all_value_function_training",
        model=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_100m",
            consistency=False,
            action_dim = 32,
            freeze_paligemma = True,
            num_value_bins = 256,
            use_self_preprocess = False,
            use_history_image = True,
            only_predict_remain_time = True
        ),
        sample_from_ratio=1.0,
        decay_steps = 10000,
        batch_size= 512,
        data=LeRobot_X_Humanoid_hl_gauss_Value_RTG_Multi_Camera_DataConfig(
            repo_id=None,
            use_multi_repo=True,
            repo_ids=[
            "tienkung_station_dualArm-gripper-3cameras_02_Chip_33_20260408_AM",
            ],
            # repo_ids=UR5_DATA_LIST,
            root=paths.data("chip_dataset"),

            filtered_bc_root=paths.data("for_value_training_other_tienkung_data"),
            filtered_bc_repo_ids=[
            'tienkung_pro2_dualArm-dexHand-3cameras_215_screw_6dforce_13dtactile_checked_1_6',
            ],
            openx_root=paths.data("lerobot_openx_dataset"),
            openx_repo_ids=[
                'utaustin_mutex_0.1.0_lerobot',
            ],
            rollout_root=paths.data("block_dataset"),
            rollout_repo_ids=[
"tienkung_station_dualArm-gripper-3cameras_11_lego_1_20260228",
            ],
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        num_train_steps=10_000,
        save_interval = 1000,
        pytorch_weight_path = paths.ckpt("pytorch_pi0_base")
    ),
    

    TrainConfig(
        name="online_adaptation",
        model=pi0_config.Pi0Config(
            paligemma_variant="gemma_2b",
            action_expert_variant="gemma_300m",
            consistency=True,
            action_dim = 16,
            freeze_paligemma = False,
            num_value_bins = 256,
            use_self_preprocess = False,
            use_consistency_loss = True,
            consistency_loss_weight = 0.02,
            time_consistency_loss_weight = 0.0,
            use_zero_residual_online= True,
            adaptive_layer_num_layers = 1,
            allow_online_not_q2_gate = True
        ),
        sample_from_ratio=1.0,
        batch_size= 512,
        decay_steps=50000,
        data=LeRobot_X_Humanoid_Multi_Camera_DataConfig(
            repo_id=None,
            use_multi_repo=True,
            repo_ids=[
            "tienkung_station_dualArm-gripper-3cameras_24_Chippress_5_20260423_AM"
            ],
            root=paths.data("chip_dataset"),
            filtered_bc_root=paths.data("chip_iteration_1_data"),
            filtered_bc_repo_ids=[
            'tienkung_station_dualArm-gripper-3cameras_25_Chip_11_20260402_AM',
            ],
            rollout_root=paths.data("chip_iteration_2_data"),
            rollout_repo_ids=[
                '25_intervention_20_21_merged', 
                'best_rtc_data_rollout_0518_merged', 
            ],

            norm_stats_path=paths.NORM_STATS_PATH,
            base_config=DataConfig(
                prompt_from_task=True,
            ),
        ),
        num_train_steps=20_000,
        save_interval = 2000,
        adaptive_layer_ckpt_path = None,
        pytorch_weight_path = paths.ckpt(
            "chip_ckpt",
            "newest_ckpt"
        )
    ),

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
