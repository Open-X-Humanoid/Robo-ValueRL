"""See _CONFIGS for the list of available configs."""

import dataclasses
import difflib
import pathlib
from typing import Any, TypeAlias
import flax.nnx as nnx
import tyro
Filter: TypeAlias = nnx.filterlib.Filter


@dataclasses.dataclass(frozen=True)
class ValueEstimationConfig:
    # Name of the config. Must be unique. Will be used to reference this config.
    name: tyro.conf.Suppress[str]
    # Project name.
    project_name: str = "openpi"
    # Experiment name. Will be used to name the metadata and checkpoint directories.
    exp_name: str = tyro.MISSING

    model: str = "Qwen/Qwen2.5-VL-3B-Instruct"

    # Base directory for checkpoints.
    checkpoint_base_dir: str = "./checkpoints"

    trained_ckpt_dir:str | None = None

    # Random seed that will be used by random generators during training.
    seed: int = 42
    use_lora: bool = False
    
    # Global batch size.
    batch_size: int = 32
    sample_from_ratio: float = 1.0
    # Number of workers to use for the data loader. Increasing this number will speed up data loading but
    # will increase memory and CPU usage.
    num_workers: int = 8
    # Number of train steps (batches) to run.
    num_train_steps: int = 50_000
    lr: float = 3e-4
    
    num_epochs: int = 100
    dataset_root: str = "/mnt/dataset/toago6/x_humanoid_value_training"

    # How often (in steps) to log training metrics.
    log_interval: int = 10
    # How often (in steps) to save checkpoints.
    save_interval: int = 20
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
_VALUE_ESTIMATION_CONFIGS = [
  
    ValueEstimationConfig(
        name="qwen2_5_vl_value_function",
        model="Qwen/Qwen2.5-VL-3B-Instruct",
        batch_size=32,
        num_train_steps=50_000,
        use_lora = True,
        save_interval=20,
        num_epochs = 500
    ),
    ValueEstimationConfig(
        name="qwen2_5_vl_value_function_without_lora",
        model="Qwen/Qwen2.5-VL-3B-Instruct",
        batch_size=16,
        num_train_steps=50_000,
        use_lora = False,
        save_interval=20,
        num_epochs = 500
    ),
    ValueEstimationConfig(
        name="qwen2_5_vl_value_function_without_discount",
        model="Qwen/Qwen2.5-VL-3B-Instruct",
        batch_size=32,
        num_train_steps=50_000,
        use_lora = True,
        save_interval=20,
        num_epochs = 200,
        dataset_root = "/mnt/dataset/toago6/x_humanoid_value_training_without_discount"
    ),
    ValueEstimationConfig(
        name="qwen2_5_vl_text_value_function",
        model="Qwen/Qwen2.5-VL-3B-Instruct",
        batch_size=32,
        num_train_steps=50_000,
        use_lora = True,
        save_interval=5,
        num_epochs = 20,
        dataset_root = "/mnt/dataset/toago6/x_humanoid_value_training"
    )
    
]

if len({config.name for config in _VALUE_ESTIMATION_CONFIGS}) != len(_VALUE_ESTIMATION_CONFIGS):
    raise ValueError("Config names must be unique.")
_CONFIGS_DICT = {config.name: config for config in _VALUE_ESTIMATION_CONFIGS}


def cli() -> ValueEstimationConfig:
    return tyro.extras.overridable_config_cli({k: (k, v) for k, v in _CONFIGS_DICT.items()})



def get_config(config_name: str) -> _VALUE_ESTIMATION_CONFIGS:
    """Get a config by name."""
    if config_name not in _CONFIGS_DICT:
        closest = difflib.get_close_matches(config_name, _CONFIGS_DICT.keys(), n=1, cutoff=0.0)
        closest_str = f" Did you mean '{closest[0]}'? " if closest else ""
        raise ValueError(f"Config '{config_name}' not found.{closest_str}")

    return _CONFIGS_DICT[config_name]
