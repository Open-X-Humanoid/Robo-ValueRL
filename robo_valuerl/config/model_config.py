import dataclasses
from agents.resnet_mlp_agent import ResNetMLPAgent
from agents.resnet_gmm_agent import ResNetGMMAgent
from agents.calql_agent import CalQLAgent
from agents.base_agent import BaseAgent
from agents.resnet_diffusion_agent import ResNetDiffusionAgent
from agents.x_humanoid_resnet_diffusion_agent import XHumanoidResNetDiffusionAgent
from agents.calql_x_humanoid_resnet_diffusion_agent import XHumanoidCalQLAgent
from agents.x_humanoid_resnet_conrft_consistency_policy import XHumanoidResNetConRFTConsistencyPolicyAgent
from agents.x_humanoid_consistency import XHumanoidConsistencyAgent
from agents.x_humanoid_mlp_agent import XHumanoidMLPAgent

@dataclasses.dataclass(frozen=True)
class ResNetMLPConfig:
    dtype: str = "bfloat16"
    action_dim: int = 7
    chunk_size: int = 10
    state_dim: int = 7
    max_token_len: int = None  # type: ignore
    resnet_model: str = "resnet18"
    hidden_dim: int = 1024
    model_type:BaseAgent = ResNetMLPAgent
    pretrained_ckpt_path: str = None


@dataclasses.dataclass(frozen=True)
class XHumanoidMLPConfig:
    dtype: str = "bfloat16"
    action_dim: int = 16
    chunk_size: int = 10
    state_dim: int = 16
    resnet_model: str = "resnet50"
    hidden_dim: int = 1024
    pretrained_ckpt_path: str = None
    model_type:BaseAgent = XHumanoidMLPAgent


@dataclasses.dataclass(frozen=True)
class ResNetGMMConfig:
    dtype: str = "bfloat16"
    action_dim: int = 7
    chunk_size: int = 10
    state_dim: int = 7
    num_components: int = 5
    resnet_model: str = "resnet18"
    pretrained_ckpt_path: str = None
    model_type:BaseAgent = ResNetGMMAgent

@dataclasses.dataclass(frozen=True)
class ResNetDiffusionConfig:
    dtype: str = "bfloat16"
    action_dim: int = 7
    chunk_size: int = 10
    state_dim: int = 7
    resnet_model: str = "resnet18"
    hidden_dim: int = 1024
    pretrained_ckpt_path: str = None
    num_diffusion_timesteps: int = 10
    num_inference_steps: int = 10
    model_type:BaseAgent = ResNetDiffusionAgent


@dataclasses.dataclass(frozen=True)
class XHumanoidResNetDiffusionConfig:
    dtype: str = "bfloat16"
    action_dim: int = 7
    chunk_size: int = 10
    state_dim: int = 7
    resnet_model: str = "resnet18"
    hidden_dim: int = 1024
    pretrained_ckpt_path: str = None
    num_diffusion_timesteps: int = 10
    num_inference_steps: int = 10
    model_type:BaseAgent = XHumanoidResNetDiffusionAgent

@dataclasses.dataclass(frozen=True)
class CalQLConfig:
    dtype: str = "bfloat16"
    action_dim: int = 7
    chunk_size: int = 10
    state_dim: int = 7
    resnet_model: str = "resnet18"
    hidden_dim: int = 1024
    pretrained_ckpt_path: str = None
    model_type: BaseAgent = CalQLAgent

@dataclasses.dataclass(frozen=True)
class XHumanoidCalQLConfig:
    dtype: str = "bfloat16"
    action_dim: int = 16
    chunk_size: int = 10
    state_dim: int = 16
    resnet_model: str = "resnet50"
    hidden_dim: int = 1024
    pretrained_ckpt_path: str = None
    num_diffusion_timesteps: int = 10
    num_inference_steps: int = 10
    model_type: BaseAgent = XHumanoidCalQLAgent


# @dataclasses.dataclass(frozen=True)
# class XHumanoidResNetConsistencyPolicyConfig:
#     dtype: str = "bfloat16"
#     action_dim: int = 7
#     chunk_size: int = 10
#     state_dim: int = 7
#     resnet_model: str = "resnet18"
#     hidden_dim: int = 1024
#     pretrained_ckpt_path: str = None
#     num_diffusion_timesteps: int = 20
#     num_inference_steps: int = 1
#     ema_decay: float = 0.999
#     loss_ctm_weight: float = 1.0
#     loss_dsm_weight: float = 1.0
#     model_type:BaseAgent = XHumanoidResNetConsistencyPolicyAgent
    
@dataclasses.dataclass(frozen=True)
class XHumanoidResNetConRFTConsistencyPolicyConfig:
    dtype: str = "bfloat16"
    action_dim: int = 7
    chunk_size: int = 10
    state_dim: int = 7
    resnet_model: str = "resnet18"
    hidden_dim: int = 1024
    pretrained_ckpt_path: str = None
    num_diffusion_timesteps: int = 20
    num_inference_steps: int = 1
    ema_decay: float = 0.999
    sigma_data: float = 0.5
    sigma_min: float = 0.002
    sigma_max: float = 80.0
    rho: float = 7.0
    steps: int = 40
    clip_denoised: bool = True
    model_type:BaseAgent = XHumanoidResNetConRFTConsistencyPolicyAgent


@dataclasses.dataclass(frozen=True)
class XHumanoidConsistencyConfig:
    dtype: str = "bfloat16"
    action_dim: int = 16
    chunk_size: int = 10
    state_dim: int = 16
    resnet_model: str = "resnet50"
    hidden_dim: int = 1024
    pretrained_ckpt_path: str = None
    
    sigma_data: float = 0.5
    ema_decay: float = 0.999
    
    pretrained_encoder: bool = False
    use_state: bool = True
    model_type: BaseAgent = XHumanoidConsistencyAgent
    weight_schedule: str = "uniform"
    
    