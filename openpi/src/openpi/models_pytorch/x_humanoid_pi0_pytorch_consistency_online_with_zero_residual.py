import copy
import logging
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
from openpi.models_pytorch.gemma_pytorch_consistency import PaliGemmaWithExpertModelWithConsistency
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing

import numpy as np

def mean_flat(tensor):
    """
    Take the mean over all non-batch dimensions.
    """
    return tensor.mean(dim=list(range(1, len(tensor.shape))))

def append_dims(x, target_dims):
    """Appends dimensions to the end of a tensor until it has target_dims dimensions."""
    dims_to_append = target_dims - x.ndim
    if dims_to_append < 0:
        raise ValueError(f"input has {x.ndim} dims but target_dims is {target_dims}, which is less")
    return x[(...,) + (None,) * dims_to_append]

def append_zero(x):
    return torch.cat([x, x.new_zeros([1])])

def get_weightings(weight_schedule, snrs, sigma_data):
    if weight_schedule == "snr":
        weightings = snrs
    elif weight_schedule == "snr+1":
        weightings = snrs + 1
    elif weight_schedule == "karras":
        weightings = snrs + 1.0 / sigma_data**2
    elif weight_schedule == "truncated-snr":
        weightings = torch.clamp(snrs, min=1.0)
    elif weight_schedule == "uniform":
        weightings = torch.ones_like(snrs)
    else:
        raise NotImplementedError()
    return weightings



class KarrasDenoiser:
    def __init__(
        self,
        action_dim,
        device,
        sigma_data: float = 0.5,
        sigma_max=80.0,
        sigma_min=0.002,
        rho=7.0,
        weight_schedule="uniform",
        steps=40,
        ts=None,
        sampler="onestep", 
        clip_denoised=False,
        action_horizon = 10
    ):
        self.action_dim = action_dim
        self.sigma_data = sigma_data
        self.sigma_max = sigma_max
        self.sigma_min = sigma_min
        self.weight_schedule = weight_schedule
        self.rho = rho
        self.action_horizon = action_horizon
        
        self.device = device
        self.sampler = sampler
        self.steps = steps
        
        # If multistep sampling time points are not specified, use defaults
        if ts is None:
            self.ts = [0, 20, 40] # Example time points
        else:
            self.ts = ts

        self.sigmas = self.get_sigmas_karras(self.steps, self.sigma_min, self.sigma_max, self.rho, self.device)
        self.clip_denoised = clip_denoised

    def get_snr(self, sigmas):
        return sigmas**-2

    def get_scalings(self, sigma):
        c_skip = self.sigma_data**2 / (sigma**2 + self.sigma_data**2)
        c_out = sigma * self.sigma_data / (sigma**2 + self.sigma_data**2) ** 0.5
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_scalings_for_boundary_condition(self, sigma):
        # Boundary condition scaling for Consistency Models
        c_skip = self.sigma_data**2 / (
            (sigma - self.sigma_min) ** 2 + self.sigma_data**2
        )
        c_out = (
            (sigma - self.sigma_min)
            * self.sigma_data
            / (sigma**2 + self.sigma_data**2) ** 0.5
        )
        c_in = 1 / (sigma**2 + self.sigma_data**2) ** 0.5
        return c_skip, c_out, c_in

    def get_sigmas_karras(self, n, sigma_min, sigma_max, rho=7.0, device="cpu"):
        ramp = torch.linspace(0, 1, n, device=device)
        min_inv_rho = sigma_min ** (1 / rho)
        max_inv_rho = sigma_max ** (1 / rho)
        sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
        return append_zero(sigmas)

    def denoise(self, model, x_t, sigmas, state):
        """
        Call the neural network and apply Consistency Scaling (c_skip, c_out, c_in).
        model(x, t, state) -> output
        """
        c_skip, c_out, c_in = [
            append_dims(x, x_t.ndim) for x in self.get_scalings_for_boundary_condition(sigmas)
        ]
        # Convert sigma into a time-embedding input suitable for the neural network (log domain)
        rescaled_t = 1000 * 0.25 * torch.log(sigmas + 1e-44)

        # Call the model: input = c_in * x_t
        model_output = model(c_in * x_t, rescaled_t, state)

        # Compute the denoised value D(x, sigma)
        denoised = c_out * model_output + c_skip * x_t
        
        if self.clip_denoised:
            denoised = denoised.clamp(-1, 1)
        return model_output, denoised

    @torch.no_grad()
    def euler_solver(self, samples, t, next_t, x0,dims):
        x = samples
        denoiser = x0
        d = (x - denoiser) / append_dims(t, dims)
        samples = x + d * append_dims(next_t - t, dims)
        return samples

    def consistency_losses(
        self,
        model,
        x_start,
        num_scales,
        state=None,
        target_model=None,
        noise=None,
    ):
        """
        Compute the Consistency Loss (Distillation)
        """
        if noise is None:
            noise = torch.randn_like(x_start)

        dims = x_start.ndim

        def denoise_fn(x, t, state=None):
            return self.denoise(model, x, t, state)[1]

        @torch.no_grad()
        def target_denoise_fn(x, t, state=None):
            return self.denoise(target_model, x, t, state)[1]

        # Euler solver used to generate pseudo-labels (Simulate the ODE step)


        # Randomly sample time step indices
        indices = torch.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        # Compute the current time step t
        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        # Compute the next time step t2 (next step after discretization)
        t2 = self.sigma_max ** (1 / self.rho) + (indices + 1) / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t2 = t2**self.rho

        x_t = x_start + noise * append_dims(t, dims)

        # Online Model denoising
        dropout_state = torch.get_rng_state()
        distiller = denoise_fn(x_t, t, state)
        
        pseudo_x0 = distiller.detach()
        x_t2 = self.euler_solver(x_t, t, t2, pseudo_x0, dims).detach() 

        torch.set_rng_state(dropout_state)
        distiller_target = target_denoise_fn(x_t2, t2, state)
        distiller_target = distiller_target.detach()

        snrs = self.get_snr(t)
        weights = get_weightings(self.weight_schedule, snrs, self.sigma_data)

        # Consistency Loss
        consistency_diffs = (distiller - distiller_target) ** 2
        consistency_loss = mean_flat(consistency_diffs) * weights

        # get the clean action prediction loss
        clean_action_prediction_loss = F.mse_loss(distiller, x_start)



        return {
            "consistency_loss": consistency_loss.mean(),
            "clean_action_prediction_loss": clean_action_prediction_loss.mean()
        }
    

    def sample_onestep(self, model, state, num=1):
        # One-step sampling: map directly from max noise sigma_max to x_0
        x_T = torch.randn((state.shape[0], self.action_horizon, self.action_dim), device=self.device) * self.sigma_max
        s_in = x_T.new_ones([x_T.shape[0]])
        # Only need to call denoise once
        return self.denoise(model, x_T, self.sigmas[0] * s_in, state)[1]


    def sample_multistep(self, model, state, steps=None):
        """
        Multistep Consistency Sampling
        Logic: Predict x0 -> Add noise to next sigma -> Repeat
        """
        # 1. Determine which steps and sigmas to use
        if steps is None:
            # Use the steps defined at initialization
            sigmas = self.sigmas
        else:
            # If a different number of steps is specified at inference time, recompute sigmas
            sigmas = self.get_sigmas_karras(steps, self.sigma_min, self.sigma_max, self.rho, self.device)

        # 2. Initialize noise x_T ~ N(0, sigma_max^2)
        # Note: the initial noise magnitude here is sigma_max
        x = torch.randn((state.shape[0], self.action_horizon, self.action_dim), device=self.device) * self.sigma_max

        # 3. Sampling loop
        # Iterate over the sigma sequence (descending: sigma_max -> ... -> 0)
        # len(sigmas) is usually steps + 1, with the last one being 0
        for i in range(len(sigmas) - 1):
            sigma_curr = sigmas[i]
            sigma_next = sigmas[i + 1]

            # Construct the batch-dimension sigma
            s_in = x.new_ones([x.shape[0]]) * sigma_curr

            # [Core step 1] Call the model to denoise, obtaining a prediction of x_start (x0)
            # Note: the Consistency Model's output is directly the estimated x0
            _, x_denoised = self.denoise(model, x, s_in, state)

            # [Core step 2] Re-add noise to the next sigma level (unless this is already the last step)
            if sigma_next > 0:
                # The logic here: since x_denoised is our estimated clean data x0,
                # the next time step x_{t_next} should be x0 + sigma_next * noise
                noise = torch.randn_like(x)
                x = x_denoised + sigma_next * noise
            else:
                # Last step, output the clean data directly
                x = x_denoised
                
        return x




def get_safe_dtype(target_dtype, device_type):
    """Get a safe dtype for the given device type."""
    if device_type == "cpu":
        # CPU doesn't support bfloat16, use float32 instead
        if target_dtype == torch.bfloat16:
            return torch.float32
        if target_dtype == torch.float64:
            return torch.float64
    return target_dtype


def create_sinusoidal_pos_embedding(
    time: torch.tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    return torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)


def sample_beta(alpha, beta, bsize, device):
    alpha_t = torch.as_tensor(alpha, dtype=torch.float32, device=device)
    beta_t = torch.as_tensor(beta, dtype=torch.float32, device=device)
    dist = torch.distributions.Beta(alpha_t, beta_t)
    return dist.sample((bsize,))


def make_att_2d_masks(pad_masks, att_masks):
    """Copied from big_vision.

    Tokens can attend to valid inputs tokens which have a cumulative mask_ar
    smaller or equal to theirs. This way `mask_ar` int[B, N] can be used to
    setup several types of attention, for example:

      [[1 1 1 1 1 1]]: pure causal attention.

      [[0 0 0 1 1 1]]: prefix-lm attention. The first 3 tokens can attend between
          themselves and the last 3 tokens have a causal attention. The first
          entry could also be a 1 without changing behaviour.

      [[1 0 1 0 1 0 0 1 0 0]]: causal attention between 4 blocks. Tokens of a
          block can attend all previous blocks and all tokens on the same block.

    Args:
      input_mask: bool[B, N] true if its part of the input, false if padding.
      mask_ar: int32[B, N] mask that's 1 where previous tokens cannot depend on
        it and 0 where it shares the same attention mask as the previous token.
    """
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    return att_2d_masks & pad_2d_masks


class AdaptiveResidualLayer(nn.Module):
    """Adaptive residual layer for fine-tuning a frozen VLA.

    Takes both the decoded action (v_t) and the raw expert hidden states
    (suffix_features) as input, fuses them with compressed vision-language
    context via Perceiver cross-attention, and predicts a tanh-bounded
    residual correction.

    Changes vs. the original dual-head version:
      - Full (bidirectional) attention on action tokens instead of causal.
      - Accepts suffix_features (expert hidden, dim=suffix_dim) alongside v_t.
      - Single residual head (alpha_max * tanh) — magnitude head removed.
    """

    def __init__(
        self,
        prefix_dim: int,
        action_dim: int,
        state_dim: int,
        suffix_dim: int = 1024,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 1,
        num_context_queries: int = 32,
        max_action_horizon: int = 50,
        alpha_max: float = 0.15,
    ):
        super().__init__()
        assert hidden_dim % num_heads == 0
        self.num_context_queries = num_context_queries
        self.num_layers = num_layers
        self.alpha_max = alpha_max

        # Perceiver cross-attention: compress prefix_features → context tokens
        self.context_queries = nn.Parameter(torch.randn(1, num_context_queries, hidden_dim) * 0.02)
        self.prefix_proj = nn.Linear(prefix_dim, hidden_dim)
        self.perceiver_attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.perceiver_norm = nn.LayerNorm(hidden_dim)

        # State projection
        self.state_proj = nn.Linear(state_dim, hidden_dim)

        # Action projection: v_t (action_dim) + suffix_features (suffix_dim)
        self.action_proj = nn.Linear(action_dim + suffix_dim, hidden_dim)

        # Learnable temporal position embedding for action tokens
        self.action_pos_embed = nn.Parameter(torch.randn(1, max_action_horizon, hidden_dim) * 0.02)

        # K layers of self-attention + FFN (full attention, no causal mask)
        self.self_attns = nn.ModuleList([
            nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
            for _ in range(num_layers)
        ])
        self.attn_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])
        self.ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 4),
                nn.SiLU(),
                nn.Linear(hidden_dim * 4, hidden_dim),
            )
            for _ in range(num_layers)
        ])
        self.ffn_norms = nn.ModuleList([nn.LayerNorm(hidden_dim) for _ in range(num_layers)])

        # Single residual head, zero-init → starts at identity (no correction)
        self.residual_head = nn.Linear(hidden_dim, action_dim)
        nn.init.zeros_(self.residual_head.weight)
        nn.init.zeros_(self.residual_head.bias)

        # Gate head: per-timestep scalar gate, init bias to -2 so sigmoid ≈ 0.12 at start
        self.gate_head = nn.Linear(hidden_dim, 1)
        nn.init.zeros_(self.gate_head.weight)
        nn.init.constant_(self.gate_head.bias, -2.0)

    def _extract_features(
        self,
        v_t: Tensor,
        suffix_features: Tensor,
        prefix_features: Tensor,
        prefix_mask: Tensor | None,
        state: Tensor,
    ) -> Tensor:
        """Extract hidden features from adapter for residual and gate heads."""
        v_t = v_t.float()
        suffix_features = suffix_features.float()
        prefix_features = prefix_features.float()
        state = state.float()
        B = v_t.shape[0]

        # 1. Perceiver cross-attention: compress prefix → K context tokens
        prefix_kv = self.prefix_proj(prefix_features)
        queries = self.context_queries.expand(B, -1, -1)

        key_padding_mask = None
        if prefix_mask is not None:
            key_padding_mask = ~prefix_mask

        ctx_tokens, _ = self.perceiver_attn(
            queries, prefix_kv, prefix_kv, key_padding_mask=key_padding_mask
        )
        ctx_tokens = self.perceiver_norm(queries + ctx_tokens)

        # 2. State token
        state_token = self.state_proj(state).unsqueeze(1)

        # 3. Action tokens: concatenate v_t and suffix_features, then project
        action_input = torch.cat([v_t, suffix_features], dim=-1)
        action_tokens = self.action_proj(action_input)
        H = action_tokens.shape[1]
        action_tokens = action_tokens + self.action_pos_embed[:, :H, :]

        # 4. Self-attention fusion (full attention, no causal mask)
        tokens = torch.cat([ctx_tokens, state_token, action_tokens], dim=1)

        for i in range(self.num_layers):
            attn_out, _ = self.self_attns[i](tokens, tokens, tokens)
            tokens = self.attn_norms[i](tokens + attn_out)
            tokens = self.ffn_norms[i](tokens + self.ffns[i](tokens))

        # 5. Extract action portion
        action_out = tokens[:, self.num_context_queries + 1 :]  # (B, H, hidden)
        return action_out

    def forward(
        self,
        v_t: Tensor,
        suffix_features: Tensor,
        prefix_features: Tensor,
        prefix_mask: Tensor | None,
        state: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """
        Args:
            v_t:              (B, action_horizon, action_dim) — decoded action from frozen VLA
            suffix_features:  (B, action_horizon, suffix_dim) — expert hidden states
            prefix_features:  (B, L_prefix, prefix_dim)
            prefix_mask:      (B, L_prefix) bool, True = valid token
            state:            (B, state_dim)
        Returns:
            corrected:  (B, H, action_dim) — v_t + gate * residual
            gate_logit: (B, H, 1) — raw logit for gate (before sigmoid)
            residual:   (B, H, action_dim) — tanh-bounded residual
        """
        action_out = self._extract_features(v_t, suffix_features, prefix_features, prefix_mask, state)

        # Residual head: tanh-bounded correction
        residual = self.alpha_max * torch.tanh(self.residual_head(action_out))  # (B, H, action_dim)

        # Gate head: per-timestep scalar
        gate_logit = self.gate_head(action_out)  # (B, H, 1)
        gate = torch.sigmoid(gate_logit)         # (B, H, 1)

        corrected = v_t.float() + gate * residual

        return corrected, gate_logit, residual


class X_Humanoid_PI0PytorchConsistencyOnlineZeroResidual(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.consistency_loss_weight = config.consistency_loss_weight
        self.time_consistency_loss_weight = config.time_consistency_loss_weight

        self.paligemma_config = paligemma_config
        self.action_expert_config = action_expert_config

        self.denoiser = KarrasDenoiser(
            action_dim = config.action_dim,
            device = "cuda"
        )

        self.paligemma_with_expert = PaliGemmaWithExpertModelWithConsistency(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        self.action_in_proj = nn.Linear(config.action_dim, action_expert_config.width)
        self.action_out_proj = nn.Linear(action_expert_config.width, config.action_dim)
        self.target_action_out_proj = copy.deepcopy(self.action_out_proj)
        for p in self.target_action_out_proj.parameters():
            p.requires_grad = False

        

            
        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(config.action_dim, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")



        # Initialize gradient checkpointing flag
        self.gradient_checkpointing_enabled = False

        msg = "transformers_replace is not installed correctly. Please install it with `uv pip install transformers==4.53.2` and `cp -r ./src/openpi/models_pytorch/transformers_replace/* .venv/lib/python3.11/site-packages/transformers/`."
        try:
            from transformers.models.siglip import check

            if not check.check_whether_transformers_replace_is_installed_correctly():
                raise ValueError(msg)
        except ImportError:
            raise ValueError(msg) from None

        if config.freeze_paligemma:
            for name, param in self.paligemma_with_expert.named_parameters():
                if "paligemma." in name:
                    param.requires_grad = False

        self.ema_decay = 0.95
        
        self.use_consistency_loss = config.use_consistency_loss
        self.use_euler_solver = config.use_euler_solver
        self.min_scales = 78            
        self.max_scales = 80           
        self.curriculum_steps = config.curriculum_steps
        self.global_step = 0
        
        self.use_state = config.use_state

        # --- Adaptive residual layer (optional) ---
        self.use_zero_residual_online = getattr(config, 'use_zero_residual_online', False)
        self.allow_online_not_q2_gate = getattr(config, 'allow_online_not_q2_gate', False)
        if self.use_zero_residual_online:
            # Quality-based loss weights: [q0_weight, q1_weight, q2_weight]
            _quality_weights = getattr(config, 'adaptive_quality_weights', [0.0, 0.1, 1.0])
            self.register_buffer(
                'adaptive_quality_weights',
                torch.tensor(_quality_weights, dtype=torch.float32),
            )
            self.adaptive_layer = AdaptiveResidualLayer(
                prefix_dim=paligemma_config.width,
                action_dim=config.action_dim,
                state_dim=config.action_dim,  # state_proj input dim matches action_dim in pi0
                suffix_dim=action_expert_config.width,
                hidden_dim=getattr(config, 'adaptive_layer_hidden_dim', 512),
                num_heads=getattr(config, 'adaptive_layer_num_heads', 8),
                num_layers=getattr(config, 'adaptive_layer_num_layers', 1),
                num_context_queries=getattr(config, 'adaptive_layer_num_queries', 32),
                alpha_max=getattr(config, 'adaptive_layer_alpha_max', 0.15),
            )

    def freeze_base_model(self):
        """Freeze all parameters except the adaptive layer.

        Call this after loading offline-RL weights before fine-tuning
        the adaptive layer.
        """
        for name, param in self.named_parameters():
            if 'adaptive_layer' not in name:
                param.requires_grad = False
        for param in self.adaptive_layer.parameters():
            param.requires_grad = True

    # Modified within the class definition
    @property
    def action_quality_query_feature(self):
        return torch.cat([
            self.action_quality_learnable_query_feature[0:1],
            self.null_query,
            self.action_quality_learnable_query_feature[1:2]
        ], dim=0)

    # This way, embed_suffix can directly use self.action_quality_query_feature[action_quality]

    def update_target_model(self):
        """
        Perform an EMA (Exponential Moving Average) update.
        This function must be called after every optimizer.step().

        Supports DDP environments: automatically handles model wrapping
        """
        with torch.no_grad():
            # Get the student model (handle DDP wrapping)
            student_model = self.paligemma_with_expert.gemma_expert
            if hasattr(student_model, 'module'):
                student_model = student_model.module

            # Get the target model (handle DDP wrapping)
            target_model = self.paligemma_with_expert.target_model
            if hasattr(target_model, 'module'):
                target_model = target_model.module

            # EMA update for the transformer backbone
            for student_p, target_p in zip(student_model.parameters(), target_model.parameters()):
                target_p.data.mul_(self.ema_decay).add_(student_p.data, alpha=1 - self.ema_decay)
            # EMA update for action_out_proj
            for student_p, target_p in zip(self.action_out_proj.parameters(), self.target_action_out_proj.parameters()):
                target_p.data.mul_(self.ema_decay).add_(student_p.data, alpha=1 - self.ema_decay)

    def apply_lora(self):
        self.paligemma_with_expert.apply_lora(self.paligemma_config, self.action_expert_config)

    def gradient_checkpointing_enable(self):
        """Enable gradient checkpointing for memory optimization."""
        self.gradient_checkpointing_enabled = True
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = True
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = True
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = True

        logging.info("Enabled gradient checkpointing for PI0Pytorch model")

    def gradient_checkpointing_disable(self):
        """Disable gradient checkpointing."""
        self.gradient_checkpointing_enabled = False
        self.paligemma_with_expert.paligemma.language_model.gradient_checkpointing = False
        self.paligemma_with_expert.paligemma.vision_tower.gradient_checkpointing = False
        self.paligemma_with_expert.gemma_expert.model.gradient_checkpointing = False

        logging.info("Disabled gradient checkpointing for PI0Pytorch model")

    def is_gradient_checkpointing_enabled(self):
        """Check if gradient checkpointing is enabled."""
        return self.gradient_checkpointing_enabled

    def _apply_checkpoint(self, func, *args, **kwargs):
        """Helper method to apply gradient checkpointing if enabled."""
        if self.gradient_checkpointing_enabled and self.training:
            return torch.utils.checkpoint.checkpoint(
                func, *args, use_reentrant=False, preserve_rng_state=False, **kwargs
            )
        return func(*args, **kwargs)

    def _prepare_attention_masks_4d(self, att_2d_masks):
        """Helper method to prepare 4D attention masks for transformer."""
        att_2d_masks_4d = att_2d_masks[:, None, :, :]
        return torch.where(att_2d_masks_4d, 0.0, -2.3819763e38)

    def _preprocess_observation(self, observation, *, train=True):
        """Helper method to preprocess observation."""
        observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
            observation.action_quality,
        )

    def sample_noise(self, shape, device):
        return torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )

    def sample_time(self, bsize, device):
        time_beta = sample_beta(1.5, 1.0, bsize, device)
        time = time_beta * 0.999 + 0.001
        return time.to(dtype=torch.float32, device=device)
    
        
    def embed_prefix(
        self, images, img_masks, lang_tokens, lang_masks
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Embed images with SigLIP and language tokens with embedding layer to prepare
        for PaliGemma transformer processing.
        """
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for img, img_mask in zip(images, img_masks, strict=True):

            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)

            img_emb = self._apply_checkpoint(image_embed_func, img)

            bsize, num_img_embs = img_emb.shape[:2]

            embs.append(img_emb)
            pad_masks.append(img_mask[:, None].expand(bsize, num_img_embs))

            # Create attention masks so that image tokens attend to each other
            att_masks += [0] * num_img_embs

        # Process language tokens
        def lang_embed_func(lang_tokens):
            lang_emb = self.paligemma_with_expert.embed_language_tokens(lang_tokens)
            lang_emb_dim = lang_emb.shape[-1]
            return lang_emb * math.sqrt(lang_emb_dim)

        lang_emb = self._apply_checkpoint(lang_embed_func, lang_tokens)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)

        # full attention between image and language inputs
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)

        # Get batch size from the first dimension of the concatenated tensors
        bsize = pad_masks.shape[0]
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        # lang_emb is returned separately so callers can use it for language conditioning
        return embs, pad_masks, att_masks, lang_emb


    def embed_suffix(self, state, noisy_actions, timestep, action_quality = None):
        """Embed state, noisy_actions, timestep to prepare for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []

        if not self.pi05:
            if self.use_state:
                if self.state_proj.weight.dtype == torch.float32:
                    state = state.to(torch.float32)

                # Embed state
                def state_proj_func(state):
                    return self.state_proj(state)

                state_emb = self._apply_checkpoint(state_proj_func, state)

                embs.append(state_emb[:, None, :])
                bsize = state_emb.shape[0]
                device = state_emb.device

                state_mask = torch.ones(bsize, 1, dtype=torch.bool, device=device)
                pad_masks.append(state_mask)

                # Set attention masks so that image and language inputs do not attend to state or actions
                att_masks += [1]

        # Embed timestep using sine-cosine positional encoding with sensitivity in the range [0, 1]
        time_emb = create_sinusoidal_pos_embedding(
            timestep, self.action_in_proj.out_features, min_period=4e-3, max_period=4.0, device=timestep.device
        )
        time_emb = time_emb.type(dtype=timestep.dtype)

        # Fuse timestep + action information using an MLP
        def action_proj_func(noisy_actions):
            return self.action_in_proj(noisy_actions)

        action_emb = self._apply_checkpoint(action_proj_func, noisy_actions)

        if not self.pi05:
            # print("the action_emb is:", action_emb.shape)
            # print("the time_emb is:", time_emb.shape)
            time_emb = time_emb[:, None, :].expand_as(action_emb)
            action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        # Apply MLP layers
            def mlp_func(action_time_emb):
                x = self.action_time_mlp_in(action_time_emb)
                x = F.silu(x)  # swish == silu
                return self.action_time_mlp_out(x)

            action_time_emb = self._apply_checkpoint(mlp_func, action_time_emb)
            adarms_cond = None
        else:
            # time MLP (for adaRMS)
            def time_mlp_func(time_emb):
                x = self.time_mlp_in(time_emb)
                x = F.silu(x)  # swish == silu
                x = self.time_mlp_out(x)
                return F.silu(x)

            time_emb = self._apply_checkpoint(time_mlp_func, time_emb)
            action_time_emb = action_emb
            adarms_cond = time_emb

        # Add to input tokens
        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=timestep.device)
        pad_masks.append(action_time_mask)

        # Set attention masks so that image, language and state inputs do not attend to action tokens
        att_masks += [1] + ([0] * (self.config.action_horizon - 1))

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond



    def model_to_target_dtype(self, dtype):
        if dtype == "bfloat16":
            dtype = torch.bfloat16
        elif dtype == "float32":
            dtype = torch.float32
        else:
            raise ValueError(f"Invalid dtype: {dtype}")
        
        self.action_in_proj.to(dtype=dtype)
        self.action_out_proj.to(dtype=dtype)
        self.return_to_go_proj.to(dtype=dtype)
        self.state_proj.to(dtype=dtype)
        self.action_time_mlp_in.to(dtype=dtype)
        self.action_time_mlp_out.to(dtype=dtype)


    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
            
        images, img_masks, lang_tokens, lang_masks, state, action_quality = self._preprocess_observation(observation, train=True)
        is_online_data = observation.is_online_data
        # print("the is online data flag is:", is_online_data)
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        
        dims = actions.ndim

        ratio = min(self.global_step / self.curriculum_steps, 1.0) # Clamp to [0, 1]
        current_num_scales = int(self.min_scales + ratio * (self.max_scales - self.min_scales))
        current_num_scales = max(current_num_scales, self.min_scales)
        self.global_step += 1
        indices = torch.randint(
            0, current_num_scales - 1, (actions.shape[0],), device=actions.device
        )

        # Compute the current time step t
        t = self.denoiser.sigma_max ** (1 / self.denoiser.rho) + indices / (current_num_scales - 1) * (
            self.denoiser.sigma_min ** (1 / self.denoiser.rho) - self.denoiser.sigma_max ** (1 / self.denoiser.rho)
        )
        t = t**self.denoiser.rho

        # Compute the next time step t2 (next step after discretization)
        t2 = self.denoiser.sigma_max ** (1 / self.denoiser.rho) + (indices + 1) / (current_num_scales - 1) * (
            self.denoiser.sigma_min ** (1 / self.denoiser.rho) - self.denoiser.sigma_max ** (1 / self.denoiser.rho)
        )
        t2 = t2**self.denoiser.rho

        # import pdb; pdb.set_trace()

        x_t = actions + noise * append_dims(t, dims)

        # Student forward: compute prefix with gradients enabled
        prefix_embs, prefix_pad_masks, prefix_att_masks, lang_emb = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)


        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, t)
            
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        # Prepare attention masks
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Apply gradient checkpointing if enabled
        def forward_func(prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond):
            (prefix_out, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return prefix_out, suffix_out

        # import pdb; pdb.set_trace()
        prefix_out, suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        # Save raw expert hidden states before quality conditioning (used by adaptive layer)
        suffix_out_raw = suffix_out

        # Apply gradient checkpointing to final action projection if enabled
        def action_out_proj_func(suffix_out):
            return self.action_out_proj(suffix_out)

        v_t = self._apply_checkpoint(action_out_proj_func, suffix_out)
        
        dropout_state = torch.get_rng_state() 
        pseudo_x0 = v_t.detach()
        # v_t is the predicted action at time t from student model
        
        x_t2 = self.denoiser.euler_solver(x_t, t, t2, pseudo_x0, dims).detach()
        

        with torch.no_grad():
            torch.set_rng_state(dropout_state)
            

            target_suffix_embs, target_suffix_pad_masks, target_suffix_att_masks, target_adarms_cond = self.embed_suffix(state, x_t2, t2)
            
            if (
                self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
                == torch.bfloat16
            ):
                target_suffix_embs = target_suffix_embs.to(dtype=torch.bfloat16)
                
            target_pad_masks = torch.cat([prefix_pad_masks, target_suffix_pad_masks], dim=1)
            target_att_masks = torch.cat([prefix_att_masks, target_suffix_att_masks], dim=1)
            target_att_2d_masks = make_att_2d_masks(target_pad_masks, target_att_masks)
            target_position_ids = torch.cumsum(target_pad_masks, dim=1) - 1
            target_att_2d_masks_4d = self._prepare_attention_masks_4d(target_att_2d_masks)
            
            (_, target_suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=target_att_2d_masks_4d,
                position_ids=target_position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, target_suffix_embs],
                use_cache=False,
                adarms_cond=[None, target_adarms_cond],
                use_target_model=True,
            )
            target_suffix_out = target_suffix_out[:, -self.config.action_horizon :]
            target_suffix_out = target_suffix_out.to(dtype=torch.float32)
            

            target_v_t = self.target_action_out_proj(target_suffix_out)
        
        
        
        
        snrs = self.denoiser.get_snr(t)
        weights = get_weightings(self.denoiser.weight_schedule, snrs, self.denoiser.sigma_data)

        consistency_diffs = (v_t - target_v_t) ** 2
        consistency_loss = self.consistency_loss_weight * (mean_flat(consistency_diffs) * weights)
        action_loss = F.mse_loss(v_t, actions, reduction='none')
        action_loss = action_loss.mean(dim=-1)
        action_loss = action_loss.mean(dim=-1)

        action_quality = action_quality[:,0]

        # Print per-quality losses (before adapter)
        with torch.no_grad():
            for q in [0, 1, 2]:
                mask = (action_quality == q)
                count = mask.sum().item()
                if count > 0:
                    q_action_loss = action_loss[mask].mean().item()
                    q_consistency_loss = (mean_flat(consistency_diffs[mask]) * weights[mask]).mean().item()
                    # print(f"[Before-Adapter][Quality={q}] count={count}, action_loss={q_action_loss:.6f}, consistency_loss={q_consistency_loss:.6f}")
                else:
                    print(f"[Before-Adapter][Quality={q}] count=0, no samples")

        # --- Adaptive residual layer with gate + zero-residual ---
        if self.use_zero_residual_online:
            prefix_features = prefix_out.detach().float()
            suffix_feats = suffix_out_raw.detach().float()

            # base_action from frozen model (detached)
            base_action = v_t.detach()  # (B, H, action_dim)

            # Student path: returns (corrected, gate_logit, residual)
            adaptive_output, gate_logit, residual = self.adaptive_layer(
                base_action, suffix_feats, prefix_features, prefix_pad_masks, state,
            )

            # Target path (no gradient) for consistency loss
            with torch.no_grad():
                target_suffix_feats = target_suffix_out.float()
                target_base_action = target_v_t.detach()
                target_adaptive_output, _, _ = self.adaptive_layer(
                    target_base_action, target_suffix_feats, prefix_features, prefix_pad_masks, state,
                )
                target_adaptive_output = target_adaptive_output.detach()

            # --- Build masks for three categories ---
            # Category 1: Offline (is_online=False) → learn loss_zero_residual, loss_gate(target=0), loss_base_keep
            # Category 2: Online + quality != 2 (is_online=True, quality!=2) → learn loss_base_keep, loss_zero_residual (keep original)
            # Category 3: Online + quality == 2 (is_online=True, quality==2) → learn loss_correction (direct action), gate→1
            is_online = is_online_data.bool()  # (B,)
            is_offline = ~is_online            # (B,)
            is_online_q2 = is_online & (action_quality == 2)       # Category 3
            is_online_not_q2 = is_online & (action_quality != 2)   # Category 2

            n_offline = is_offline.sum().item()
            n_online_q2 = is_online_q2.sum().item()
            n_online_not_q2 = is_online_not_q2.sum().item()

            # --- Correction loss (Category 3: online + quality==2): directly predict actions ---
            if n_online_q2 > 0:
                # Direct action prediction loss (not residual target)
                per_sample_corr = (adaptive_output[is_online_q2] - actions[is_online_q2]).pow(2).mean(dim=(1, 2))
                loss_correction = per_sample_corr.mean()
            else:
                loss_correction = torch.tensor(0.0, device=actions.device)

            # --- Zero-residual loss (Category 1: offline + Category 2: online not q2): residual should be 0 ---
            # When allow_online_not_q2_gate=True, Category 2 is excluded (only offline)
            if self.allow_online_not_q2_gate:
                mask_zero_residual = is_offline
            else:
                mask_zero_residual = is_offline | is_online_not_q2
            n_zero_residual = mask_zero_residual.sum().item()
            if n_zero_residual > 0:
                residual_zero = residual[mask_zero_residual]
                per_sample_zero = residual_zero.pow(2).mean(dim=(1, 2))
                loss_zero_residual = per_sample_zero.mean()
            else:
                loss_zero_residual = torch.tensor(0.0, device=actions.device)

            # --- Gate loss ---
            # Default: offline → target=0, online+q2 → target=0.7; online not q2 does NOT participate
            # When allow_online_not_q2_gate=True: online not q2 also participates with target=0
            gate_logit_flat = gate_logit.squeeze(-1)  # (B, H)
            if self.allow_online_not_q2_gate:
                mask_gate = is_offline | is_online_q2 | is_online_not_q2
            else:
                mask_gate = is_offline | is_online_q2
            n_gate = mask_gate.sum().item()
            if n_gate > 0:
                gate_target_full = torch.zeros_like(gate_logit_flat)
                gate_target_full[is_online_q2] = 0.7
                # is_offline and is_online_not_q2 stay at 0
                gate_target = gate_target_full[mask_gate]
                gate_logit_gate = gate_logit_flat[mask_gate]
                loss_gate = F.binary_cross_entropy_with_logits(gate_logit_gate, gate_target)
            else:
                loss_gate = torch.tensor(0.0, device=actions.device)

            # --- Gate sparse loss: prevent gate from overfitting ---
            loss_gate_sparse = torch.sigmoid(gate_logit).mean()

            # --- Base-keep loss (Category 1: offline + Category 2: online not q2): keep action close to base ---
            # When allow_online_not_q2_gate=True, Category 2 is excluded (only offline)
            if self.allow_online_not_q2_gate:
                mask_base_keep = is_offline
            else:
                mask_base_keep = is_offline | is_online_not_q2
            n_base_keep = mask_base_keep.sum().item()
            if n_base_keep > 0:
                per_sample_keep = (adaptive_output[mask_base_keep] - base_action[mask_base_keep]).pow(2).mean(dim=(1, 2))
                loss_base_keep = per_sample_keep.mean()
            else:
                loss_base_keep = torch.tensor(0.0, device=actions.device)

            # --- Consistency loss (all data, quality-weighted) ---
            sample_weights = self.adaptive_quality_weights[action_quality]  # [B]
            weight_sum = sample_weights.sum().clamp(min=1e-8)

            adaptive_consistency_diffs = (adaptive_output - target_adaptive_output) ** 2
            adaptive_consistency_loss = self.consistency_loss_weight * (
                (mean_flat(adaptive_consistency_diffs) * weights * sample_weights).sum() / weight_sum
            )

            # --- Combine losses ---
            lambda_corr = 1.0
            lambda_zero = 1.0
            lambda_gate = 0.5
            lambda_keep = 0.1
            lambda_gate_sparse = 0.1

            adaptive_mse_loss = (
                lambda_corr * loss_correction
                + lambda_zero * loss_zero_residual
                + lambda_gate * loss_gate
                + lambda_keep * loss_base_keep
                + lambda_gate_sparse * loss_gate_sparse
            )

            # Print debug info
            with torch.no_grad():
                gate_sigmoid = torch.sigmoid(gate_logit_flat)

                print(f"[Adapter] offline={n_offline}, online_not_q2={n_online_not_q2}, online_q2={n_online_q2}")
                print(f"  Losses: L_corr={loss_correction.item():.6f}, L_zero={loss_zero_residual.item():.6f}, "
                      f"L_gate={loss_gate.item():.6f}, L_keep={loss_base_keep.item():.6f}, L_gate_sparse={loss_gate_sparse.item():.6f}")

                if n_online_q2 > 0:
                    q2_gate = gate_sigmoid[is_online_q2]
                    print(f"  [Online Q2]  gate_mean={q2_gate.mean().item():.4f}, gate_min={q2_gate.min().item():.4f}, gate_max={q2_gate.max().item():.4f}")

                if n_online_not_q2 > 0:
                    nq2_gate = gate_sigmoid[is_online_not_q2]
                    print(f"  [Online !Q2] gate_mean={nq2_gate.mean().item():.4f}, gate_min={nq2_gate.min().item():.4f}, gate_max={nq2_gate.max().item():.4f}")

                if n_offline > 0:
                    off_gate = gate_sigmoid[is_offline]
                    print(f"  [Offline]    gate_mean={off_gate.mean().item():.4f}, gate_min={off_gate.min().item():.4f}, gate_max={off_gate.max().item():.4f}")

            return (
                adaptive_mse_loss,
                adaptive_consistency_loss,
            )

        if not self.use_consistency_loss:
            return action_loss.mean(), torch.tensor(0.0)

        return action_loss.mean(), consistency_loss.mean()





    @torch.no_grad()
    def get_action_with_return_to_go(self, device, observation, num_steps=1) -> tuple[Tensor, Tensor | None]:
        """
        Inference entry point.
        Consistency Policy defaults to single-step inference (num_steps=1).
        """
        # Consistency Policy usually only needs a single step
        actions = self.sample_actions(observation, num_steps=1)
        
        return_to_go = None
        if hasattr(self, "return_to_go_proj"):
            pass

        return actions, return_to_go

    def _prefix_forward(self, images, img_masks, lang_tokens, lang_masks):
        """Embed prefix (vision + language) and run prefix forward to get KV cache.

        Returns:
            prefix_out:       (B, L_prefix, prefix_dim)
            prefix_pad_masks: (B, L_prefix)
            past_key_values:  cached KV for suffix forward
        """
        prefix_embs, prefix_pad_masks, prefix_att_masks, _ = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks,
        )
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        (prefix_out, _), past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )
        return prefix_out, prefix_pad_masks, past_key_values

    def _suffix_forward(self, state, x_t, t_in, prefix_pad_masks, past_key_values,
                        action_quality=None):
        """Embed suffix, build masks, run suffix forward with KV cache.

        Returns:
            suffix_out_raw: (B, action_horizon, expert_width) — raw expert hidden states
            suffix_out:     (B, action_horizon, expert_width) — after optional quality conditioning
        """

        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, t_in)

        if self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            suffix_embs = suffix_embs.to(torch.bfloat16)

        # Build attention masks: suffix attends to prefix (via KV cache) + itself
        batch_size = prefix_pad_masks.shape[0]
        suffix_len = suffix_pad_masks.shape[1]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks_expanded = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks_expanded, suffix_att_2d_masks], dim=2)
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        suffix_position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=suffix_position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        suffix_out_raw = suffix_out


        return suffix_out_raw, suffix_out

    def _decode_actions(self, suffix_out_raw, suffix_out, prefix_out, prefix_pad_masks, state):
        """Project suffix_out to action space and optionally apply adaptive residual layer."""
        x0_pred = self.action_out_proj(suffix_out)
        if self.use_zero_residual_online:
            x0_pred, gate_logit, residual = self.adaptive_layer(x0_pred, suffix_out_raw, prefix_out, prefix_pad_masks, state)
            gate = torch.sigmoid(gate_logit)  # (B, H, 1)
            print(f"[Inference] gate mean={gate.mean().item():.4f}, min={gate.min().item():.4f}, max={gate.max().item():.4f}, per-step={[f'{g:.4f}' for g in gate[0, :, 0].tolist()]}")
        return x0_pred

    def _prepare_inference(self, observation):
        """Common inference setup: get device/dtype, quality, preprocess, prefix forward.

        Returns:
            prefix_out, prefix_pad_masks, past_key_values, state, action_quality, bsize, device, dtype
        """
        device = self.action_in_proj.weight.device
        dtype = self.action_in_proj.weight.dtype
        action_quality = observation.action_quality

        images, img_masks, lang_tokens, lang_masks, state, _ = self._preprocess_observation(observation, train=False)
        bsize = state.shape[0]

        prefix_out, prefix_pad_masks, past_key_values = self._prefix_forward(
            images, img_masks, lang_tokens, lang_masks,
        )
        return prefix_out, prefix_pad_masks, past_key_values, state, action_quality, bsize, device, dtype

    @torch.no_grad()
    def sample_actions(self, device, observation, num_steps=1) -> Tensor:
        """Consistency Policy single-step sampling: noise → x0 in one forward pass."""
        prefix_out, prefix_pad_masks, past_key_values, state, action_quality, bsize, device, dtype = \
            self._prepare_inference(observation)

        sigma_max = self.denoiser.sigma_max
        x_t = torch.randn((bsize, self.config.action_horizon, self.config.action_dim), device=device, dtype=dtype) * sigma_max
        t_in = torch.full((bsize,), sigma_max, device=device, dtype=dtype)

        suffix_out_raw, suffix_out = self._suffix_forward(
            state, x_t, t_in, prefix_pad_masks, past_key_values, action_quality,
        )
        return self._decode_actions(suffix_out_raw, suffix_out, prefix_out, prefix_pad_masks, state)

    def guided_sample_actions(
        self,
        device,
        observation,
        a_prev: np.ndarray,
        d: int,
        s: int,
        num_steps: int = 1,
        guidance_scale: float = 1.0,
    ) -> Tensor:
        """RTC gradient-guided sampling: 1 forward + 1 backward (VJP) to align
        the first *d* steps with *a_prev* via gradient adjustment.
        """
        prefix_out, prefix_pad_masks, past_key_values, state, action_quality, bsize, device, dtype = \
            self._prepare_inference(observation)

        # Convert a_prev
        if isinstance(a_prev, np.ndarray):
            a_prev_tensor = torch.from_numpy(a_prev).to(device=device, dtype=dtype).unsqueeze(0)
        else:
            a_prev_tensor = a_prev.to(device=device, dtype=dtype)
        frozen_steps = min(d, a_prev_tensor.shape[1])

        # Noisy input (requires_grad for VJP)
        sigma_max = self.denoiser.sigma_max
        x_t = torch.randn(
            (bsize, self.config.action_horizon, self.config.action_dim),
            device=device, dtype=dtype, requires_grad=True,
        ) * sigma_max
        t_in = torch.full((bsize,), sigma_max, device=device, dtype=dtype)

        # Forward (suffix needs grad flow through x_t → suffix_embs → x0_pred)

        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, t_in)

        if self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            suffix_embs = suffix_embs.to(torch.bfloat16)

        batch_size = prefix_pad_masks.shape[0]
        suffix_len = suffix_pad_masks.shape[1]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks_expanded = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks_expanded, suffix_att_2d_masks], dim=2)
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        suffix_position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"
        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=suffix_position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1][:, -self.config.action_horizon :].to(dtype=torch.float32)
        x0_pred = self.action_out_proj(suffix_out)

        if self.use_zero_residual_online:
            x0_pred, gate_logit, _ = self.adaptive_layer(x0_pred, suffix_out, prefix_out, prefix_pad_masks, state)
            gate = torch.sigmoid(gate_logit)
            print(f"[Inference-Guided] gate mean={gate.mean().item():.4f}, min={gate.min().item():.4f}, max={gate.max().item():.4f}, per-step={[f'{g:.4f}' for g in gate[0, :, 0].tolist()]}")

        # Gradient guidance
        if frozen_steps > 0:
            guidance_loss = F.mse_loss(x0_pred[:, :frozen_steps], a_prev_tensor[:, :frozen_steps])
            try:
                grad_x_t = torch.autograd.grad(guidance_loss, x_t, create_graph=False, retain_graph=False)[0]
                with torch.no_grad():
                    H = x0_pred.shape[1]
                    decay_weight = torch.exp(-0.1 * torch.arange(H, device=device, dtype=dtype))
                    decay_weight = (decay_weight / decay_weight.max()).view(1, -1, 1)
                    x0_pred = x0_pred - guidance_scale * grad_x_t * 0.01 * decay_weight
                    x0_pred[:, :frozen_steps] = a_prev_tensor[:, :frozen_steps]
                return x0_pred
            except RuntimeError:
                with torch.no_grad():
                    x0_pred = x0_pred.clone()
                    x0_pred[:, :frozen_steps] = a_prev_tensor[:, :frozen_steps]
                return x0_pred
        else:
            with torch.no_grad():
                return x0_pred.detach()

    def denoise_step(self, state, prefix_out, prefix_pad_masks, past_key_values, x_t, timestep):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_out_raw, suffix_out = self._suffix_forward(
            state, x_t, timestep, prefix_pad_masks, past_key_values,
        )
        return self._decode_actions(suffix_out_raw, suffix_out, prefix_out, prefix_pad_masks, state)

    @torch.no_grad()
    def guided_noise_sample_actions(
        self,
        device,
        observation,
        a_prev: np.ndarray,
        d: int,
        noise_mix_decay: float = 0.01,
    ) -> Tensor:
        """Noise-guided sampling: inject a_prev information into the initial noise,
        then run consistency sampling with hard constraint on the first *d* steps.
        """
        prefix_out, prefix_pad_masks, past_key_values, state, action_quality, bsize, device, dtype = \
            self._prepare_inference(observation)

        # Convert and pad a_prev
        if isinstance(a_prev, np.ndarray):
            a_prev_tensor = torch.from_numpy(a_prev).to(device=device, dtype=dtype)
            if a_prev_tensor.ndim == 2:
                a_prev_tensor = a_prev_tensor.unsqueeze(0)
        else:
            a_prev_tensor = a_prev.to(device=device, dtype=dtype)
            if a_prev_tensor.ndim == 2:
                a_prev_tensor = a_prev_tensor.unsqueeze(0)

        if a_prev_tensor.shape[0] == 1 and bsize > 1:
            a_prev_tensor = a_prev_tensor.expand(bsize, -1, -1)

        if a_prev_tensor.shape[1] < self.config.action_horizon:
            pad_len = self.config.action_horizon - a_prev_tensor.shape[1]
            a_prev_tensor = F.pad(a_prev_tensor, (0, 0, 0, pad_len), mode='constant', value=0)

        action_horizon = a_prev_tensor.shape[1]
        frozen_steps = min(d, action_horizon)

        # Build mixed noise: guidance noise for frozen steps, random for the rest,
        # with exponential decay blending in-between
        sigma_max = self.denoiser.sigma_max

        x_t_guidance = a_prev_tensor + torch.randn_like(a_prev_tensor) * sigma_max
        x_t_random = torch.randn((bsize, action_horizon, self.config.action_dim), device=device, dtype=dtype) * sigma_max

        if frozen_steps > 0:
            x_t = x_t_guidance.clone()
            x_t[:, frozen_steps:] = x_t_random[:, frozen_steps:]
        else:
            x_t = x_t_random

        # Exponential decay blending for steps beyond frozen region
        if frozen_steps < action_horizon:
            t_offsets = torch.arange(action_horizon - frozen_steps, device=device, dtype=dtype)
            lambdas = torch.exp(-noise_mix_decay * t_offsets).view(1, -1, 1)
            x_t[:, frozen_steps:] = lambdas * x_t_guidance[:, frozen_steps:] + (1 - lambdas) * x_t_random[:, frozen_steps:]

        t_in = torch.full((bsize,), sigma_max, device=device, dtype=dtype)

        suffix_out_raw, suffix_out = self._suffix_forward(
            state, x_t, t_in, prefix_pad_masks, past_key_values, action_quality,
        )
        x0_pred = self._decode_actions(suffix_out_raw, suffix_out, prefix_out, prefix_pad_masks, state)

        # Hard constraint: replace first d steps with a_prev
        if frozen_steps > 0:
            result = x0_pred.clone()
            result[:, :frozen_steps] = a_prev_tensor[:, :frozen_steps]
            return result
        return x0_pred