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
        
        # If multistep sampling timepoints are not specified, use defaults
        if ts is None:
            self.ts = [0, 20, 40] # example timepoints
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
        # Convert sigma into a time-embedding input suitable for the network (log domain)
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


        # Randomly sample timestep indices
        indices = torch.randint(
            0, num_scales - 1, (x_start.shape[0],), device=x_start.device
        )

        # Compute the current timestep t
        t = self.sigma_max ** (1 / self.rho) + indices / (num_scales - 1) * (
            self.sigma_min ** (1 / self.rho) - self.sigma_max ** (1 / self.rho)
        )
        t = t**self.rho

        # Compute the next timestep t2 (the next step after discretization)
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
        # Note: the starting noise magnitude here is sigma_max
        x = torch.randn((state.shape[0], self.action_horizon, self.action_dim), device=self.device) * self.sigma_max

        # 3. Sampling loop
        # Iterate over the sigma sequence (largest to smallest: sigma_max -> ... -> 0)
        # len(sigmas) is usually steps + 1, with the last one being 0
        for i in range(len(sigmas) - 1):
            sigma_curr = sigmas[i]
            sigma_next = sigmas[i + 1]

            # Build the batch-dimension sigma
            s_in = x.new_ones([x.shape[0]]) * sigma_curr

            # [Core step 1] Call the model to denoise, obtaining a prediction of x_start (x0)
            # Note: the Consistency Model's output is directly the estimated x0
            _, x_denoised = self.denoise(model, x, s_in, state)

            # [Core step 2] Re-add noise to reach the next sigma level (unless this is the last step)
            if sigma_next > 0:
                # The logic here: since x_denoised is our estimate of the clean data x0,
                # the next timestep x_{t_next} should be x0 + sigma_next * noise
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


class X_Humanoid_PI0PytorchConsistency(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.consistency_loss_weight = config.consistency_loss_weight

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

        if config.use_action_quality_filter:
            self.action_quality_learnable_query_feature = nn.Parameter(torch.randn(2, action_expert_config.width))
            self.register_buffer('null_query', torch.zeros(1, action_expert_config.width))
        

            
        if self.pi05:
            self.time_mlp_in = nn.Linear(action_expert_config.width, action_expert_config.width)
            self.time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)
        else:
            self.state_proj = nn.Linear(config.action_dim, action_expert_config.width)
            self.action_time_mlp_in = nn.Linear(2 * action_expert_config.width, action_expert_config.width)
            self.action_time_mlp_out = nn.Linear(action_expert_config.width, action_expert_config.width)

        torch.set_float32_matmul_precision("high")
        self.sample_actions = torch.compile(self.sample_actions, mode="max-autotune")


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
        self.min_scales = 2            
        self.max_scales = 80           
        self.curriculum_steps = config.curriculum_steps
        self.global_step = 0
        
        self.use_state = config.use_state
    

    # Modified within the class definition
    @property
    def action_quality_query_feature(self):
        return torch.cat([
            self.action_quality_learnable_query_feature[0:1], 
            self.null_query, 
            self.action_quality_learnable_query_feature[1:2]
        ], dim=0)

    # This way, embed_suffix can simply use self.action_quality_query_feature[action_quality]

    def update_target_model(self):
        """
        Perform an EMA (Exponential Moving Average) update.
        This function must be called after every optimizer.step().

        Supports DDP environments: automatically handles model wrapping
        """
        with torch.no_grad():
            # Get the student model (handling DDP wrapping)
            student_model = self.paligemma_with_expert.gemma_expert
            if hasattr(student_model, 'module'):
                student_model = student_model.module

            # Get the target model (handling DDP wrapping)
            target_model = self.paligemma_with_expert.target_model
            if hasattr(target_model, 'module'):
                target_model = target_model.module

            # EMA update
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

        return embs, pad_masks, att_masks


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
        if self.config.use_action_quality_filter:
            self.action_quality_query_feature.to(dtype=dtype)

    def forward(self, observation, actions, noise=None, time=None) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""
        if self.config.use_action_quality_filter:
            action_quality = observation.action_quality
        else:
            action_quality = None
            
        images, img_masks, lang_tokens, lang_masks, state, _ = self._preprocess_observation(observation, train=True)

        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)
        
        dims = actions.ndim

        ratio = min(self.global_step / self.curriculum_steps, 1.0) # clamp to [0, 1]
        current_num_scales = int(self.min_scales + ratio * (self.max_scales - self.min_scales))
        current_num_scales = max(current_num_scales, self.min_scales)
        self.global_step += 1
        indices = torch.randint(
            0, current_num_scales - 1, (actions.shape[0],), device=actions.device
        )

        # Compute the current timestep t
        t = self.denoiser.sigma_max ** (1 / self.denoiser.rho) + indices / (current_num_scales - 1) * (
            self.denoiser.sigma_min ** (1 / self.denoiser.rho) - self.denoiser.sigma_max ** (1 / self.denoiser.rho)
        )
        t = t**self.denoiser.rho

        # Compute the next timestep t2 (the next step after discretization)
        t2 = self.denoiser.sigma_max ** (1 / self.denoiser.rho) + (indices + 1) / (current_num_scales - 1) * (
            self.denoiser.sigma_min ** (1 / self.denoiser.rho) - self.denoiser.sigma_max ** (1 / self.denoiser.rho)
        )
        t2 = t2**self.denoiser.rho

        # import pdb; pdb.set_trace()

        x_t = actions + noise * append_dims(t, dims)

        # Student forward: compute prefix with gradients enabled
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)

        if self.config.use_action_quality_filter:
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, t, action_quality)
        else:
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
            (_, suffix_out), _ = self.paligemma_with_expert.forward(
                attention_mask=att_2d_masks_4d,
                position_ids=position_ids,
                past_key_values=None,
                inputs_embeds=[prefix_embs, suffix_embs],
                use_cache=False,
                adarms_cond=[None, adarms_cond],
            )
            return suffix_out

        # import pdb; pdb.set_trace()
        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )
        
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)

        if self.config.use_action_quality_filter:
            # print("the action_quality is:", action_quality.shape)
            # print("the self.action_quality_query_feature is:", self.action_quality_query_feature.shape)
            quality_condition_feature = self.action_quality_query_feature[action_quality]
            
            suffix_out = suffix_out + quality_condition_feature

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
            
            if self.config.use_action_quality_filter:
                target_suffix_embs, target_suffix_pad_masks, target_suffix_att_masks, target_adarms_cond = self.embed_suffix(state, x_t2, t2, action_quality)
            else:
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
            
            if self.config.use_action_quality_filter:
                quality_condition_feature = self.action_quality_query_feature[action_quality]
                target_suffix_out = target_suffix_out + quality_condition_feature
                
            target_v_t = self.target_action_out_proj(target_suffix_out)
        
        snrs = self.denoiser.get_snr(t)
        weights = get_weightings(self.denoiser.weight_schedule, snrs, self.denoiser.sigma_data)
        # import pdb; pdb.set_trace()
    
        consistency_diffs = (v_t - target_v_t) ** 2
        consistency_loss = self.consistency_loss_weight * (mean_flat(consistency_diffs) * weights)
        action_loss = F.mse_loss(v_t, actions, reduction='none')
        action_loss = action_loss.mean(dim=-1)
        action_loss = action_loss.mean(dim=-1)
        
        

        
        if not self.use_consistency_loss:
            return action_loss.mean(), torch.tensor(0.0)
        

        return action_loss.mean(), consistency_loss.mean()





    @torch.no_grad()
    def get_action_with_return_to_go(self, device, observation, num_steps=1) -> tuple[Tensor, Tensor | None]:
        """
        Inference entry point.
        Consistency Policy defaults to single-step inference (num_steps=1).
        """
        # Consistency Policy usually only needs one step
        actions = self.sample_actions(observation, num_steps=1)
        
        return_to_go = None
        if hasattr(self, "return_to_go_proj"):
            pass

        return actions, return_to_go

    @torch.no_grad()
    def sample_actions(self, device, observation, num_steps=1) -> Tensor:
        """
        Consistency Policy single-step sampling implementation.
        Maps directly from Gaussian noise (scaled by sigma_max) to the clean action x0.
        Retains the KV-Cache (Prefix/Suffix separation) to maximize inference speed.
        """
        device = self.action_in_proj.weight.device
        dtype = self.action_in_proj.weight.dtype
        
        if self.config.use_action_quality_filter:
            action_quality = observation.action_quality
        else:
            action_quality = None
            
        # 1. Preprocess Observation

        images, img_masks, lang_tokens, lang_masks, state, _  = self._preprocess_observation(observation, train=False)

        # print("the images is:", images[0].shape)
        bsize = state.shape[0]

        # 2. Compute Prefix Embedding (Vision + Language)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)

        # Build the Prefix's Attention Masks
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # 3. Run Prefix Forward to obtain Past Key Values (KV Cache)
        # This is the most expensive part; in single-step sampling it only needs to run once
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

        # Prefix Forward: input inputs_embeds=[prefix_embs, None]
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True, # enable Cache
        )

        # 4. Prepare single-step input
        # Consistency Policy: Start from noise * sigma_max, Time condition = sigma_max
        sigma_max = self.denoiser.sigma_max

        # Initial noise x_T ~ N(0, sigma_max^2)
        x_t = torch.randn((bsize, self.config.action_horizon, self.config.action_dim), device=device, dtype=dtype) * sigma_max

        # Timestep embedding: t = sigma_max
        t_in = torch.full((bsize,), sigma_max, device=device, dtype=dtype)

        # --- Single Denoise Step ---

        # 5. Embed Suffix (State + Noisy Action + Time)
        if self.config.use_action_quality_filter:
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, t_in, action_quality)
        else:
            suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, t_in)
        
        if self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            suffix_embs = suffix_embs.to(torch.bfloat16)

        # 6. Build Suffix-related Masks (concatenated with Prefix info)
        batch_size = prefix_pad_masks.shape[0]
        suffix_len = suffix_pad_masks.shape[1]
        prefix_len = prefix_pad_masks.shape[1]

        # Allow Suffix to attend to Prefix (Causal)
        prefix_pad_2d_masks_expanded = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        # Concatenate Masks
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks_expanded, suffix_att_2d_masks], dim=2)
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)

        # Build Position IDs
        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        suffix_position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # 7. Suffix Forward (using the KV Cache)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=suffix_position_ids,
            past_key_values=past_key_values, # reuse the Prefix's KV
            inputs_embeds=[None, suffix_embs], # Prefix is None
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        # 8. Get predicted action
        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        
        if self.config.use_action_quality_filter:
            # print("the action_quality is:", action_quality.shape)
            # print("the self.action_quality_query_feature is:", self.action_quality_query_feature.shape)
            quality_condition_feature = self.action_quality_query_feature[action_quality]
            
            suffix_out = suffix_out + quality_condition_feature

        # Consistency Model directly predicts x0 (Clean Action)
        x0_pred = self.action_out_proj(suffix_out)

        return x0_pred

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
        """
        RTC (Recurrent Target Consistency) guided sampling - single Forward + Backward implementation

        Core idea: with one forward pass + one backward pass (VJP), align the predicted action with a_prev over the first d steps.

        Key point: only 1 forward + 1 backward, consistent with the Consistency Model's fast-inference design.

        Args:
            device: compute device
            observation: observation dict
            a_prev: remaining portion of the previous action chunk (physical space, shape: [remaining_steps, action_dim])
            d: expected inference latency in steps (size of the frozen region)
            s: current execution progress
            num_steps: number of sampling steps (usually 1 for Consistency policy)
            guidance_scale: guidance strength, controls how much the gradient affects the output

        Returns:
            actions: new action chunk [batch_size, action_horizon, action_dim] (normalized space)
        """
        device = self.action_in_proj.weight.device
        dtype = self.action_in_proj.weight.dtype

        # 1. Preprocess Observation
        images, img_masks, lang_tokens, lang_masks, state, _ = self._preprocess_observation(observation, train=False)
        bsize = state.shape[0]

        # 2. Convert a_prev to a Tensor
        if isinstance(a_prev, np.ndarray):
            a_prev_tensor = torch.from_numpy(a_prev).to(device=device, dtype=dtype)
            a_prev_tensor = a_prev_tensor.unsqueeze(0)
        else:
            a_prev_tensor = a_prev.to(device=device, dtype=dtype)

        # Get the length of a_prev
        a_prev_len = a_prev_tensor.shape[1]
        frozen_steps = min(d, a_prev_len)


        # 3. Compute Prefix Embedding (Vision + Language)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)

        # Build the Prefix's Attention Masks
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        # 4. Run Prefix Forward to obtain Past Key Values (KV Cache)
        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"

        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        # 5. Prepare initial noise (single-step sampling)
        sigma_max = self.denoiser.sigma_max

        # Initial noise x_T ~ N(0, sigma_max^2), requires gradient for VJP
        x_t = torch.randn(
            (bsize, self.config.action_horizon, self.config.action_dim),
            device=device,
            dtype=dtype,
            requires_grad=True
        ) * sigma_max

        t_in = torch.full((bsize,), sigma_max, device=device, dtype=dtype)

        # ========== Single forward pass ==========

        # Step 1: forward pass to get the predicted action
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, t_in)

        # Check whether embed_suffix preserved the gradient
        if suffix_embs.requires_grad:
            print(f"[Debug-embed] ✓ embed_suffix preserved the gradient!")
        else:
            print(f"[Debug-embed] ✗ embed_suffix cut off the gradient!")

        if self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            suffix_embs_before = suffix_embs
            suffix_embs = suffix_embs.to(torch.bfloat16)
            # Check whether to() preserved the gradient
            if suffix_embs.requires_grad:
                print(f"[Debug-to(bfloat16)] ✓ to(bfloat16) preserved the gradient")
            else:
                print(f"[Debug-to(bfloat16)] ✗ to(bfloat16) cut off the gradient!")

        # Build Suffix-related Masks
        batch_size = prefix_pad_masks.shape[0]
        suffix_len = suffix_pad_masks.shape[1]

        prefix_pad_2d_masks_expanded = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_pad_masks.shape[1])
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

        x0_pred = self.action_out_proj(suffix_out)

        # ========== Gradient flow debugging ==========
        print(f"[Debug-1] x_t.requires_grad: {x_t.requires_grad}")
        print(f"[Debug-2] suffix_embs.requires_grad (True means embed_suffix preserved the gradient): {suffix_embs.requires_grad if hasattr(suffix_embs, 'requires_grad') else 'N/A (may be bool or mask)'}")
        print(f"[Debug-3] x0_pred.requires_grad: {x0_pred.requires_grad}")

        # ========== Single backward pass (VJP) + gradient guidance ==========

        if frozen_steps > 0:
            # Compute the guidance loss: the first frozen_steps should align with a_prev
            pred_frozen = x0_pred[:, :frozen_steps, :]
            target_frozen = a_prev_tensor[:, :frozen_steps, :]
            # print("the pred_frozen is:", pred_frozen.shape)
            # print("the target_frozen is:", target_frozen.shape)
            guidance_loss = F.mse_loss(pred_frozen, target_frozen)

            # Use torch.autograd.grad to directly compute the VJP gradient
            # Note: x_t needs to preserve its gradient; past_key_values does not participate in gradient computation
            try:
                grad_x_t = torch.autograd.grad(
                    outputs=guidance_loss,
                    inputs=x_t,
                    create_graph=False,
                    retain_graph=False,
                )[0]

                # Use the gradient to fine-tune at the output level
                with torch.no_grad():
                    # Compute a (first-order) Jacobian approximation of output w.r.t. input
                    # Adjust the output via the gradient so the frozen region better matches a_prev

                    # Create time-decay weights: steps closer to the start get larger weight, farther steps get smaller weight
                    # Use exponential decay: weight[t] = exp(-decay * t)
                    action_horizon = x0_pred.shape[1]
                    t_indices = torch.arange(action_horizon, device=device, dtype=dtype)

                    # Decay rate: adjustable as needed; 0.1 gives a relatively gentle decay
                    decay_rate = 0.1
                    # Compute weights: earlier steps (t=0) get weight 1.0, later steps decay exponentially
                    weight = torch.exp(-decay_rate * t_indices)

                    # Normalize weights to the range [0, 1]
                    weight = weight / weight.max()

                    # Expand dims to match x0_pred's shape [batch_size, action_horizon, action_dim]
                    weight = weight.view(1, -1, 1).expand_as(x0_pred)

                    # guidance_scale controls the adjustment strength
                    # Apply weighting: earlier steps receive a larger gradient adjustment
                    x0_pred_adjusted = x0_pred - guidance_scale * grad_x_t * 0.01 * weight

                    # Hard constraint: force the frozen region to exactly match a_prev
                    x0_pred_adjusted[:, :frozen_steps, :] = a_prev_tensor[:, :frozen_steps, :]

                    print(f"[Debug] Gradient guidance succeeded! grad norm: {grad_x_t.norm().item():.6f}")

                return x0_pred_adjusted
            except RuntimeError as e:
                # Fall back to the hard constraint if gradient computation fails
                print(f"[Warning] Gradient computation failed: {e}, falling back to hard constraint")
                print("[Debug] Checking gradient flow:")
                print(f"  - x_t.requires_grad: {x_t.requires_grad}")
                print(f"  - x0_pred.requires_grad: {x0_pred.requires_grad}")
                print(f"  - guidance_loss.requires_grad: {guidance_loss.requires_grad}")

                with torch.no_grad():
                    x0_pred_adjusted = x0_pred.clone()
                    x0_pred_adjusted[:, :frozen_steps, :] = a_prev_tensor[:, :frozen_steps, :]

                return x0_pred_adjusted
        else:
            # No frozen region, return the prediction directly
            with torch.no_grad():
                return x0_pred.detach()


    def denoise_step(
        self,
        state,
        prefix_pad_masks,
        past_key_values,
        x_t,
        timestep,
    ):
        """Apply one denoising step of the noise `x_t` at a given timestep."""
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, timestep)

        suffix_len = suffix_pad_masks.shape[1]
        batch_size = prefix_pad_masks.shape[0]
        prefix_len = prefix_pad_masks.shape[1]

        prefix_pad_2d_masks = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)

        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        full_att_2d_masks = torch.cat([prefix_pad_2d_masks, suffix_att_2d_masks], dim=2)

        prefix_offsets = torch.sum(prefix_pad_masks, dim=-1)[:, None]
        position_ids = prefix_offsets + torch.cumsum(suffix_pad_masks, dim=1) - 1

        # Prepare attention masks
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)
        self.paligemma_with_expert.gemma_expert.model.config._attn_implementation = "eager"  # noqa: SLF001

        outputs_embeds, _ = self.paligemma_with_expert.forward(
            attention_mask=full_att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=[None, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        suffix_out = outputs_embeds[1]
        suffix_out = suffix_out[:, -self.config.action_horizon :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        return self.action_out_proj(suffix_out)

    @torch.no_grad()
    def guided_noise_sample_actions(
        self,
        device,
        observation,
        a_prev: np.ndarray,
        d: int,
        noise_mix_decay: float = 0.01,
    ) -> Tensor:
        """
        Generate actions using a noise-guidance method, guiding subsequent action generation based on a_prev.

        Core idea:
        1. Encode a_prev back into noise space (at the sigma_max level)
        2. Build mixed noise: use the encoded noise for the first d steps, time-decay mixing for later steps
        3. Perform consistency sampling using the mixed noise
        4. Hard constraint: replace the first d steps with a_prev

        Args:
            device: compute device
            observation: observation dict
            a_prev: previous action sequence (physical space, shape: [action_horizon, action_dim] or [batch, action_horizon, action_dim])
            d: number of frozen steps (the first d steps align with a_prev)
            noise_mix_decay: time-decay rate for noise mixing

        Returns:
            actions: new action sequence [batch_size, action_horizon, action_dim]
        """
        device = self.action_in_proj.weight.device
        dtype = self.action_in_proj.weight.dtype

        # 1. Preprocess Observation
        images, img_masks, lang_tokens, lang_masks, state, action_quality = self._preprocess_observation(observation, train=False)
        bsize = state.shape[0]

        # 2. Convert a_prev to a Tensor and ensure the correct shape
        if isinstance(a_prev, np.ndarray):
            a_prev_tensor = torch.from_numpy(a_prev).to(device=device, dtype=dtype)
            if a_prev_tensor.ndim == 2:
                a_prev_tensor = a_prev_tensor.unsqueeze(0)  # [1, action_horizon, action_dim]
        else:
            a_prev_tensor = a_prev.to(device=device, dtype=dtype)
            if a_prev_tensor.ndim == 2:
                a_prev_tensor = a_prev_tensor.unsqueeze(0)

        # Expand the batch dimension to match bsize
        if a_prev_tensor.shape[0] == 1 and bsize > 1:
            a_prev_tensor = a_prev_tensor.expand(bsize, -1, -1)

        # 1. First pad a_prev to action_horizon (if needed)
        current_len = a_prev_tensor.shape[1]
        target_len = self.config.action_horizon
        if current_len < target_len:
            pad_len = target_len - current_len
            a_prev_tensor = F.pad(a_prev_tensor, (0, 0, 0, pad_len), mode='constant', value=0)

        action_horizon = a_prev_tensor.shape[1]
        frozen_steps = min(d, action_horizon)

        # print("the a prev len is:", action_horizon)
        # print("the shape is:", a_prev_tensor.shape)

        # 2. Generate guidance noise
        # Use an intermediate noise level: sigma_mid = sqrt(sigma_max * sigma_min)
        # This better preserves a_prev's guidance signal while avoiding excessive noise
        sigma_max = self.denoiser.sigma_max
        sigma_min = self.denoiser.sigma_min
        sigma_mid = torch.sqrt(torch.tensor(sigma_max * sigma_min))

        sigma_mid = sigma_max
        print("the sigma mid is:", sigma_mid)

        # Guidance noise: x_t = a_prev + eps * sigma_mid
        guidance_noise = torch.randn_like(a_prev_tensor) * sigma_mid
        x_t_guidance = a_prev_tensor + guidance_noise
        # x_t_guidance = guidance_noise

        # 3. Build the mixed noise state
        # Generate a new random noise state for mixing
        random_noise = torch.randn((bsize, action_horizon, self.config.action_dim), device=device, dtype=dtype) * sigma_max
        x_t_random = random_noise.clone()

        # First d steps: use the guidance noise state (containing a_prev information)
        # This way, sampling naturally denoises starting from a_prev, preserving similarity
        if frozen_steps > 0:
            x_t = x_t_guidance.clone()
            x_t[:, frozen_steps:, :] = x_t_random[:, frozen_steps:, :]
        else:
            x_t = x_t_random

        # Later steps: time-decay mixing (transitioning from the guidance state to the random state)
        if frozen_steps < action_horizon:
            for t in range(frozen_steps, action_horizon):
                # Compute the time-decay weight lambda(t) = exp(-decay * (t - d))
                lambda_t = math.exp(-noise_mix_decay * (t - frozen_steps))
                # Mix the noise state: guidance-state component + random-state component
                x_t[:, t, :] = (
                    lambda_t * x_t_guidance[:, t, :] +
                    (1 - lambda_t) * x_t_random[:, t, :]
                )

        # 5. Compute Prefix Embedding and KV Cache
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)
        prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
        prefix_att_2d_masks_4d = self._prepare_attention_masks_4d(prefix_att_2d_masks)
        prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1

        self.paligemma_with_expert.paligemma.language_model.config._attn_implementation = "eager"
        _, past_key_values = self.paligemma_with_expert.forward(
            attention_mask=prefix_att_2d_masks_4d,
            position_ids=prefix_position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, None],
            use_cache=True,
        )

        # 6. Perform consistency sampling (single step) using the mixed noise
        # Start sampling from the intermediate noise level, matching the guidance noise strength
        t_in = torch.full((bsize,), sigma_mid, device=device, dtype=dtype)

        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(state, x_t, t_in)

        if self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype == torch.bfloat16:
            suffix_embs = suffix_embs.to(torch.bfloat16)

        # Build Suffix Masks (allow Suffix to attend to Prefix)
        batch_size = prefix_pad_masks.shape[0]
        suffix_len = suffix_pad_masks.shape[1]
        prefix_len = prefix_pad_masks.shape[1]

        # Allow Suffix to attend to Prefix (Causal)
        prefix_pad_2d_masks_expanded = prefix_pad_masks[:, None, :].expand(batch_size, suffix_len, prefix_len)
        suffix_att_2d_masks = make_att_2d_masks(suffix_pad_masks, suffix_att_masks)

        # Concatenate Masks: [Suffix attends to Prefix, Suffix attends to Suffix]
        full_att_2d_masks = torch.cat([prefix_pad_2d_masks_expanded, suffix_att_2d_masks], dim=2)
        full_att_2d_masks_4d = self._prepare_attention_masks_4d(full_att_2d_masks)

        # Build Position IDs: Suffix follows after Prefix
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

        if self.config.use_action_quality_filter:
            # print("the action_quality is:", action_quality.shape)
            # print("the self.action_quality_query_feature is:", self.action_quality_query_feature.shape)
            quality_condition_feature = self.action_quality_query_feature[action_quality]
            
            suffix_out = suffix_out + quality_condition_feature

        x0_pred = self.action_out_proj(suffix_out)

        # 7. Hard constraint: replace the first d steps with a_prev
        result = x0_pred.clone()
        if frozen_steps > 0:
            result[:, :frozen_steps, :] = a_prev_tensor[:, :frozen_steps, :]
        # print("the result is:", result.shape)
        return result