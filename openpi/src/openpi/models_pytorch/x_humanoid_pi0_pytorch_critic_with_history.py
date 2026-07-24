import logging
import math

import torch
from torch import Tensor
from torch import nn
import torch.nn.functional as F  # noqa: N812

import openpi.models.gemma as _gemma
from openpi.models_pytorch.gemma_pytorch import PaliGemmaWithExpertModel
import openpi.models_pytorch.preprocessing_pytorch as _preprocessing


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


class X_Humanoid_PI0Pytorch_History_Critic(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.pi05 = config.pi05

        paligemma_config = _gemma.get_config(config.paligemma_variant)
        action_expert_config = _gemma.get_config(config.action_expert_variant)

        self.paligemma_config = paligemma_config
        self.action_expert_config = action_expert_config


        self.paligemma_with_expert = PaliGemmaWithExpertModel(
            paligemma_config,
            action_expert_config,
            use_adarms=[False, True] if self.pi05 else [False, False],
            precision=config.dtype,
        )

        self.remain_time_estimation_head = nn.Sequential(
            nn.Linear(action_expert_config.width, action_expert_config.width),
            nn.SiLU(),
            nn.Dropout(0.2),
            nn.Linear(action_expert_config.width, action_expert_config.width),
            nn.SiLU(),
            nn.Linear(action_expert_config.width, config.num_value_bins),
        )

        self.remain_time_query_feature = nn.Parameter(torch.randn(1, action_expert_config.width, dtype=torch.bfloat16))
        
        # History query tokens (16 queries)
        self.num_history_queries = 48
        self.history_query_tokens = nn.Parameter(
            torch.randn(self.num_history_queries, action_expert_config.width, dtype=torch.bfloat16)
        )
        
        # Projection layer to convert SigLIP output dimension to action_expert dimension
        # SigLIP output has paligemma_config.width, we need to project to action_expert_config.width
        self.history_image_projection = nn.Linear(paligemma_config.width, action_expert_config.width)

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
        # observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        if self.config.use_self_preprocess:
            observation = _preprocessing.preprocess_observation_pytorch(observation, train=train)
        else:
            observation = observation
            
        return (
            list(observation.images.values()),
            list(observation.image_masks.values()),
            observation.tokenized_prompt,
            observation.tokenized_prompt_mask,
            observation.state,
        )

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

    def _preprocess_history_images(self, history_images, train=False):
        """Preprocess history images similar to preprocessing_pytorch.py
        
        Args:
            history_images: dict or tensor of history images
            train: whether in training mode
        
        Returns:
            processed images ready for SigLIP encoder in [B, C, H, W] format
        """
        # If history_images is a dict, extract the images
        if isinstance(history_images, dict):
            # Assume we want to process all images in the dict
            processed_images = []
            for key in _preprocessing.IMAGE_KEYS:
                if key in history_images:
                    image = history_images[key]
                    
                    # Handle both [B, C, H, W] and [B, H, W, C] formats
                    is_channels_first = image.shape[1] == 3
                    
                    if not is_channels_first:
                        # Convert [B, H, W, C] to [B, C, H, W] for processing
                        image = image.permute(0, 3, 1, 2)
                    
                    # Now image is in [B, C, H, W] format
                    # Check if we need to resize
                    if image.shape[2:4] != _preprocessing.IMAGE_RESOLUTION:
                        from openpi.shared import image_tools
                        # resize_with_pad_torch expects [B, H, W, C], so convert temporarily
                        image = image.permute(0, 2, 3, 1)
                        image = image_tools.resize_with_pad_torch(image, *_preprocessing.IMAGE_RESOLUTION)
                        # Convert back to [B, C, H, W]
                        image = image.permute(0, 3, 1, 2)
                    
                    processed_images.append(image)
            
            return processed_images
        else:
            # history_images is already a tensor
            image = history_images
            
            # Handle both [B, C, H, W] and [B, H, W, C] formats
            is_channels_first = image.shape[1] == 3
            
            if not is_channels_first:
                # Convert [B, H, W, C] to [B, C, H, W]
                image = image.permute(0, 3, 1, 2)
            
            # Now image is in [B, C, H, W] format
            # Check if we need to resize
            if image.shape[2:4] != _preprocessing.IMAGE_RESOLUTION:
                from openpi.shared import image_tools
                # resize_with_pad_torch expects [B, H, W, C], so convert temporarily
                image = image.permute(0, 2, 3, 1)
                image = image_tools.resize_with_pad_torch(image, *_preprocessing.IMAGE_RESOLUTION)
                # Convert back to [B, C, H, W]
                image = image.permute(0, 3, 1, 2)
            
            return [image]

    def process_history_images(self, history_images, train=False):
        """Process history images through SigLIP encoder and extract features using query tokens.
        
        Args:
            history_images: history images to process (dict or tensor)
            train: whether in training mode
            
        Returns:
            history_features: [B, num_history_queries, width]
        """
        # Preprocess images
        processed_images = self._preprocess_history_images(history_images, train=train)
        
        # Encode images with SigLIP
        history_image_features = []
        for img in processed_images:
            # Use the same image embedding function as in embed_prefix
            def image_embed_func(img):
                return self.paligemma_with_expert.embed_image(img)
            
            with torch.no_grad():
                img_emb = image_embed_func(img)
            history_image_features.append(img_emb)
        
        # Concatenate all image features: [B, total_num_tokens, paligemma_width]
        if len(history_image_features) > 0:
            history_image_features = torch.cat(history_image_features, dim=1)
        else:
            # If no history images, return zero features
            bsize = 1  # Will be handled by caller
            device = self.history_query_tokens.device
            return torch.zeros(bsize, self.num_history_queries, self.history_query_tokens.shape[-1], device=device)
        
        bsize = history_image_features.shape[0]
        
        # Project history image features to action_expert dimension
        # history_image_features: [B, num_tokens, paligemma_width] -> [B, num_tokens, action_expert_width]
        def projection_func(history_image_features):
            return self.history_image_projection(history_image_features)
        
        history_image_features = self._apply_checkpoint(projection_func, history_image_features.to(dtype=torch.float32))
        
        # Expand query tokens for batch: [B, num_queries, action_expert_width]
        query_tokens = self.history_query_tokens.unsqueeze(0).expand(bsize, -1, -1)
        # Avoid .to() operation if dtype already matches to prevent non-leaf tensor issues
        if query_tokens.dtype != history_image_features.dtype:
            query_tokens = query_tokens.type(history_image_features.dtype)
        if query_tokens.device != history_image_features.device:
            query_tokens = query_tokens.to(history_image_features.device)

        # Use standard cross-attention: queries attend to history image features
        # Q: query_tokens [B, num_queries, action_expert_width]
        # K: history_image_features [B, num_tokens, action_expert_width]
        # V: history_image_features [B, num_tokens, action_expert_width]

        d_k = query_tokens.shape[-1]
        # print("the query_tokens shape is:", query_tokens.shape)
        # print("the history_image_features shape is:", history_image_features.shape)

        # Compute attention scores (Q @ K^T / sqrt(d_k))
        attn_scores = torch.matmul(query_tokens, history_image_features.transpose(-2, -1)) / math.sqrt(d_k)
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply attention to get history features (attn_weights @ V)
        history_features = torch.matmul(attn_weights, history_image_features)

        return history_features

    def embed_suffix(self, history_features, device):
        """Embed history features and remain_time query for Expert Gemma processing."""
        embs = []
        pad_masks = []
        att_masks = []
        adarms_cond = None

        bsize = history_features.shape[0]

        # Add history features
        embs.append(history_features)
        pad_masks.append(torch.ones(bsize, history_features.shape[1], dtype=torch.bool, device=device))
        # History tokens can attend to all previous tokens (including themselves)
        att_masks += [1] + ([0] * (history_features.shape[1] - 1))

        # Add remain_time query feature
        remain_time_query_feature = self.remain_time_query_feature.expand(bsize, -1, -1)
        if remain_time_query_feature.dtype != history_features.dtype:
            remain_time_query_feature = remain_time_query_feature.type(history_features.dtype)
        if remain_time_query_feature.device != device:
            remain_time_query_feature = remain_time_query_feature.to(device)
        embs.append(remain_time_query_feature)
        pad_masks.append(torch.ones(bsize, 1, dtype=torch.bool, device=device))
        att_masks += [0]

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=history_features.dtype, device=device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks, adarms_cond

    def model_to_target_dtype(self, dtype):
        if dtype == "bfloat16":
            dtype = torch.bfloat16
        elif dtype == "float32":
            dtype = torch.float32
        else:
            raise ValueError(f"Invalid dtype: {dtype}")

        self.remain_time_estimation_head.to(dtype=dtype)
        self.history_image_projection.to(dtype=dtype)

    def check_gradients(self, verbose=False):
        """Check if key parameters have gradients after backward pass.

        Args:
            verbose: If True, print detailed gradient information for all parameters.

        Returns:
            dict: A dictionary mapping parameter names to their gradient status.
                - True: Parameter has gradients with non-zero magnitude
                - False: Parameter has no gradients or zero magnitude
        """
        key_params = {
            'history_query_tokens': self.history_query_tokens,
            'remain_time_query_feature': self.remain_time_query_feature,
            'history_image_projection.weight': self.history_image_projection.weight,
        }

        grad_status = {}

        for name, param in key_params.items():
            if param.grad is None:
                grad_status[name] = False
                if verbose:
                    logging.warning(f"[Gradient Check] {name}: grad=None")
            else:
                grad_norm = param.grad.norm().item()
                grad_has_nan = torch.isnan(param.grad).any().item()
                grad_has_inf = torch.isinf(param.grad).any().item()

                if grad_has_nan or grad_has_inf:
                    grad_status[name] = False
                    if verbose:
                        logging.warning(f"[Gradient Check] {name}: grad_norm={grad_norm:.6f}, has_nan={grad_has_nan}, has_inf={grad_has_inf}")
                elif grad_norm < 1e-10:
                    grad_status[name] = False
                    if verbose:
                        logging.warning(f"[Gradient Check] {name}: grad_norm too small={grad_norm:.6e}")
                else:
                    grad_status[name] = True
                    if verbose:
                        logging.info(f"[Gradient Check] {name}: grad_norm={grad_norm:.6f} ✓")

        return grad_status


    def diagnose_gradient_flow(self, tensor, name, stage="forward"):
        """Diagnose the gradient status of a tensor.

        Args:
            tensor: the tensor to diagnose
            name: tensor name (used for log output)
            stage: diagnostic stage label
        """
        if tensor is None:
            print(f"[{stage}] {name}: None")
            return

        has_grad_fn = tensor.grad_fn is not None
        requires_grad = tensor.requires_grad
        print(f"[{stage}] {name}: requires_grad={requires_grad}, has_grad_fn={has_grad_fn}, shape={tensor.shape}")

    def forward(self, observation, actions) -> Tensor:
        """Do a full training forward pass and compute the loss (batch_size x num_steps x num_motors)"""

        remain_time_gt = observation.remain_time_gt
        history_image = observation.history_image

        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=True)
        device = images[0].device

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)

        # Process history images
        history_features = self.process_history_images(history_image, train=True)

        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(history_features, device)

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

        suffix_out = self._apply_checkpoint(
            forward_func, prefix_embs, suffix_embs, att_2d_masks_4d, position_ids, adarms_cond
        )

        # Only extract remain_time output (last token)
        remain_time_out = suffix_out[:, -1]

        def remain_time_estimation_head_func(remain_time_out):
            return self.remain_time_estimation_head(remain_time_out)

        remain_time_t = self._apply_checkpoint(remain_time_estimation_head_func, remain_time_out)

        remain_time_estimation_loss = F.cross_entropy(remain_time_t, remain_time_gt, reduction="mean")

        return remain_time_estimation_loss


    @torch.no_grad()
    def predict_remain_time(self, observation) -> dict[str, Tensor]:
        """
        Inference function: predict the Remain Time distribution.

        Args:
            observation: observation dict containing images, language, etc.
        Returns:
            dict: probability distribution for 'remain_time' (after Softmax)
        """
        images, img_masks, lang_tokens, lang_masks, state = self._preprocess_observation(observation, train=False)
        device = images[0].device

        # Encode Prefix (Vision + Language)
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(images, img_masks, lang_tokens, lang_masks)

        # Process history images
        history_image = getattr(observation, 'history_image', None)
        history_features = self.process_history_images(history_image, train=False)

        # Encode Suffix (History Features + remain_time Query Token)
        suffix_embs, suffix_pad_masks, suffix_att_masks, adarms_cond = self.embed_suffix(history_features, device)

        # Unify precision
        if (
            self.paligemma_with_expert.paligemma.language_model.layers[0].self_attn.q_proj.weight.dtype
            == torch.bfloat16
        ):
            suffix_embs = suffix_embs.to(dtype=torch.bfloat16)
            prefix_embs = prefix_embs.to(dtype=torch.bfloat16)

        # Merge Mask and Position IDs
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        att_2d_masks_4d = self._prepare_attention_masks_4d(att_2d_masks)

        # Model forward computation
        (_, suffix_out), _ = self.paligemma_with_expert.forward(
            attention_mask=att_2d_masks_4d,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            adarms_cond=[None, adarms_cond],
        )

        # Extract remain_time Query Token feature (last token)
        remain_time_out = suffix_out[:, -1].to(dtype=torch.float32)

        # Pass through Head and compute probability distribution
        remain_time_logits = self.remain_time_estimation_head(remain_time_out)

        return {
            "remain_time": F.softmax(remain_time_logits, dim=-1),
        }

