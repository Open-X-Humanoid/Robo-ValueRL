"""PyTorch version of LoRA configuration, adapted from JAX version."""

import dataclasses
from typing import Optional


@dataclasses.dataclass
class LoRAConfig:
    """Configuration for LoRA (Low-Rank Adaptation).
    
    This configuration is compatible with PEFT library's LoRA implementation.
    """
    
    # LoRA rank - dimension of the low-rank matrices
    rank: int
    
    # LoRA scaling factor (alpha)
    alpha: float = 1.0
    
    # Dropout probability for LoRA layers
    dropout: float = 0.0
    
    # Enable rank-stabilized LoRA: https://arxiv.org/pdf/2312.03732
    # If True, scaling = alpha / sqrt(rank), otherwise scaling = alpha / rank
    rslora: bool = False
    
    # Target modules to apply LoRA to
    # For attention: ["q_proj", "k_proj", "v_proj", "o_proj"]
    # For FFN: ["gate_proj", "up_proj", "down_proj"]
    target_modules: Optional[list[str]] = None
    
    @property
    def scaling_value(self) -> float:
        """Compute the scaling value for LoRA based on rank and alpha."""
        import math
        return self.alpha / math.sqrt(self.rank) if self.rslora else self.alpha / self.rank
    
    def to_peft_config(self, task_type=None):
        """Convert to PEFT's LoraConfig format.
        
        Args:
            task_type: The task type for PEFT. If None, uses "FEATURE_EXTRACTION" 
                      which works for models without generation methods.
        """
        try:
            from peft import LoraConfig as PeftLoraConfig
            
            target_modules = self.target_modules
            if target_modules is None:
                # Default target modules for Gemma
                target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", 
                                "gate_proj", "up_proj", "down_proj"]
            
            # Use FEATURE_EXTRACTION as default task type to avoid requiring
            # generation methods like prepare_inputs_for_generation
            if task_type is None:
                task_type = "FEATURE_EXTRACTION"
            
            return PeftLoraConfig(
                r=self.rank,
                lora_alpha=self.alpha,
                lora_dropout=self.dropout,
                target_modules=target_modules,
                use_rslora=self.rslora,
                bias="none",
                task_type=task_type,
            )
        except ImportError:
            raise ImportError(
                "PEFT library is not installed. Please install it with: pip install peft"
            )


def get_attn_lora_config(rank: int = 16, alpha: float = 16.0, rslora: bool = False) -> LoRAConfig:
    """Get LoRA config for attention layers."""
    return LoRAConfig(
        rank=rank,
        alpha=alpha,
        rslora=rslora,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )


def get_ffn_lora_config(rank: int = 16, alpha: float = 16.0, rslora: bool = False) -> LoRAConfig:
    """Get LoRA config for feedforward layers."""
    return LoRAConfig(
        rank=rank,
        alpha=alpha,
        rslora=rslora,
        target_modules=["gate_proj", "up_proj", "down_proj"],
    )


def get_full_lora_config(rank: int = 16, alpha: float = 16.0, rslora: bool = False) -> LoRAConfig:
    """Get LoRA config for both attention and FFN layers."""
    return LoRAConfig(
        rank=rank,
        alpha=alpha,
        rslora=rslora,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                       "gate_proj", "up_proj", "down_proj"],
    )

