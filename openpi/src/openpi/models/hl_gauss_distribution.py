import torch
import torch.nn as nn
import torch.nn.functional as F

class HLGaussDistribution:
    def __init__(self, bins, sigma):
        """
        bins: Tensor of shape [num_bins], representing quantiles or uniformly distributed centers
        sigma: Width of the Gaussian kernel (scalar)
        """
        self.bins = bins
        self.num_bins = len(bins)
        self.sigma = sigma

    # --- Batch processing methods ---

    def batch_encode(self, target_values):
        """
        Input: [batch_size] continuous values
        Output: [batch_size, num_bins] probability distribution
        """
        device = target_values.device
        bins = self.bins.to(device)

        # Broadcast computation: [batch, 1] - [1, num_bins]
        diff = target_values.unsqueeze(1) - bins.unsqueeze(0)
        logits = -0.5 * (diff / self.sigma)**2
        return F.softmax(logits, dim=-1)

    def batch_decode(self, probs):
        """
        Input: [batch_size, num_bins] network output or probabilities
        Output: [batch_size] scalar mean values
        """
        # If input is logits, first apply Softmax to convert to probabilities
        # probs = F.softmax(pred_logits, dim=-1)
        bins = self.bins.to(probs.device)
        # Expected value E[x] = sum(p * b)
        return torch.sum(probs * bins, dim=-1)

    # --- Single data processing methods ---

    def encode(self, target_value):
        """
        Input: A scalar value (float or 0D Tensor)
        Output: [num_bins] 1D probability distribution
        """
        if not isinstance(target_value, torch.Tensor):
            target_value = torch.tensor(target_value)

        # Use batch_encode, reshape input to [1], squeeze batch dim after output
        res = self.batch_encode(target_value.unsqueeze(0))
        return res.squeeze(0)

    def decode(self, pred_logits_single):
        """
        Input: [num_bins] 1D network output or probabilities
        Output: A scalar value (float)
        """
        # Use batch_decode, add batch dimension to input, convert output to scalar
        res = self.batch_decode(pred_logits_single.unsqueeze(0))
        return res.item()


import pandas as pd
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ==========================================
# 1. Quantile-based dynamic Bins computation
# ==========================================
def compute_quantile_bins(all_rtgs, num_bins=256):
    """
    Compute bins based on data quantiles:
    - Uses actual data min/max values
    - Uniform quantile division, adaptive to data distribution
    """
    rtgs_tensor = torch.tensor(all_rtgs, dtype=torch.float32)

    # Compute actual data min/max values
    min_val = rtgs_tensor.min().item()
    max_val = rtgs_tensor.max().item()

    print(f"Data statistics | Min: {min_val:.2f} | Max: {max_val:.2f} | Samples: {len(all_rtgs)}")

    # Compute bins based on quantiles
    quantiles = torch.linspace(0, 1, num_bins)
    bins = torch.quantile(rtgs_tensor, quantiles)

    # Ensure bins are monotonically increasing (deduplicate)
    bins = torch.unique(bins)

    # If bin count is insufficient after deduplication, supplement with uniform distribution
    if len(bins) < num_bins:
        print(f"Warning: Bin count after deduplication is {len(bins)}, supplementing to {num_bins}")
        bins = torch.linspace(min_val, max_val, num_bins)

    return bins

# ==========================================
# 2. Optimized dynamic Sigma HL-Gauss class
# ==========================================
class DynamicHLGauss:
    def __init__(self, bins, sigma_scale=0.75):
        self.support = bins.clone().detach()
        self.num_bins = len(bins)

        # Compute local spacing to determine dynamic Sigma
        intervals = self.support[1:] - self.support[:-1]
        padded_intervals = torch.cat([intervals[:1], intervals, intervals[-1:]])
        local_spacing = (padded_intervals[:-1] + padded_intervals[1:]) / 2.0

        self.sigmas = torch.clamp(local_spacing * sigma_scale, min=1e-4)
        print(f"HL-Gauss initialization | Bin count: {self.num_bins} | Sigma range: [{self.sigmas.min():.4f}, {self.sigmas.max():.4f}]")

    def to(self, device):
        self.support = self.support.to(device)
        self.sigmas = self.sigmas.to(device)
        return self

    def batch_encode(self, target_values):
        device = target_values.device
        target = target_values.unsqueeze(1)
        # Core formula: each bin uses corresponding sigma
        diff = target - self.support.to(device).unsqueeze(0)
        logits = -0.5 * (diff / self.sigmas.to(device).unsqueeze(0))**2
        return F.softmax(logits, dim=-1)

    def batch_decode(self, probs):
        # probs = F.softmax(pred_logits, dim=-1)
        return torch.sum(probs * self.support.to(probs.device), dim=-1)

    def encode(self, value):
        v = torch.tensor([float(value)])
        return self.batch_encode(v).squeeze(0)

# ==========================================
# 3. Visualization and bimodal testing script
# ==========================================
def run_diagnostic_plots(hlg, sample_rtgs):
    plt.figure(figsize=(15, 10))

    # Subplot 1: Distribution visualization (demonstrating dynamic Sigma)
    plt.subplot(2, 1, 1)
    # Select three representative values: low, medium, high
    min_rtg = np.min(sample_rtgs)
    max_rtg = np.max(sample_rtgs)
    mid_rtg = (min_rtg + max_rtg) / 2
    test_vals = [min_rtg + (max_rtg - min_rtg) * 0.2, mid_rtg, max_rtg - (max_rtg - min_rtg) * 0.2]
    colors = ['red', 'orange', 'green']
    labels = [f'Low region ({test_vals[0]:.1f})', f'Mid region ({test_vals[1]:.1f})', f'High region ({test_vals[2]:.1f})']

    for val, c, l in zip(test_vals, colors, labels):
        dist = hlg.encode(val).cpu().numpy()
        plt.plot(hlg.support.cpu().numpy(), dist, color=c, label=l, lw=2)
        plt.fill_between(hlg.support.cpu().numpy(), dist, color=c, alpha=0.1)

    plt.scatter(hlg.support.cpu().numpy(), [-0.02]*len(hlg.support), marker='|', color='black', alpha=0.3, label='Bin Supports')
    plt.title("Dynamic HL-Gauss Encoding (Quantile-based adaptive Sigma)")
    plt.xlabel("Return-to-Go")
    plt.ylabel("Probability")
    plt.legend()

    # Subplot 2: Simulated s0 bimodal prediction (Ambiguity Handling)
    plt.subplot(2, 1, 2)
    # Simulate network output: Logit peaks in both low and high value regions of data distribution
    bimodal_logits = torch.full((hlg.num_bins,), -10.0)

    # Find bin indices near 20% and 80% quantiles (representing low and high values)
    q20_val = torch.quantile(torch.tensor(sample_rtgs, dtype=torch.float32), 0.2)
    q80_val = torch.quantile(torch.tensor(sample_rtgs, dtype=torch.float32), 0.8)

    idx_low = torch.argmin(torch.abs(hlg.support - q20_val))
    idx_high = torch.argmin(torch.abs(hlg.support - q80_val))

    bimodal_logits[max(0, idx_low-2):min(hlg.num_bins, idx_low+3)] = 10.0  # Simulate low value peak
    bimodal_logits[max(0, idx_high-1):min(hlg.num_bins, idx_high+2)] = 12.0  # Simulate high value peak (slightly stronger)

    probs = F.softmax(bimodal_logits, dim=-1).cpu().numpy()
    plt.plot(hlg.support.cpu().numpy(), probs, color='purple', lw=2, label='Bimodal prediction (s0)')
    plt.fill_between(hlg.support.cpu().numpy(), probs, color='purple', alpha=0.2)

    # Decode expected value
    expected_v = hlg.batch_decode(bimodal_logits.unsqueeze(0)).item()
    plt.axvline(expected_v, color='black', linestyle='--', label=f'Expected value: {expected_v:.2f}')

    plt.title("Handling uncertainty: Bimodal probability distribution example")
    plt.xlabel("Return-to-Go")
    plt.ylabel("Probability")
    plt.legend()

    plt.tight_layout()

    plt.savefig("hlg_diagnostic_plots.png")
# ==========================================
# 4. Main program: Data reading and processing
# ==========================================
if __name__ == "__main__":
    base_dir = "/mnt/dataset/toago6/workspace/code/rl_block_x_humanoid_data/lerobot_version"

    discount_value_list = []
    parquet_count = 0

    # Use os.walk to recursively traverse all subdirectories
    print(f"Starting directory traversal: {base_dir}")
    for root, dirs, files in os.walk(base_dir):
        for file_name in files:
            if file_name.endswith(".parquet"):
                file_path = os.path.join(root, file_name)
                try:
                    state_data = pd.read_parquet(file_path)
                    if 'discounted_value_return' in state_data.columns:
                        discount_value_list.extend(state_data['discounted_value_return'].tolist())
                        parquet_count += 1
                        if parquet_count % 10 == 0:
                            print(f"Processed {parquet_count} files, cumulative data points: {len(discount_value_list)}")
                except Exception as e:
                    print(f"Failed to read file {file_path}: {e}")

    print(f"\nData collection complete | File count: {parquet_count} | Total data points: {len(discount_value_list)}")

    # Use quantiles to compute bins
    my_bins = compute_quantile_bins(discount_value_list, num_bins=256)

    # Save bins to file
    np.save("/mnt/dataset/toago6/workspace/code/hierarchical_rl_in_real_world/12_0114_all_lerobot_version_bins_without_failure_case.npy", my_bins.cpu().numpy())
    print(f"Bins saved to my_bins.npy | Bin count: {len(my_bins)}")

    # Initialize utility class
    hlg_tool = DynamicHLGauss(bins=my_bins, sigma_scale=0.75)

    # Run diagnostic visualization
    run_diagnostic_plots(hlg_tool, discount_value_list)
