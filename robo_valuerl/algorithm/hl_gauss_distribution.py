import torch
import torch.nn.functional as F

class HLGaussDistribution:
    def __init__(self, bins, sigma):
        """
        bins: Tensor of shape [num_bins], i.e., quantile points or centers of a uniform distribution
        sigma: width of the Gaussian kernel (scalar)
        """
        self.bins = bins
        self.num_bins = len(bins)
        self.sigma = sigma

    # --- Batch processing methods ---

    def batch_encode(self, target_values):
        """
        Input: continuous values of shape [batch_size]
        Output: probability distribution of shape [batch_size, num_bins]
        """
        device = target_values.device
        bins = self.bins.to(device)
        
        # Broadcast computation: [batch, 1] - [1, num_bins]
        diff = target_values.unsqueeze(1) - bins.unsqueeze(0)
        logits = -0.5 * (diff / self.sigma)**2
        return F.softmax(logits, dim=-1)

    def batch_decode(self, probs):
        """
        Input: network output or probabilities of shape [batch_size, num_bins]
        Output: scalar mean of shape [batch_size]
        """
        # If the input is logits, apply softmax first to convert to probabilities
        # probs = F.softmax(pred_logits, dim=-1)
        bins = self.bins.to(probs.device)
        # Expectation E[x] = sum(p * b)
        return torch.sum(probs * bins, dim=-1)

    # --- Single-sample processing methods ---

    def encode(self, target_value):
        """
        Input: a scalar value (float or 0D Tensor)
        Output: 1D probability distribution of shape [num_bins]
        """
        if not isinstance(target_value, torch.Tensor):
            target_value = torch.tensor(target_value)
            
        # Reuse batch_encode: reshape input to [1], then squeeze out the batch dim from the output
        res = self.batch_encode(target_value.unsqueeze(0))
        return res.squeeze(0)

    def decode(self, pred_logits_single):
        """
        Input: 1D network output or probabilities of shape [num_bins]
        Output: a scalar value (float)
        """
        # Reuse batch_decode: add a batch dim to the input, then convert output to scalar
        res = self.batch_decode(pred_logits_single.unsqueeze(0))
        return res.item()


import pandas as pd
import os
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

# ==========================================
# 0. Outlier analysis function
# ==========================================
def analyze_outliers(data, name="Data"):
    """
    Analyze outliers in the data and compute statistics for each percentile
    """
    data_array = np.array(data)
    total = len(data_array)

    print(f"\n{'='*60}")
    print(f"{name} outlier analysis")
    print(f"{'='*60}")

    # Basic statistics
    print(f"Total samples: {total:,}")
    print(f"Min: {data_array.min():.2f}")
    print(f"Max: {data_array.max():.2f}")
    print(f"Mean: {data_array.mean():.2f}")
    print(f"Median: {np.median(data_array):.2f}")
    print(f"Std: {data_array.std():.2f}")

    # Percentile statistics
    percentiles = [1, 5, 10, 25, 50, 75, 90, 95, 99]
    print(f"\nPercentile statistics:")
    for p in percentiles:
        val = np.percentile(data_array, p)
        print(f"  {p:3d}% percentile: {val:10.2f}")

    # Outlier statistics (using IQR method)
    q1 = np.percentile(data_array, 25)
    q3 = np.percentile(data_array, 75)
    iqr = q3 - q1
    lower_bound = q1 - 3 * iqr
    upper_bound = q3 + 3 * iqr

    outliers_lower = np.sum(data_array < lower_bound)
    outliers_upper = np.sum(data_array > upper_bound)
    outliers_total = outliers_lower + outliers_upper

    print(f"\nOutlier detection (IQR method, 3x range):")
    print(f"  Lower bound: {lower_bound:.2f}")
    print(f"  Upper bound: {upper_bound:.2f}")
    print(f"  Samples below lower bound: {outliers_lower:,} ({outliers_lower/total*100:.4f}%)")
    print(f"  Samples above upper bound: {outliers_upper:,} ({outliers_upper/total*100:.4f}%)")
    print(f"  Total outliers: {outliers_total:,} ({outliers_total/total*100:.4f}%)")

    # Extreme value analysis (top 0.1%)
    extreme_threshold = np.percentile(data_array, 99.9)
    extreme_count = np.sum(data_array >= extreme_threshold)
    print(f"\nExtreme value analysis (Top 0.1%):")
    print(f"  Threshold: {extreme_threshold:.2f}")
    print(f"  Sample count: {extreme_count:,} ({extreme_count/total*100:.4f}%)")
    print(f"  Max: {data_array.max():.2f}")

    return {
        'total': total,
        'outliers_total': outliers_total,
        'outliers_pct': outliers_total / total * 100
    }

# ==========================================
# 1. Quantile-based dynamic bin computation
# ==========================================
def compute_quantile_bins(all_rtgs, num_bins=256):
    """
    Compute bins based on data quantiles:
    - Use the actual min/max of the data
    - Uniform quantile division, adapting to the data distribution
    """
    rtgs_tensor = torch.tensor(all_rtgs, dtype=torch.float32)
    
    # Compute the actual min/max of the data
    min_val = rtgs_tensor.min().item()
    max_val = rtgs_tensor.max().item()
    
    print(f"Data stats | Min: {min_val:.2f} | Max: {max_val:.2f} | Samples: {len(all_rtgs)}")
    
    # Compute bins based on quantiles
    quantiles = torch.linspace(0, 1, num_bins)
    bins = torch.quantile(rtgs_tensor, quantiles)
    
    # Ensure bins are monotonically increasing (deduplicate)
    bins = torch.unique(bins)
    
    # If too few bins remain after dedup, pad with a uniform distribution
    if len(bins) < num_bins:
        print(f"Warning: {len(bins)} bins remain after dedup, padding to {num_bins}")
        bins = torch.linspace(min_val, max_val, num_bins)
    
    return bins

# ==========================================
# 2. Optimized dynamic-sigma HL-Gauss class
# ==========================================
class DynamicHLGauss:
    def __init__(self, bins, sigma_scale=0.75):
        self.support = bins.clone().detach()
        self.num_bins = len(bins)
        
        # Compute local spacing to determine dynamic sigma
        intervals = self.support[1:] - self.support[:-1]
        padded_intervals = torch.cat([intervals[:1], intervals, intervals[-1:]])
        local_spacing = (padded_intervals[:-1] + padded_intervals[1:]) / 2.0
        
        self.sigmas = torch.clamp(local_spacing * sigma_scale, min=1e-4)
        print(f"HL-Gauss initialized | Bin count: {self.num_bins} | Sigma range: [{self.sigmas.min():.4f}, {self.sigmas.max():.4f}]")

    def to(self, device):
        self.support = self.support.to(device)
        self.sigmas = self.sigmas.to(device)
        return self

    def batch_encode(self, target_values):
        device = target_values.device
        target = target_values.unsqueeze(1)
        # Core formula: each bin uses its corresponding sigma
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
# 3. Visualization and bimodal test script
# ==========================================
def run_diagnostic_plots(hlg, sample_rtgs, title_prefix=""):
    plt.figure(figsize=(15, 10))

    # Subplot 1: distribution visualization (demonstrating dynamic sigma)
    plt.subplot(2, 1, 1)
    # Choose three representative values: low, mid, high
    min_rtg = np.min(sample_rtgs)
    max_rtg = np.max(sample_rtgs)
    mid_rtg = (min_rtg + max_rtg) / 2
    test_vals = [min_rtg + (max_rtg - min_rtg) * 0.2, mid_rtg, max_rtg - (max_rtg - min_rtg) * 0.2]
    colors = ['red', 'orange', 'green']
    labels = [f'Low Region ({test_vals[0]:.1f})', f'Mid Region ({test_vals[1]:.1f})', f'High Region ({test_vals[2]:.1f})']

    for val, c, l in zip(test_vals, colors, labels):
        dist = hlg.encode(val).cpu().numpy()
        plt.plot(hlg.support.cpu().numpy(), dist, color=c, label=l, lw=2)
        plt.fill_between(hlg.support.cpu().numpy(), dist, color=c, alpha=0.1)

    plt.scatter(hlg.support.cpu().numpy(), [-0.02]*len(hlg.support), marker='|', color='black', alpha=0.3, label='Bin Supports')
    plt.title(f"{title_prefix} - Dynamic HL-Gauss Encoding (Adaptive Sigma)")
    plt.xlabel("Value")
    plt.ylabel("Probability")
    plt.legend()

    # Subplot 2: simulate bimodal prediction for s0 (Ambiguity Handling)
    plt.subplot(2, 1, 2)
    # Simulate network output: one logit peak each in the low-value and high-value regions of the data distribution
    bimodal_logits = torch.full((hlg.num_bins,), -10.0)

    # Find the bin indices near the 20% and 80% quantiles (representing low and high values)
    q20_val = torch.quantile(torch.tensor(sample_rtgs, dtype=torch.float32), 0.2)
    q80_val = torch.quantile(torch.tensor(sample_rtgs, dtype=torch.float32), 0.8)

    idx_low = torch.argmin(torch.abs(hlg.support - q20_val))
    idx_high = torch.argmin(torch.abs(hlg.support - q80_val))

    bimodal_logits[max(0, idx_low-2):min(hlg.num_bins, idx_low+3)] = 10.0  # Simulate the low-value peak
    bimodal_logits[max(0, idx_high-1):min(hlg.num_bins, idx_high+2)] = 12.0  # Simulate the high-value peak (slightly stronger)

    probs = F.softmax(bimodal_logits, dim=-1).cpu().numpy()
    plt.plot(hlg.support.cpu().numpy(), probs, color='purple', lw=2, label='Bimodal Prediction (s0)')
    plt.fill_between(hlg.support.cpu().numpy(), probs, color='purple', alpha=0.2)

    # Decode the expected value
    expected_v = hlg.batch_decode(bimodal_logits.unsqueeze(0)).item()
    plt.axvline(expected_v, color='black', linestyle='--', label=f'Expected Value: {expected_v:.2f}')

    plt.title(f"{title_prefix} - Handling Uncertainty: Bimodal Distribution Example")
    plt.xlabel("Value")
    plt.ylabel("Probability")
    plt.legend()

    plt.tight_layout()

    # Generate a different filename based on the prefix
    save_name = f"hlg_diagnostic_plots_{title_prefix.replace(' ', '_')}.png"
    plt.savefig(save_name)
    print(f"Chart saved: {save_name}")
    plt.close()
# ==========================================
# 4. Main program: data loading and processing
# ==========================================
if __name__ == "__main__":
    base_dir = "/mnt/dataset/toago6/workspace/code/rl_block_x_humanoid_data/filtered_finished_outpoint"

    discount_value_list = []
    remain_time_list = []
    parquet_count = 0

    # Recursively traverse all subdirectories using os.walk
    print(f"Starting directory traversal: {base_dir}")
    for root, dirs, files in os.walk(base_dir):
        for file_name in files:
            if file_name.endswith(".parquet"):
                file_path = os.path.join(root, file_name)
                try:
                    state_data = pd.read_parquet(file_path)
                    if 'discounted_value_return' in state_data.columns:
                        # Flatten any nested arrays that may be present
                        dv_data = state_data['discounted_value_return'].tolist()
                        for item in dv_data:
                            if isinstance(item, (list, np.ndarray)):
                                discount_value_list.extend(item)
                            else:
                                discount_value_list.append(item)
                    if 'remain_time' in state_data.columns:
                        # Flatten any nested arrays that may be present
                        rt_data = state_data['remain_time'].tolist()
                        for item in rt_data:
                            if isinstance(item, (list, np.ndarray)):
                                remain_time_list.extend(item)
                            else:
                                remain_time_list.append(item)
                    parquet_count += 1
                    if parquet_count % 10 == 0:
                        print(f"Processed {parquet_count} files, cumulative discounted_value_return: {len(discount_value_list)}, remain_time: {len(remain_time_list)}")
                except Exception as e:
                    print(f"Failed to read file {file_path}: {e}")

    print(f"\nData collection complete | Files: {parquet_count} | discounted_value_return data points: {len(discount_value_list)} | remain_time data points: {len(remain_time_list)}")

    # ========== Process discounted_value_return ==========
    if len(discount_value_list) > 0:
        print("\n" + "="*60)
        print("Processing discounted_value_return")
        print("="*60)

        # Analyze outliers
        analyze_outliers(discount_value_list, name="discounted_value_return")

        # Compute bins using quantiles
        discount_value_bins = compute_quantile_bins(discount_value_list, num_bins=256)

        # Save bins to file
        discount_value_bins_path = "/mnt/dataset/toago6/workspace/code/hierarchical_rl_in_real_world/12_0114_all_lerobot_version_discounted_value_return_bins_without_rollout.npy"
        np.save(discount_value_bins_path, discount_value_bins.cpu().numpy())
        print(f"discounted_value_return bins saved to {discount_value_bins_path} | Bin count: {len(discount_value_bins)}")

        # Initialize the tool class
        hlg_tool_discount = DynamicHLGauss(bins=discount_value_bins, sigma_scale=0.75)

        # Run diagnostic visualization
        run_diagnostic_plots(hlg_tool_discount, discount_value_list, title_prefix="discounted_value_return")
    else:
        print("\nWarning: no discounted_value_return data found")

    # ========== Process remain_time ==========
    if len(remain_time_list) > 0:
        print("\n" + "="*60)
        print("Processing remain_time")
        print("="*60)

        # Analyze outliers
        analyze_outliers(remain_time_list, name="remain_time")

        # Compute bins using quantiles
        remain_time_bins = compute_quantile_bins(remain_time_list, num_bins=256)

        # Save bins to file
        remain_time_bins_path = "/mnt/dataset/toago6/workspace/code/hierarchical_rl_in_real_world/12_0114_all_lerobot_version_remain_time_bins_without_rollout.npy"
        np.save(remain_time_bins_path, remain_time_bins.cpu().numpy())
        print(f"remain_time bins saved to {remain_time_bins_path} | Bin count: {len(remain_time_bins)}")

        # Initialize the tool class
        hlg_tool_remain = DynamicHLGauss(bins=remain_time_bins, sigma_scale=0.75)

        # Run diagnostic visualization
        run_diagnostic_plots(hlg_tool_remain, remain_time_list, title_prefix="remain_time")
    else:
        print("\nWarning: no remain_time data found")

    print("\n" + "="*60)
    print("Processing complete!")
    print("="*60)