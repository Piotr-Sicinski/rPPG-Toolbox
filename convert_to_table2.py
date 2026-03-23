#%%
import pandas as pd
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from pathlib import Path
from matplotlib.lines import Line2D

# from chironhr import styles
# styles.set_matplotlib_style(mpl)

#%%
def compute_hr_metrics(hr_preds, hr_trues):
    hr_preds, hr_trues = map(np.asarray, (hr_preds, hr_trues))
    return {
        'acc': np.mean(np.abs(hr_preds - hr_trues) <= 5),
        'r': np.corrcoef(hr_preds, hr_trues)[0][1],
        'mae': np.mean(np.abs(hr_preds - hr_trues)),
        'rmse': np.sqrt(np.mean((hr_preds - hr_trues)**2)),
        'sd': np.std(hr_preds - hr_trues),
    }

#%%
# dataset = 'PURE'
dataset = 'UBFC-rPPG'
train_dataset = 'PURE' if dataset == 'UBFC-rPPG' else 'UBFC-rPPG'

method_files = {
    "CHROM": "/zfsauton/data/straps/ICRA2026_toolboxcache/psicinsk/resultsFineFFT/CHROM_FIELD/raw_predictions_per_window.csv",
    "GREEN": "/zfsauton/data/straps/ICRA2026_toolboxcache/psicinsk/resultsFineFFT/GREEN_FIELD/raw_predictions_per_window.csv",
    "ICA": "/zfsauton/data/straps/ICRA2026_toolboxcache/psicinsk/resultsFineFFT/ICA_FIELD/raw_predictions_per_window.csv",
    "LGI": "/zfsauton/data/straps/ICRA2026_toolboxcache/psicinsk/resultsFineFFT/LGI_FIELD/raw_predictions_per_window.csv",
    "OMIT": "/zfsauton/data/straps/ICRA2026_toolboxcache/psicinsk/resultsFineFFT/OMIT_FIELD/raw_predictions_per_window.csv",
    "PBV": "/zfsauton/data/straps/ICRA2026_toolboxcache/psicinsk/resultsFineFFT/PBV_FIELD/raw_predictions_per_window.csv",
    "POS": "/zfsauton/data/straps/ICRA2026_toolboxcache/psicinsk/resultsFineFFT/POS_FIELD/raw_predictions_per_window.csv",
    "TSCAN": "/zfsauton/data/straps/ICRA2026_toolboxcache/psicinsk/resultsFineFFT/UBFC-rPPG_TSCAN_FIELD/predictions_per_window.csv",
    "DeepPhys": "/zfsauton/data/straps/ICRA2026_toolboxcache/psicinsk/resultsFineFFT/UBFC-rPPG_DeepPhys_FIELD/predictions_per_window.csv",
    "EfficientPhys": "/zfsauton/data/straps/ICRA2026_toolboxcache/psicinsk/resultsFineFFT/UBFC-rPPG_EfficientPhys_FIELD/predictions_per_window.csv",
    "PhysFormer": "/zfsauton/data/straps/ICRA2026_toolboxcache/psicinsk/resultsFineFFT/UBFC-rPPG_PhysFormer_DiffNormalized_FIELD/predictions_per_window.csv",
    "PhysNet": "/zfsauton/data/straps/ICRA2026_toolboxcache/psicinsk/resultsFineFFT/UBFC-rPPG_PhysNet_DiffNormalized_FIELD/predictions_per_window.csv"
}
all_metrics = {}
# if dataset == 'PURE':
#     all_metrics['PRISM'] = {'acc': 0.97248, 'r': 0.9973, 'mae': 0.7736, 'rmse': 1.6942, 'sd': 1.6898}
# else:
#     all_metrics['PRISM'] = {'acc': 0.97527, 'r': 0.9958, 'mae': 0.6623, 'rmse': 1.5373, 'sd': 1.5373}
for method_name, file_path in method_files.items():
   df = pd.read_csv(file_path)
   all_metrics[method_name] = compute_hr_metrics(df['predicted_hr'], df['ground_truth_hr'])

df_metrics = pd.DataFrame(all_metrics).T.round(3)

print(df_metrics)


#%%
# Create histograms for CHROM and POS absolute errors
chrom_file = "/zfsauton/data/straps/ICRA2026_toolboxcache/psicinsk/resultsFineFFT/CHROM_FIELD/raw_predictions_per_window_old.csv"
pos_file = "/zfsauton/data/straps/ICRA2026_toolboxcache/psicinsk/resultsFineFFT/POS_FIELD/raw_predictions_per_window.csv"

# Load the data
chrom_df = pd.read_csv(chrom_file)
pos_df = pd.read_csv(pos_file)

# Filter out samples with error < 5
chrom_filtered = chrom_df[chrom_df['absolute_error'] >= 5]
pos_filtered = pos_df[pos_df['absolute_error'] >= 5]

print(f"CHROM: {len(chrom_df)} total samples, {len(chrom_filtered)} samples with error >= 5 bpm")
print(f"POS: {len(pos_df)} total samples, {len(pos_filtered)} samples with error >= 5 bpm")

# Create separate plot for CHROM
plt.figure(figsize=(10, 6))
plt.hist(chrom_filtered['absolute_error'], bins=30, alpha=0.7, color='darkmagenta', edgecolor='black')
plt.title('CHROM - Absolute Error Distribution (≥5 bpm)', fontweight='bold', fontsize=14)
plt.xlabel('Absolute Error (bpm)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(True, alpha=0.3)

# Add statistics to CHROM plot
chrom_mean = chrom_filtered['absolute_error'].mean()
chrom_std = chrom_filtered['absolute_error'].std()
plt.axvline(chrom_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {chrom_mean:.2f}')
plt.legend()
plt.tight_layout()
plt.savefig('chrom_absolute_error_filtered.png')
plt.show()

# Create separate plot for POS
plt.figure(figsize=(10, 6))
plt.hist(pos_filtered['absolute_error'], bins=30, alpha=0.7, color='blue', edgecolor='black')
plt.title('POS - Absolute Error Distribution (≥5 bpm)', fontweight='bold', fontsize=14)
plt.xlabel('Absolute Error (bpm)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.grid(True, alpha=0.3)

# Add statistics to POS plot
pos_mean = pos_filtered['absolute_error'].mean()
pos_std = pos_filtered['absolute_error'].std()
plt.axvline(pos_mean, color='red', linestyle='--', linewidth=2, label=f'Mean: {pos_mean:.2f}')
plt.legend()
plt.tight_layout()
plt.savefig('pos_absolute_error_filtered.png')
plt.show()

# Print summary statistics for filtered data
print("\n=== CHROM Absolute Error Statistics (≥5 bpm) ===")
print(f"Mean: {chrom_mean:.3f} bpm")
print(f"Std: {chrom_std:.3f} bpm")
print(f"Min: {chrom_filtered['absolute_error'].min():.3f} bpm")
print(f"Max: {chrom_filtered['absolute_error'].max():.3f} bpm")
print(f"Median: {chrom_filtered['absolute_error'].median():.3f} bpm")

print("\n=== POS Absolute Error Statistics (≥5 bpm) ===")
print(f"Mean: {pos_mean:.3f} bpm")
print(f"Std: {pos_std:.3f} bpm")
print(f"Min: {pos_filtered['absolute_error'].min():.3f} bpm")
print(f"Max: {pos_filtered['absolute_error'].max():.3f} bpm")
print(f"Median: {pos_filtered['absolute_error'].median():.3f} bpm")

# Find samples with error > 100 bpm
print("\n" + "="*60)
print("SAMPLES WITH ABSOLUTE ERROR > 100 BPM")
print("="*60)

chrom_high_error = chrom_df[chrom_df['absolute_error'] > 100]
pos_high_error = pos_df[pos_df['absolute_error'] > 100]

print(f"\nCHROM - {len(chrom_high_error)} samples with error > 100 bpm:")
if len(chrom_high_error) > 0:
    for idx, row in chrom_high_error.iterrows():
        print(f"  {row['participant_id']}: {row['absolute_error']:.2f} bpm (predicted: {row['predicted_hr']:.2f}, ground_truth: {row['ground_truth_hr']:.2f})")
else:
    print("  No samples found")

print(f"\nPOS - {len(pos_high_error)} samples with error > 100 bpm:")
if len(pos_high_error) > 0:
    for idx, row in pos_high_error.iterrows():
        print(f"  {row['participant_id']}: {row['absolute_error']:.2f} bpm (predicted: {row['predicted_hr']:.2f}, ground_truth: {row['ground_truth_hr']:.2f})")
else:
    print("  No samples found")

# Summary of unique participants with high errors
chrom_participants = set(chrom_high_error['participant_id'].unique()) if len(chrom_high_error) > 0 else set()
pos_participants = set(pos_high_error['participant_id'].unique()) if len(pos_high_error) > 0 else set()
all_participants = chrom_participants.union(pos_participants)

print(f"\nSUMMARY:")
print(f"Unique participants with error > 100 bpm:")
print(f"  CHROM only: {chrom_participants - pos_participants}")
print(f"  POS only: {pos_participants - chrom_participants}")
print(f"  Both methods: {chrom_participants.intersection(pos_participants)}")
print(f"  Total unique participants: {len(all_participants)}")