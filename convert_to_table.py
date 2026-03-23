import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# ----------------------------
# 1) Point these to your files
# ----------------------------
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
# If your CSVs are tab-separated (like your excerpt), leave sep=None (auto-infer) with engine="python"
read_kwargs = dict(sep=None, engine="python")  # change to sep="," if strictly comma-separated

# ---------------------------------------------
# 2) Helper: load and standardize one CSV frame
# ---------------------------------------------
def load_results(method_name, path):
    df = pd.read_csv(path, **read_kwargs)
    # Normalize likely column names (edit here if yours differ)
    # Expected columns: predicted_hr, ground_truth_hr
    # If yours are named differently, adjust the mapping below.
    rename_map = {
        "predicted_hr": "predicted_hr",
        "ground_truth_hr": "ground_truth_hr",
        "participant_id": "participant_id",
        "window_index": "window_index",
    }
    # only rename columns that actually exist
    rename_map = {k: v for k, v in rename_map.items() if k in df.columns}
    df = df.rename(columns=rename_map)

    # Basic checks
    required = {"predicted_hr", "ground_truth_hr"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{method_name}: Missing columns {missing} in {path}")

    df["method"] = method_name
    df["abs_error"] = (df["predicted_hr"] - df["ground_truth_hr"]).abs()
    return df

# ----------------------------------------
# 3) Load all methods into a single table
# ----------------------------------------
frames = []
for m, p in method_files.items():
    if not Path(p).exists():
        raise FileNotFoundError(f"File not found for method '{m}': {p}")
    frames.append(load_results(m, p))

all_df = pd.concat(frames, ignore_index=True)

# ---------------------------------------------------
# 4) Compute accuracy curve for thresholds 1..10 bpm
# ---------------------------------------------------
thresholds = np.arange(1, 11)  # 1,2,...,10 bpm
rows = []
for method, g in all_df.groupby("method"):
    for t in thresholds:
        acc = (g["abs_error"] <= t).mean() * 100.0  # percentage
        rows.append({"method": method, "threshold_bpm": t, "accuracy_pct": acc})

acc_curve = pd.DataFrame(rows)

# ---------------------------------------------------
# 5) Compute MAE, RMSE, and MAPE for each method
# ---------------------------------------------------
metrics_rows = []
for method, g in all_df.groupby("method"):
    predictions = g["predicted_hr"].values
    ground_truth = g["ground_truth_hr"].values
    num_samples = len(predictions)
    
    # MAE calculation
    mae = np.mean(np.abs(predictions - ground_truth))
    mae_std_error = np.std(np.abs(predictions - ground_truth)) / np.sqrt(num_samples)
    
    # RMSE calculation
    squared_errors = np.square(predictions - ground_truth)
    rmse = np.sqrt(np.mean(squared_errors))
    rmse_std_error = np.sqrt(np.std(squared_errors) / np.sqrt(num_samples))
    
    # MAPE calculation
    mape = np.mean(np.abs((predictions - ground_truth) / ground_truth)) * 100
    mape_std_error = np.std(np.abs((predictions - ground_truth) / ground_truth)) / np.sqrt(num_samples) * 100
    
    metrics_rows.append({
        "method": method,
        "MAE": mae,
        "MAE_SE": mae_std_error,
        "RMSE": rmse,
        "RMSE_SE": rmse_std_error,
        "MAPE": mape,
        "MAPE_SE": mape_std_error
    })

metrics_df = pd.DataFrame(metrics_rows).set_index("method").sort_index()

# -------------------------------------------------------------
# 6) Print the headline accuracy numbers (1, 3, 5, and 10 bpm)
# -------------------------------------------------------------
headline_ts = [1, 3, 5, 10]
headline_acc = (
    acc_curve[acc_curve["threshold_bpm"].isin(headline_ts)]
    .pivot(index="method", columns="threshold_bpm", values="accuracy_pct")
    .rename(columns={1: "acc_1bpm(%)", 3: "acc_3bpm(%)", 5: "acc_5bpm(%)", 10: "acc_10bpm(%)"})
    .sort_index()
)

print("=== ACCURACY WITHIN THRESHOLDS ===")
print(headline_acc.round(2))
print()

print("=== MAE, RMSE, AND MAPE METRICS ===")
print(metrics_df.round(3))
print()

# Optional: Combined summary table
print("=== COMBINED SUMMARY ===")
combined_summary = pd.concat([
    headline_acc[["acc_5bpm(%)"]],  # Just 5 BPM accuracy for brevity
    metrics_df[["MAE", "RMSE", "MAPE"]]    # MAE, RMSE, and MAPE
], axis=1).round(3)
print(combined_summary)

# ------------------------------------------------
# 7) Plot: accuracy vs bpm threshold (per method)
# ------------------------------------------------
plt.figure(figsize=(12, 4))

# Plot 1: Accuracy curves
plt.subplot(1, 2, 1)
for method, g in acc_curve.groupby("method"):
    g = g.sort_values("threshold_bpm")
    plt.plot(g["threshold_bpm"], g["accuracy_pct"], marker="o", label=method)

plt.xlabel("BPM threshold")
plt.ylabel("Accuracy within threshold (%)")
plt.title("PURE - Accuracy vs. BPM Threshold")
plt.grid(True, linestyle="--", alpha=0.5)
plt.legend()

# Plot 2: Error metrics comparison
plt.subplot(1, 2, 2)
metrics_sorted = metrics_df.sort_values("MAE")
x_pos = np.arange(len(metrics_sorted))

# Create grouped bar chart for MAE and RMSE
width = 0.35
plt.bar(x_pos - width/2, metrics_sorted["MAE"], width, 
        yerr=metrics_sorted["MAE_SE"], capsize=3, label='MAE')
plt.bar(x_pos + width/2, metrics_sorted["RMSE"], width,
        yerr=metrics_sorted["RMSE_SE"], capsize=3, label='RMSE')

plt.xticks(x_pos, metrics_sorted.index, rotation=45)
plt.ylabel("Error (BPM)")
plt.title("PURE - MAE vs RMSE Comparison")
plt.legend()
plt.grid(True, linestyle="--", alpha=0.5)

plt.tight_layout()
plt.show()