import os
import json
import numpy as np

def compute_hr_metrics(hr_preds, hr_trues):
    hr_preds, hr_trues = map(np.asarray, (hr_preds, hr_trues))
    return {
        'acc': np.mean(np.abs(hr_preds - hr_trues) <= 5),
        'r': np.corrcoef(hr_preds, hr_trues)[0, 1],
        'mae': np.mean(np.abs(hr_preds - hr_trues)),
        'rmse': np.sqrt(np.mean((hr_preds - hr_trues) ** 2)),
        'sd': np.std(hr_preds - hr_trues),
    }

def main(folder):
    hr_preds, hr_trues = [], []

    for fname in os.listdir(folder):
        if not fname.endswith(".json"):
            continue
        fpath = os.path.join(folder, fname)
        with open(fpath, "r") as f:
            data = json.load(f)
            for w in data.get("windows", []):
                if "hr_pred_bpm" in w and "hr_gt_bpm" in w:
                    pred, true = w["hr_pred_bpm"], w["hr_gt_bpm"]
                    if pred is not None and true is not None:
                        hr_preds.append(pred)
                        hr_trues.append(true)

    if hr_preds:
        metrics = compute_hr_metrics(hr_preds, hr_trues)
        print("=== HR Metrics Across Dataset ===")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")
    else:
        print("No HR data found in JSON files.")

if __name__ == "__main__":
    folder_path = "/zfsauton/data/straps/Publication-Results/neurips-ts4h/jsons/prism_json_NATURE"
    main(folder_path)
