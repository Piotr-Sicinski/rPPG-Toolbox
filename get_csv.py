import os
import json
import pandas as pd
import numpy as np

json_path = "/zfsauton2/home/pushpaka/chiron-heart/data/straps/Publication-Results/neurips-ts4h/jsons/prism_json_NATURE/"
csv_path = "kai_values/field_best/output.csv"

def get_globals_and_windows(name):
    file_path = os.path.join(json_path, f"{name}_prism_details.json")
    if not os.path.exists(file_path):
        return None
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
        globals_dict = {
            "SNR_full": data.get("globals", {}).get("SNR_full"),
            "selected_C_value": data.get("globals", {}).get("selected_c_value"),
            "TV_full": data.get("globals", {}).get("TV_full"),
        }
        windows = data.get("windows", [])
        globals_dict["TV_local_arr"]    = np.array([w.get("TV_local") for w in windows], dtype=float)
        globals_dict["C_low_arr"]       = np.array([w.get("C_low") for w in windows], dtype=float)
        globals_dict["C_high_arr"]      = np.array([w.get("C_high") for w in windows], dtype=float)
        globals_dict["hr_gt_bpm_arr"]   = np.array([w.get("hr_gt_bpm") for w in windows], dtype=float)
        globals_dict["hr_pred_bpm_arr"] = np.array([w.get("hr_pred_bpm") for w in windows], dtype=float)
        globals_dict["SNR_window_arr"]  = np.array([w.get("SNR") for w in windows], dtype=float)

        return globals_dict
    except Exception as e:
        print(f"Error in {file_path}: {e}")
        return None

def main():
    df_csv = pd.read_csv(csv_path)

    all_data = []
    for name in df_csv["name"]:
        globals_dict = get_globals_and_windows(name)
        if globals_dict is not None:
            globals_dict["name"] = name
            all_data.append(globals_dict)
    df_json = pd.DataFrame(all_data)
    df_csv = pd.merge(df_csv, df_json, on="name", how="left")

    print(df_csv.head())
    return df_csv