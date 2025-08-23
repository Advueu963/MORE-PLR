import pandas as pd
import numpy as np
import os
import re
regr_name_interval_gb = "Chain-GBR-PI"
regr_name_rounding_gb = "Chain-GBR-RR"
regr_name_chain_gb_Epsi = "Chain-GBR-RR-EPSI"

regr_name_gb_singleTarget_RR = "ST-GBR-RR"
regr_name_gb_singleTarget_PI = "ST-GBR-PI"
regr_name_gb_singleTarget_Epsi = "ST-GBR-EPSI"

clas_name_gb = "RPC-GB"

ENCODINGS = ["dense","modified","fractional","standard"]
BASIC_FOLDER = [
    "chain",
    "single",
    "rpc",
    "epsi",
    "sensitivity",
    "political"
]
MODEL_TYPES = [
    *[f"LR-GBR/{folder}" for folder in BASIC_FOLDER],
    *[f"LR-GBR/missingLabels/{folder}" for folder in BASIC_FOLDER],
    *[f"PLR-GBR/{folder}" for folder in BASIC_FOLDER],
    *[f"PLR-GBR/missingLabels/{folder}" for folder in BASIC_FOLDER]
]

df_list = []
for folder in MODEL_TYPES:
        for encoding in ENCODINGS:
            if "rpc" in folder and encoding != "dense":
                continue
            path = f"{folder}/{encoding}/"
            all_files = os.listdir(path)
            for file in all_files:
                if file.endswith(".csv"):
                    df = pd.read_csv(os.path.join(path, file))
                    # Check for Missing
                    if "missing" not in  file.lower() or "percentage" not in df.columns:
                        df["percentage"] = 0
                    # Check for Sigma-Factor
                    if "coverage" not in file.lower():
                        # Check for Chain File
                        if "chain" in file.lower():
                            df["sigma-factor"] = 1
                        else:
                            df["sigma-factor"] = -1
                    else:
                        # Extract value from file string "coverage=x"
                        match = re.search(r'coverage=(\d+)', file.lower())
                        df["sigma-factor"] = float(match.group(1))
                    # Adjust Epsi 
                    if "epsi" in file.lower():
                        model_name = df["algo"]
                        model_name, epsi_value = zip(*model_name.apply(lambda x: x.split("(", 1)).values)
                        epsi_value = list(map(lambda x: float(x[:-1]),epsi_value))
                        df["epsi"] = epsi_value
                        df["algo"] = model_name
                    else:
                        df["epsi"] = -1
                    # Remove all RPC Rows if encoding not dense
                    if "RPC-GB" in df.algo.values and encoding != "dense":
                        df = df[~df.algo.str.contains("RPC-GB")]
                    df["encoding"] = encoding
                    
                    # Prepend problem lr or plr to dataset name
                    if "LR-GBR" in folder:
                        df["data"] = "lr-" + df["data"]
                    elif "PLR-GBR" in folder:
                        df["data"] = "plr-" + df["data"]
                    df_list.append(df)
# Add the political Datasets
# all_files = os.listdir("political/")
# for file in all_files:
#     if file.endswith(".csv"):
#         df = pd.read_csv(os.path.join("political/", file))
#         if "dense" in file.lower():
#             df["encoding"] = "dense"
#         elif "modified" in file.lower():
#             df["encoding"] = "modified"
#         elif "fractional" in file.lower():
#             df["encoding"] = "fractional"
#         elif "standard" in file.lower():
#             df["encoding"] = "standard"
#         df_list.append(df)

if df_list:
    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df.drop(columns=["Unnamed: 0"], inplace=True)
    merged_df.to_csv(f"Everything.csv", index=False)