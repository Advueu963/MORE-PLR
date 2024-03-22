import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml

from utils import mean_bucketSize, amount_rankings
from dataLinks import name_to_data
if __name__ == "__main__":

    as_frame = False
    return_X_y = True
    # build data with colums: DATA_NAME, DATA_SAMPLES, DATA_FEATURES, DATA_TARGETS, DATA_MEAN_BUCKETS
    columns = ["Name", "N-Samples", "N-Features", "N-Targets","N-RANKINGS", "Target-Mean-Buckets"]
    res = [" & ".join(columns) + "\\\\"]
    for data_name,data_id in name_to_data.items():
        X, Y = fetch_openml(data_id=data_id, as_frame=as_frame, return_X_y=return_X_y, parser='auto')
        n_samples,n_features = X.shape
        _,n_targets = Y.shape
        target_mean_bucketSize = mean_bucketSize(Y)
        n_rankings = amount_rankings(Y)
        res.append(
            " & ".join([data_name, str(n_samples), str(n_features), str(n_targets),str(n_rankings), str(round(target_mean_bucketSize,3))]) + " \\\\ "
        )
    for file in ["PLR-REAL-Political","LR-REAL-Political"]:
        dataFrame = pd.read_csv(f"../data/{file}.csv")
        # Split in Features and Targets
        X, Y = dataFrame.iloc[:, :-6], dataFrame.iloc[:, -6:]
        # Extract the numpy arrays
        X_data = X.values
        Y_data = Y.values

        n_samples, n_features = X.shape
        _, n_targets = Y.shape
        target_mean_bucketSize = mean_bucketSize(Y)
        n_rankings = amount_rankings(Y)
        res.append(
            " & ".join([file, str(n_samples), str(n_features), str(n_targets), str(n_rankings),
                        str(round(target_mean_bucketSize, 3))]) + " \\\\ "
        )

    res = "\n".join(res)
    print(res)