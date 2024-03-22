# PLR Letter
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import RepeatedKFold
random_state = 0
n_folds = 10
n_repeats = 5
folder = RepeatedKFold(n_splits=n_folds, n_repeats=n_repeats, random_state=random_state)

X, Y = fetch_openml(data_id=42853,
                    as_frame=False,
                    return_X_y=True,
                    parser='auto')
X = np.ascontiguousarray(X)  # need for performance improvement at SVM
Y = Y.astype(np.float64)

all_splits = [
    (train_idx, test_idx)
    for train_idx, test_idx in folder.split(X, Y)

]