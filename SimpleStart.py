"""
    This scripts gives a simple overview of how to use this repo.

"""

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import fetch_openml

from sklr.metrics import tau_x_score

from MORE.MORE_models import PLR_MultiOutputRegressor
from MORE.constants import name_to_data


regr_model = PLR_MultiOutputRegressor(
    estimator=RandomForestRegressor(),
    n_jobs=1,
    missing_label_strategy=None
)
X,Y = fetch_openml(data_id=name_to_data["LR-IRIS"],
                    as_frame=False,
                    return_X_y=True)

X_train,X_test,Y_train,Y_test = train_test_split(X,Y,train_size=0.8, random_state=0)
regr_model.fit(X_train, Y_train)
Y_pred = regr_model.predict(X_test)
print("ACCURACY: ", tau_x_score(Y_test, Y_pred))
print("First 5 Predictions: \n", Y_pred[:5])
