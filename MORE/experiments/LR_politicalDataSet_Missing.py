# external
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

from sklr.pairwise import PairwisePartialLabelRanker

import numpy as np
import pandas as pd

# internal
from MORE.MORE_models import (
    PLR_RegressorChainInterval,
    PLR_RegressorChain,
    PLR_RandomForestRegressor,
    PLR_MultiOutputRegressor,
)
from MORE.utils import model_evaluation_missingLabels, model_scores_Pipeline
from MORE.constants import *
import os


"""
    Script to produce Political Data results for label ranking with missing labels
"""

if __name__ == "__main__":
    random_state = 0

    n_jobs = -1
    # n_jobs = int(os.environ["SLURM_CPUS_PER_TASK"])  # HPC Configuration

    DATA_FOLDER = "missingLabels"
    data_files = ["LR-REAL-Political"]

    random_state = 0

    ##### State of the art ######

    estimator = RandomForestClassifier(
        n_estimators=100, n_jobs=n_jobs, random_state=random_state
    )
    clas_model_randomForest = PairwisePartialLabelRanker(estimator, n_jobs=n_jobs)

    #### Single Target #####
    regr_estimator = RandomForestRegressor(random_state=random_state)
    regr_model_singleTarget_RF = PLR_MultiOutputRegressor(
        estimator=regr_estimator, n_jobs=n_jobs
    )

    ##### RFR #####
    regr_model_mort = PLR_RandomForestRegressor(
        random_state=random_state, n_jobs=n_jobs
    )

    regr_estimator = RandomForestRegressor(
        n_estimators=100, n_jobs=n_jobs, random_state=random_state
    )
    regr_model_interval = PLR_RegressorChainInterval(
        estimator=regr_estimator, order=None, random_state=random_state
    )

    regr_estimator = RandomForestRegressor(
        n_estimators=100, n_jobs=n_jobs, random_state=random_state
    )
    regr_model_rounding_rf = PLR_RegressorChain(
        estimator=regr_estimator, order=None, random_state=random_state
    )

    plotData = {
        "data": [],
        "tau_x_score": [],
        "prediction_time": [],
        "buckets_per_rank": [],
        "algo": [],
        "percentage": [],
    }
    model_names = [
        regr_name_interval_rf,
        regr_name_rounding_rf,
        regr_name_singleTarget_rf,
        regr_name_mort,
        clas_name_randomForest,
    ]

    def add_to_data(data_name, score, time, bucket_per_rank, algo, percentage):
        if type(bucket_per_rank) in [
            np.ndarray,
            list,
        ]:  # just to make sure it is a list or np.Array
            assert len(score) == len(bucket_per_rank) == len(time)
            for i in range(len(score)):
                # add the scores
                plotData["data"].append(data_name)
                plotData["tau_x_score"].append(score[i])
                plotData["prediction_time"].append(np.sqrt(time[i]))
                plotData["buckets_per_rank"].append(bucket_per_rank[i])
                plotData["algo"].append(algo)
                plotData["percentage"].append(percentage)
        else:
            plotData["data"].append(data_name)
            plotData["tau_x_score"].append(score)
            plotData["prediction_time"].append(np.sqrt(time))
            plotData["buckets_per_rank"].append(bucket_per_rank)
            plotData["algo"].append(algo)
            plotData["percentage"].append(percentage)

    for file in data_files:
        # Read Data
        dataFrame = pd.read_csv(f"../data/{file}.csv")

        # Split in Features and Targets
        X, Y = dataFrame.iloc[:, :-6], dataFrame.iloc[:, -6:]

        # Extract the numpy arrays
        X_data = X.values
        Y_data = Y.values

        # Build Numerical Preprocessor
        numerical_cols = [
            i
            for i, colname in enumerate(X.columns)
            if X[colname].dtype in ["int64", "float64"]
        ]
        numerical_transformer = Pipeline(
            steps=[
                ("imputer_nan", SimpleImputer(strategy="most_frequent")),
                ("impute_977", SimpleImputer(missing_values=977, strategy="median")),
                ("scaler", StandardScaler()),
            ]
        )

        # Build Categorical Preprocessor
        categorical_cols = [
            i for i, colname in enumerate(X.columns) if X[colname].dtype == "object"
        ]
        categorical_transformer = Pipeline(
            steps=[
                ("imputer_nan", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        preprocessor = ColumnTransformer(
            transformers=[
                ("numerical", numerical_transformer, numerical_cols),
                ("categorical", categorical_transformer, categorical_cols),
            ]
        )

        ############ Random Forest Base Estimator##############

        # MORT
        mort_Pipeline = Pipeline(
            steps=[("preprocessor", preprocessor), (regr_name_mort, regr_model_mort)]
        )
        # Chain-RF-Rounding
        roundingRF_Pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (regr_name_rounding_rf, regr_model_rounding_rf),
            ]
        )
        # Chain-RF-Interval
        intervalRF_Pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (regr_name_interval_rf, regr_model_interval),
            ]
        )
        # Singletarget-RF
        singleTargetRF_Pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (regr_name_singleTarget_rf, regr_model_singleTarget_RF),
            ]
        )
        # JC-RF
        jc_Pipeline_rf = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (clas_name_randomForest, clas_model_randomForest),
            ]
        )

        # res.shape == (n_models, n_splits, 3) with accuracy speed and mean-buckets
        for percentage in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
            results = model_evaluation_missingLabels(
                models=[
                    intervalRF_Pipeline,
                    roundingRF_Pipeline,
                    singleTargetRF_Pipeline,
                    mort_Pipeline,
                    jc_Pipeline_rf,
                ],
                X=X_data,
                Y=Y_data,
                random_state=random_state,
                model_score_function=model_scores_Pipeline,
                percentage_missing_labels=percentage,
            )

            # Add Outputs ot dataFrame
            for index_models in range(results.shape[0]):
                model_accuracies, model_times, model_bucket_sizes = results[
                    index_models
                ].T
                add_to_data(
                    data_name=file,
                    score=model_accuracies,
                    time=model_times,
                    bucket_per_rank=model_bucket_sizes,
                    algo=model_names[index_models],
                    percentage=percentage,
                )

    plotData = pd.DataFrame(plotData)
    plotData.to_csv(DATA_DIR / DATA_FOLDER / "LR_politicalEvaluation_Missing.csv")
    # file_name = f"benchMarkPolitical_MORE-vs-JC"
    # plot_evaluation_data(plotData,file_name,data_files)
