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
    PLR_MultiOutputRegressor_Epsilon,
    PLR_RegressorChain_Epsilon,
    PLR_RandomForestRegressor_Epsilon
)

from MORE.utils import model_scores_Pipeline, model_evaluation
from MORE.constants import *
import os


"""
    Script to produce Political Data results for label ranking
"""

if __name__ == "__main__":
    random_state = 0

    n_jobs = int(os.environ["SLURM_CPUS_PER_TASK"])  # HPC Configuration
    data_files = ["LR-REAL-Political"]

    random_state = 0

    #### RF Base Estimator ####
    regr_estimator = RandomForestRegressor(n_estimators=100,random_state=random_state, n_jobs=n_jobs)

    plotData = {
        "data": [],
        "tau_x_score": [],
        "prediction_time": [],
        "buckets_per_rank": [],
        "algo": [],
    }

    def add_to_data(data_name, score, time, bucket_per_rank, algo):
        if type(bucket_per_rank) in [
            np.ndarray,
            list,
        ]:  # just to make sure it is a list or np.Array
            # Check if we are algo==DATA or a normal Model
            if type(score) in [np.ndarray, list]:
                assert len(score) == len(bucket_per_rank) == len(time)
                for i in range(len(score)):
                    # add the scores
                    plotData["data"].append(data_name)
                    plotData["tau_x_score"].append(score[i])
                    plotData["prediction_time"].append(np.sqrt(time[i]))
                    plotData["buckets_per_rank"].append(bucket_per_rank[i])
                    plotData["algo"].append(algo)
            else:
                # Special Case were we have algo == DATA
                for i in range(len(bucket_per_rank)):
                    # add the scores
                    plotData["data"].append(data_name)
                    plotData["tau_x_score"].append(score)
                    plotData["prediction_time"].append(np.sqrt(time))
                    plotData["buckets_per_rank"].append(bucket_per_rank[i])
                    plotData["algo"].append(algo)
        else:
            plotData["data"].append(data_name)
            plotData["tau_x_score"].append(score)
            plotData["prediction_time"].append(np.sqrt(time))
            plotData["buckets_per_rank"].append(bucket_per_rank)
            plotData["algo"].append(algo)

    for file in data_files:
        # Read Data
        dataFrame = pd.read_csv(f"data/{file}.csv")

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

        # Singletarget-RF
        models_singletarget_epsilon = [Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (regr_name_singleTarget_Epsi+f"({epsilon})", 
                    PLR_MultiOutputRegressor_Epsilon(
                        estimator=regr_estimator,
                        epsilon=epsilon,
                        n_jobs=n_jobs
                    )
                ),
            ]
        )
        for epsilon in epsilon_values ]
        models_singletarget_names = [
            regr_name_singleTarget_Epsi+f"({epsilon})" for epsilon in epsilon_values
        ]

        # Chain-RF
        models_chain_epsilon = [Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (regr_name_chain_Epsi+f"({epsilon})", 
                    PLR_RegressorChain_Epsilon(
                            estimator=regr_estimator,
                            epsilon=epsilon,
                            random_state=random_state,
                        )
                ),
            ]
        )
        for epsilon in epsilon_values]
        models_chain_names = [
            regr_name_chain_Epsi+f"({epsilon})" for epsilon in epsilon_values
        ]
        
        # Native
        models_native_epsilon = [Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (regr_name_mort_Epsi+f"({epsilon})", 
                    PLR_RandomForestRegressor_Epsilon(
                            epsilon=epsilon,
                            n_estimators=100,
                            n_jobs=n_jobs,
                            random_state=random_state
                        )
                ),
            ]
        )
        for epsilon in epsilon_values]
        models_native_names = [
                    regr_name_mort_Epsi+f"({epsilon})" for epsilon in epsilon_values
        ]
        
        


        # res.shape == (n_models, n_splits, 3) with accuracy speed and mean-buckets
        results = model_evaluation(
            models=[
                *models_singletarget_epsilon,
                *models_chain_epsilon,
                *models_native_epsilon
            ],
            X=X_data,
            Y=Y_data,
            random_state=random_state,
            model_score_function=model_scores_Pipeline,
        )
        model_names = [
            *models_singletarget_names,
            *models_chain_names,
            *models_native_names
        ]
        # Add Outputs ot dataFrame
        for index_models in range(results.shape[0]):
            model_accuracies, model_times, model_bucket_sizes = results[index_models].T
            add_to_data(
                data_name=file,
                score=model_accuracies,
                time=model_times,
                bucket_per_rank=model_bucket_sizes,
                algo=model_names[index_models],
            )

    plotData = pd.DataFrame(plotData)
    plotData.to_csv(DATA_DIR / "LR_politicalEvaluation_epsilon.csv")
    # file_name = f"benchMarkPolitical_MORE-vs-JC"
    # plot_evaluation_data(plotData,file_name,data_files)
