# external
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklr.pairwise import PairwisePartialLabelRanker

# internal
from MORE_models import (
    PLR_RandomForestRegressor,
    PLR_MultiOutputRegressor,
    PLR_RegressorChain,
    PLR_RegressorChainInterval,
)
from utils import (
    build_plottable_evaluationDataFrame_missingLabels,
    model_evaluation_missingLabels,
    model_scores,
)
from globalVariables import *
import os

"""
    Script to produce the missing labels results for label rankings problems
"""

if __name__ == "__main__":
    n_jobs = -1
    # n_jobs = int(os.environ['SLURM_CPUS_PER_TASK'])  # HPC configuration
    # number = int(os.environ["SLURM_ARRAY_TASK_ID"]) # HPC configuration

    random_state = 0
    percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    percentage = 0.5

    estimator = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clas_model_randomForest = PairwisePartialLabelRanker(estimator, n_jobs=n_jobs)

    regr_estimator = RandomForestRegressor(random_state=random_state)
    regr_model_singleTarget_RF = PLR_MultiOutputRegressor(
        estimator=regr_estimator,
        n_jobs=n_jobs,
        missing_label_strategy="drop_individuals",
    )

    regr_model_mort = PLR_RandomForestRegressor(
        random_state=random_state,
        n_jobs=n_jobs,
        missing_label_strategy="drop_individuals",
    )

    regr_estimator = RandomForestRegressor(
        n_estimators=100, n_jobs=n_jobs, random_state=random_state
    )
    regr_model_interval = PLR_RegressorChainInterval(
        estimator=regr_estimator,
        order=None,
        random_state=random_state,
        missing_label_strategy="drop_individuals",
    )

    regr_estimator = RandomForestRegressor(
        n_estimators=100, n_jobs=n_jobs, random_state=random_state
    )
    regr_model_rounding_rf = PLR_RegressorChain(
        estimator=regr_estimator,
        order=None,
        random_state=random_state,
        missing_label_strategy="drop_individuals",
    )

    model = regr_model_singleTarget_RF
    model_name = regr_name_singleTarget_rf

    df = build_plottable_evaluationDataFrame_missingLabels(
        name_to_data=name_to_data_lr,
        models=[model],
        random_state=random_state,
        percentage=percentage,
        model_evaluation_function=model_evaluation_missingLabels,
        model_names=[model_name],
        model_score_function=model_scores,
    )

    df.to_csv(
        f"data/MultiRegression/missingLabels/LR-{model_name}-{percentage}-NoLetter.csv"
    )
