# external
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklr.pairwise import PairwisePartialLabelRanker

# internal
from MORE.MORE_models import (
    PLR_RandomForestRegressor,
    PLR_MultiOutputRegressor,
    PLR_RegressorChain,
    PLR_RegressorChainInterval,
)
from MORE.utils import (
    build_plottable_evaluationDataFrame_missingLabels,
    model_evaluation_missingLabels,
    model_scores,
)
from MORE.constants import *
import os

"""
    Script to produce partial label ranking results for missing labels
"""

if __name__ == "__main__":
    
    n_jobs = -1
    number = 0
    
    #n_jobs = int(os.environ["SLURM_CPUS_PER_TASK"])  # HPC Configuration
    # number = int(os.environ["SLURM_ARRAY_TASK_ID"]) # HPC Configuration

    DATA_FOLDER = "missingLabels"

    random_state = 0
    percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    percentage = percentages[number]

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

    regr_name_singleTarget_rf = "ST-RF"
    regr_name_interval_rf = "Chain-RFR-PI"
    regr_name_rounding_rf = "Chain-RFR-RR"
    regr_name_mort = "Native-RF"
    clas_name_randomForest = "RPC-RF"

    model = regr_model_rounding_rf
    model_name = regr_name_rounding_rf

    df = build_plottable_evaluationDataFrame_missingLabels(
        name_to_data=name_to_data_plr,
        models=[model],
        random_state=random_state,
        percentage=percentage,
        model_evaluation_function=model_evaluation_missingLabels,
        model_names=[model_name],
        model_score_function=model_scores,
    )

    df.to_csv(DATA_DIR / DATA_FOLDER / f"PLR-{model_name}-{percentage}.csv"
    )
