# external
from sklearn.ensemble import RandomForestRegressor

# internal
from MORE.MORE_models import (
    PLR_RegressorChainInterval,
)
from MORE.utils import (
    build_plottable_evaluationDataFrame,
    model_evaluation,
    model_scores,
)
from MORE.constants import *
import os

"""
    Script to produce the missing labels results for label rankings problems
"""

if __name__ == "__main__":
    
    n_jobs = -1
    number = 0
    #n_jobs = int(os.environ['SLURM_CPUS_PER_TASK'])  # HPC configuration
    #number = int(os.environ["SLURM_ARRAY_TASK_ID"]) # HPC configuration

    DATA_FOLDER = "LR-RFR"

    random_state = 0
    coverages = [0, 1, 2, 3]
    coverage = coverages[number]
    
    regr_estimator = RandomForestRegressor(
        n_estimators=100, n_jobs=n_jobs, random_state=random_state
    )
    regr_model_interval = PLR_RegressorChainInterval(
        estimator=regr_estimator,
        order=None,
        random_state=random_state,
        missing_label_strategy="drop_individuals",
        q=coverage
    )

    model = regr_model_interval
    model_name = regr_name_interval_rf


    regr_name_interval_rf = "Chain-RFR-PI"

    df = build_plottable_evaluationDataFrame(
        name_to_data=name_to_data_lr,
        models=[model],
        random_state=random_state,
        model_evaluation_function=model_evaluation,
        model_names=[model_name],
        model_score_function=model_scores
    )

    df.to_csv(DATA_DIR / DATA_FOLDER / f"LR-{model_name}-coverage={coverage}.csv")
