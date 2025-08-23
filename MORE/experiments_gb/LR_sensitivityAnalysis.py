# external
from sklearn.ensemble import GradientBoostingRegressor

# internal
from MORE.MORE_models import (
    PLR_RegressorChainInterval,
    PLR_MultiOutputRegressor_Interval,
    PLR_RandomForestRegressor_Interval
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
    n_jobs = int(os.environ['SLURM_CPUS_PER_TASK'])  # HPC configuration
    number = int(os.environ["SLURM_ARRAY_TASK_ID"]) # HPC configuration

    DATA_FOLDER = "LR-GBR"

    random_state = 0
    coverages = [0, 1, 2, 3]
    coverage_encoding = [(cov,encoding) for cov in coverages for encoding in ENCODINGS][number]
    coverage = coverage_encoding[0]
    rank_encoding = coverage_encoding[1]
    
    regr_estimator = GradientBoostingRegressor(
        n_estimators=100, random_state=random_state
    )
    regr_model_interval_Chain = PLR_RegressorChainInterval(
        estimator=regr_estimator,
        order=None,
        random_state=random_state,
        missing_label_strategy=None,
        q=coverage
    )

    regr_estimator = GradientBoostingRegressor(
            n_estimators=100, random_state=random_state
    )
    regr_model_interval_ST = PLR_MultiOutputRegressor_Interval(
        estimator=regr_estimator,
        n_jobs=n_jobs,
        missing_label_strategy=None,
        q=coverage
    )

    model = regr_model_interval_ST
    model_name = regr_name_gb_singleTarget_PI


    df = build_plottable_evaluationDataFrame(
        name_to_data=name_to_data_lr,
        models=[model],
        random_state=random_state,
        model_evaluation_function=model_evaluation,
        model_names=[model_name],
        model_score_function=model_scores,
        rank_encoding=rank_encoding
    )

    df.to_csv(DATA_DIR / DATA_FOLDER / f"LR-{model_name}-coverage={coverage}_{rank_encoding}.csv")
