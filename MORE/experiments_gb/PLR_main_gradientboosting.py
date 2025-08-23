# external
from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklr.pairwise import PairwisePartialLabelRanker

# internal
from MORE.MORE_models import (
    PLR_RegressorChainInterval,
    PLR_RegressorChain,
    PLR_RandomForestRegressor,
    PLR_MultiOutputRegressor,
    PLR_MultiOutputRegressor_Interval,
    PLR_RandomForestRegressor_Interval,
)
from MORE.utils import (
    build_plottable_evaluationDataFrame,
    model_evaluation,
    model_scores,
)
from MORE.constants import *
import os

"""
    Script to produce RandomForstRegressor results applied to the label ranking datasets
"""

if __name__ == "__main__":

    n_jobs = int(os.environ["SLURM_CPUS_PER_TASK"])  # HPC Configuration
    number = int(os.environ["SLURM_ARRAY_TASK_ID"]) # HPC configuration

    random_state = 0

    DATA_FOLDER = "PLR-GBR"

    estimator = GradientBoostingClassifier(
        n_estimators=100, random_state=random_state
    )
    clas_model_gb = PairwisePartialLabelRanker(estimator, n_jobs=n_jobs)

    regr_estimator = GradientBoostingRegressor(
        n_estimators=100, random_state=random_state
    )
    

    regr_model_rounding = PLR_RegressorChain(
        estimator=regr_estimator, order=None, random_state=random_state
    )
    regr_model_interval = PLR_RegressorChainInterval(
        estimator=regr_estimator, order=None, random_state=random_state
    )

    regr_model_singleTarget = PLR_MultiOutputRegressor(
        estimator=regr_estimator, n_jobs=n_jobs
    )
    regr_model_singleTarget_interval = PLR_MultiOutputRegressor_Interval(
        estimator=regr_estimator, n_jobs=n_jobs
    )
    
    model_names = [
        #*[(regr_model_rounding,                       regr_name_rounding_gb, encoding) for encoding in ["dense", "standard", "modified", "fractional"]],
        #*[(regr_model_interval,                       regr_name_interval_gb, encoding) for encoding in ["dense", "standard", "modified", "fractional"]],
        #*[(regr_model_singleTarget,                   regr_name_gb_singleTarget_RR, encoding) for encoding in ["dense", "standard", "modified", "fractional"]],
        #*[(regr_model_singleTarget_interval,          regr_name_gb_singleTarget_PI, encoding) for encoding in ["dense", "standard", "modified", "fractional"]],
        *[(clas_model_gb,                   clas_name_gb, encoding) for encoding in ["dense"]]
    ]
    
    model, model_name, encoding  = model_names[number]


    df = build_plottable_evaluationDataFrame(
        name_to_data=name_to_data_plr,
        models=[model],
        random_state=random_state,
        model_evaluation_function=model_evaluation,
        model_names=[model_name],
        model_score_function=model_scores,
        rank_encoding=encoding
    )

    df.to_csv(DATA_DIR / DATA_FOLDER / f"PLR_{model_name}_{encoding}.csv")
    # file_name = "benchMark_LR_CHAINS-vs-JC_RF"
    # plot_evaluation_data(df,file_name, list(name_to_data_lr.keys()))
