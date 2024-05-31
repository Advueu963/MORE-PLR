# external
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklr.pairwise import PairwisePartialLabelRanker

# internal
from MORE_models import (
    PLR_RegressorChainInterval,
    PLR_RegressorChain,
    PLR_RandomForestRegressor,
    PLR_LinearRegressorCalibrater,
    PLR_MultiOutputRegressor,
)
from utils import build_plottable_evaluationDataFrame, model_evaluation, model_scores
from globalVariables import *

import os

"""
    Script to produce RFR results for partial label ranking
"""

if __name__ == "__main__":
    n_jobs = -1
    # n_jobs = int(os.environ["SLURM_CPUS_PER_TASK"])  # HPC Configuration

    random_state = 0

    estimator = RandomForestClassifier(n_estimators=100, random_state=random_state)
    clas_model_randomForest = PairwisePartialLabelRanker(estimator, n_jobs=n_jobs)

    regr_estimator = RandomForestRegressor(
        n_estimators=100, n_jobs=n_jobs, random_state=random_state
    )
    regr_model_mort = PLR_RandomForestRegressor(
        n_estimators=100, random_state=random_state, n_jobs=n_jobs
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
    regr_model_rounding = PLR_RegressorChain(
        estimator=regr_estimator, order=None, random_state=random_state
    )

    regr_estimator = RandomForestRegressor(
        n_estimators=100, n_jobs=n_jobs, random_state=random_state
    )
    regr_model_singleTarget = PLR_MultiOutputRegressor(
        estimator=estimator, n_jobs=n_jobs
    )

    regr_estimator = RandomForestRegressor(
        n_estimators=100, n_jobs=n_jobs, random_state=random_state
    )
    calibration_method_model = PLR_LinearRegressorCalibrater(
        estimator=regr_estimator, random_state=0
    )

    model = regr_model_singleTarget
    model_name = regr_name_singleTarget_rf

    df = build_plottable_evaluationDataFrame(
        name_to_data=name_to_data_plr,
        models=[model],
        random_state=random_state,
        model_evaluation_function=model_evaluation,
        model_names=[model_name],
        model_score_function=model_scores,
    )

    df.to_csv(f"../data/MultiRegression/PLR-randomForest/PLR-{model_name}.csv")
    # file_name = "benchMark_PLR_CHAINS-vs-JC_RF"
    # plot_evaluation_data(df,file_name, list(name_to_data_plr.keys()))
