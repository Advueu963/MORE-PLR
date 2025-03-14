# external
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklr.pairwise import PairwisePartialLabelRanker

# internal
from MORE.MORE_models import (
    PLR_MultiOutputRegressor_Epsilon,
    PLR_RandomForestRegressor_Epsilon,
    PLR_RegressorChain_Epsilon
)
from MORE.utils import (
    build_plottable_evaluationDataFrame,
    model_evaluation,
    model_scores,
)
from MORE.constants import *

import os

"""
    Script to produce RFR results for partial label ranking
"""

if __name__ == "__main__":
    n_jobs = int(os.environ["SLURM_CPUS_PER_TASK"])  # HPC Configuration
    number = int(os.environ["SLURM_ARRAY_TASK_ID"]) # HPC configuration

    DATA_FOLDER = "PLR-RFR"

    random_state = 0
    epsilon_value = [0.12, 0.14, 0.16, 0.18, 0.2][number]

    regr_estimator = RandomForestRegressor(
        n_estimators=100, n_jobs=n_jobs, random_state=random_state
    )
    
    regr_model_singleTarget_epsilon = PLR_MultiOutputRegressor_Epsilon(
        estimator=regr_estimator,
        epsilon=epsilon_value,
        n_jobs=n_jobs
    )
    
    regr_model_chain_epsilon = PLR_RegressorChain_Epsilon(
        estimator=regr_estimator,
        epsilon=epsilon_value,
        random_state=random_state,
    )

    regr_model_native_epsilon = PLR_RandomForestRegressor_Epsilon(
        epsilon=epsilon_value,
        n_estimators=100,
        n_jobs=n_jobs,
        random_state=random_state
    )


    model = regr_model_native_epsilon
    model_name = regr_name_mort_Epsi + f"({epsilon_value})"

    df = build_plottable_evaluationDataFrame(
        name_to_data=name_to_data_plr,
        models=[model],
        random_state=random_state,
        model_evaluation_function=model_evaluation,
        model_names=[model_name],
        model_score_function=model_scores,
    )

    df.to_csv(DATA_DIR / DATA_FOLDER / f"PLR-{model_name}.csv")
    #file_name = "benchMark_PLR_CHAINS-vs-JC_RF"
    # plot_evaluation_data(df,file_name, list(name_to_data_plr.keys()))
