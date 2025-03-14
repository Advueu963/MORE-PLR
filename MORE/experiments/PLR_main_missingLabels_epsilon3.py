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
    n_jobs = int(os.environ['SLURM_CPUS_PER_TASK']) # HPC configuration
    number = int(os.environ["SLURM_ARRAY_TASK_ID"]) # HPC configuration
    # [0.01, 0.03, 0.05, 0.07, 0.09, 0.12, 0.14, 0.16, 0.18, 0.2]
    
    DATA_FOLDER = "missingLabels"

    random_state = 0
    percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    percentage = epsi_missing_pairs[number][1]
    epsilon_value = epsi_missing_pairs[number][0]


    regr_estimator = RandomForestRegressor(
        n_estimators=100, n_jobs=n_jobs, random_state=random_state
    )
    
    regr_model_singleTarget_epsilon = PLR_MultiOutputRegressor_Epsilon(
        estimator=regr_estimator,
        epsilon=epsilon_value,
        n_jobs=n_jobs,
        missing_label_strategy="drop_individuals",
    )
    
    regr_model_chain_epsilon = PLR_RegressorChain_Epsilon(
        estimator=regr_estimator,
        epsilon=epsilon_value,
        random_state=random_state,
        missing_label_strategy="drop_individuals",
    )

    regr_model_native_epsilon = PLR_RandomForestRegressor_Epsilon(
        epsilon=epsilon_value,
        n_estimators=100,
        n_jobs=n_jobs,
        random_state=random_state,
        missing_label_strategy="drop_individuals",
    )


    model = regr_model_native_epsilon
    model_name = regr_name_mort_Epsi + f"({epsilon_value})"

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
