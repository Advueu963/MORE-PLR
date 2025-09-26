# external
from sklearn.ensemble import GradientBoostingRegressor
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
    Script to produce the missing labels results for label rankings problems
"""

if __name__ == "__main__":
    n_jobs = int(os.environ['SLURM_CPUS_PER_TASK'])  # HPC configuration
    number = int(os.environ["SLURM_ARRAY_TASK_ID"]) # HPC configuration

    DATA_FOLDER = "PLR-GBR/missingLabels/epsi"

    random_state = 0
    epsi_missing_pairs = [ (eps, perc, encoding) 
                          for eps in [0.2] 
                          for perc in [0.6]
                          for encoding in ["standard", "modified", "fractional"]
    ]

    percentage = epsi_missing_pairs[number][1]
    epsilon_value = epsi_missing_pairs[number][0]
    encoding = epsi_missing_pairs[number][2]

    regr_estimator = GradientBoostingRegressor(
        n_estimators=100, random_state=random_state
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

    model = regr_model_chain_epsilon
    model_name = regr_name_chain_gb_Epsi + f"({epsilon_value})"

    df = build_plottable_evaluationDataFrame_missingLabels(
        name_to_data=name_to_data_plr,
        models=[model],
        random_state=random_state,
        percentage=percentage,
        model_evaluation_function=model_evaluation_missingLabels,
        model_names=[model_name],
        model_score_function=model_scores,
        rank_encoding=encoding
    )

    df.to_csv(DATA_DIR / DATA_FOLDER / encoding / f"PLR-{model_name}-{percentage}_hlai.csv")
