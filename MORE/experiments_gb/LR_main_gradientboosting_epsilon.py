# external
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
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
    Script to produce RandomForstRegressor results applied to the label ranking datasets with Epsilon Layer
"""

if __name__ == "__main__":

    n_jobs = int(os.environ["SLURM_CPUS_PER_TASK"])  # HPC Configuration
    number = int(os.environ["SLURM_ARRAY_TASK_ID"]) # HPC configuration


    random_state = 0
    
    epsilon_value,rank_encoding = [(epsi,enco) for epsi in epsilon_values # 11
                                   for enco in ["dense", "standard", "modified", "fractional"]][number]

    DATA_FOLDER = "LR-GBR"

    regr_estimator = GradientBoostingRegressor(
        n_estimators=100,random_state=random_state
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
    
    
    

    model = regr_model_chain_epsilon
    model_name = regr_name_chain_gb_Epsi + f"({epsilon_value})"


    df = build_plottable_evaluationDataFrame(
        name_to_data=name_to_data_lr,
        models=[model],
        random_state=random_state,
        model_evaluation_function=model_evaluation,
        model_names=[model_name],
        model_score_function=model_scores,
        rank_encoding=rank_encoding
    )

    df.to_csv(DATA_DIR / DATA_FOLDER / f"LR_{model_name}_{rank_encoding}.csv")
