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
    PLR_MultiOutputRegressor_Interval,
    PLR_RandomForestRegressor_Interval
)
from MORE.utils import (
    build_plottable_evaluationDataFrame_missingLabels,
    model_evaluation_missingLabels,
    model_scores,
)
from MORE.constants import *
import os
import copy
"""
    Script to produce partial label ranking results for missing labels
"""
def build_model_missing(model,name):
    percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]

    return [
        (copy.deepcopy(model),p,name) for p in percentages
    ]


if __name__ == "__main__":
    n_jobs = int(os.environ["SLURM_CPUS_PER_TASK"])  # HPC Configuration
    number = int(os.environ["SLURM_ARRAY_TASK_ID"]) # HPC Configuration
    random_state = 0

    regr_estimator = RandomForestRegressor(n_estimators=100, n_jobs=n_jobs, random_state=random_state)
    estimator = RandomForestClassifier(n_estimators=100, random_state=random_state)

    DATA_FOLDER = "missingLabels"
    
    clas_model_randomForest = PairwisePartialLabelRanker(estimator, n_jobs=n_jobs)
    missing_Clas = build_model_missing(clas_model_randomForest,clas_name_randomForest)

    regr_model_singleTarget_RF = PLR_MultiOutputRegressor(
        estimator=regr_estimator,
        n_jobs=n_jobs,
        missing_label_strategy="drop_individuals",
    )
    missing_ST_RR = build_model_missing(regr_model_singleTarget_RF,regr_name_singleTarget_RR)

    regr_model_singleTarget_RF_PI = PLR_MultiOutputRegressor_Interval(
        estimator=regr_estimator,
        n_jobs=n_jobs,
        missing_label_strategy="drop_individuals",
    )
    missing_ST_PI = build_model_missing(regr_model_singleTarget_RF_PI,regr_name_singleTarget_PI)
    
    regr_model_mort = PLR_RandomForestRegressor(
        random_state=random_state,
        n_jobs=n_jobs,
        missing_label_strategy="drop_individuals",
    )
    missing_MORT_RR = build_model_missing(regr_model_mort,regr_name_mort)
    
    regr_model_mort_interval = PLR_RandomForestRegressor_Interval(
        random_state=random_state,
        n_jobs=n_jobs,
        missing_label_strategy="drop_individuals",
    )
    missing_MORT_PI = build_model_missing(regr_model_mort_interval,regr_name_mort_interval)
    
    regr_model_rounding_rf = PLR_RegressorChain(
        estimator=regr_estimator,
        order=None,
        random_state=random_state,
        missing_label_strategy="drop_individuals",
    )
    missing_CHAIN_RR = build_model_missing(regr_model_rounding_rf,regr_name_rounding_rf)

    regr_model_interval = PLR_RegressorChainInterval(
        estimator=regr_estimator,
        order=None,
        random_state=random_state,
        missing_label_strategy="drop_individuals",
    )
    missing_CHAIN_PI = build_model_missing(regr_model_interval,regr_name_interval_rf)
    
    model_percentage_names=[
        *missing_ST_RR,
        *missing_ST_PI,
        *missing_MORT_RR,
        *missing_MORT_PI,
        *missing_CHAIN_RR,
        *missing_CHAIN_PI,
        *missing_Clas
    ]
    
    model,percentage,model_name = model_percentage_names[number]

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
