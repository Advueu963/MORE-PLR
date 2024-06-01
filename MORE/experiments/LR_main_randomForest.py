# external
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklr.pairwise import PairwisePartialLabelRanker

# internal
from MORE.MORE_models import (
    PLR_RegressorChainInterval,
    PLR_RegressorChain,
    PLR_RandomForestRegressor,
    PLR_MultiOutputRegressor,
)
from MORE.utils import (
    build_plottable_evaluationDataFrame,
    model_evaluation,
    model_scores,
)
from MORE.constants import *


"""
    Script to produce RandomForstRegressor results applied to the label ranking datasets
"""

name_to_data_lr = {
    "LR-IRIS": 42851,
}


if __name__ == "__main__":

    n_jobs = -1
    # n_jobs = int(os.environ["SLURM_CPUS_PER_TASK"])  # HPC Configuration
    random_state = 0

    DATA_FOLDER = "LR-RFR"

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

    regr_name_singleTarget_rf = "ST-RF"
    regr_name_interval_rf = "Chain-RFR-PI"
    regr_name_rounding_rf = "Chain-RFR-RR"
    regr_name_mort = "Native-RF"
    clas_name_randomForest = "RPC-RF"

    model = regr_model_interval
    model_name = regr_name_interval_rf

    df = build_plottable_evaluationDataFrame(
        name_to_data=name_to_data_lr,
        models=[model],
        random_state=random_state,
        model_evaluation_function=model_evaluation,
        model_names=[model_name],
        model_score_function=model_scores,
    )

    df.to_csv(DATA_DIR / DATA_FOLDER / f"LR_{model_name}.csv")
    # file_name = "benchMark_LR_CHAINS-vs-JC_RF"
    # plot_evaluation_data(df,file_name, list(name_to_data_lr.keys()))
