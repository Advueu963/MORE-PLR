from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklr.pairwise import PairwisePartialLabelRanker
from utils import plot_evaluation_data, build_plottable_evaluationDataFrame_missingLabels
from MORE_models import PLR_RandomForestRegressor, PLR_MultiOutputRegressor, PLR_RegressorChain, PLR_RegressorChainInterval
import utils_chain
from dataLinks import name_to_data_lr
import sys
import os

name_to_data_lr = {
        'LR-AUTHORSHIP': 42834,
        "LR-GLASS": 42847,
        'LR-IRIS': 42851,
        "LR-LIBRAS": 45736,
        "LR-PENDIGITS": 42856,
        "LR-SEGMENT": 42859,
        "LR-VEHICLE": 42863,
        "LR-VOWEL": 42865,
        "LR-WINE": 42867,
        "LR-YEAST": 45737,
        # Real Szenario Dataset
        "LR-REAL-MOVIES": 45735,

    }

if __name__ == '__main__':
    n_jobs = int(os.environ['SLURM_CPUS_PER_TASK'])  # important that this equals the number of cpus on the linux cluster batch job
    #number = int(os.environ["SLURM_ARRAY_TASK_ID"])
    
    random_state = 0
    percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    percentage = 0.5
    
    estimator = DecisionTreeClassifier(random_state=random_state)
    clas_model_decisionTree = PairwisePartialLabelRanker(estimator,
                                                         n_jobs=n_jobs)

    estimator = RandomForestClassifier(n_estimators=100,
                                       random_state=random_state)
    clas_model_randomForest = PairwisePartialLabelRanker(estimator,
                                                         n_jobs=n_jobs)

    clas_estimator = SVC(probability=True,random_state=random_state, cache_size=4000)
    clas_model_svm = PairwisePartialLabelRanker(clas_estimator,
                                                n_jobs=n_jobs)

    regr_estimator = DecisionTreeRegressor(
        random_state=random_state)
    regr_model_singleTarget_DT = PLR_MultiOutputRegressor(
        estimator=regr_estimator,
        n_jobs=n_jobs,
        missing_label_strategy="drop_individuals"
    )
    
    
    regr_estimator = RandomForestRegressor(
        random_state=random_state)
    regr_model_singleTarget_RF = PLR_MultiOutputRegressor(
        estimator=regr_estimator,
        n_jobs=n_jobs,
        missing_label_strategy="drop_individuals"
    )
    regr_estimator = SVR(cache_size=4000)
    regr_model_singleTarget_SVM = PLR_MultiOutputRegressor(
        estimator=regr_estimator,
        n_jobs=n_jobs,
        missing_label_strategy="drop_individuals"
    )

    regr_model_mort = PLR_RandomForestRegressor(random_state=random_state,n_jobs=n_jobs,
                                                missing_label_strategy="drop_individuals")
    
    regr_estimator = RandomForestRegressor(n_estimators=100,
                                            n_jobs=n_jobs,
                                            random_state=random_state)
    regr_model_interval = PLR_RegressorChainInterval(estimator=regr_estimator,
                                            order=None,
                                            random_state=random_state,
                                            missing_label_strategy="drop_individuals")

    regr_estimator = RandomForestRegressor(n_estimators=100,
                                            n_jobs=n_jobs,
                                            random_state=random_state)
    regr_model_rounding_rf = PLR_RegressorChain(estimator=regr_estimator,
                                            order=None,
                                            random_state=random_state,
                                            missing_label_strategy="drop_individuals")

    regr_estimator = DecisionTreeRegressor(
                                            random_state=random_state)
    regr_model_rounding_dt = PLR_RegressorChain(estimator=regr_estimator,
                                            order=None,
                                            random_state=random_state,
                                            missing_label_strategy="drop_individuals")


    regr_name_singleTarget_dt = "SingleTarget-DT"
    regr_name_singleTarget_rf = "SingleTarget-RF"
    regr_name_singleTarget_svm = "SingleTarget-SVM"
    regr_name_interval_rf = "Chain-RF-Interval"
    regr_name_rounding_rf = "Chain-RF-Rounding"
    regr_name_rounding_dt = "Chain-DT-Rounding"
    regr_name_mort = "MORT"
    clas_name_decisionTree = "JC-DT"
    clas_name_randomForest = "JC-RF"
    clas_name_svm = "JC-SVM"





    model = clas_model_svm
    model_name = clas_name_svm
    
    df = build_plottable_evaluationDataFrame_missingLabels(name_to_data=name_to_data_lr,
                                             models=[model],
                                             random_state=random_state,
                                             percentage=percentage,
                                             model_evaluation_function=utils_chain.model_evaluation_missingLabels,
                                             model_names=[model_name],
                                             model_score_function=utils_chain.model_scores)

    df.to_csv(f"../data/MultiRegression/missingLabels/LR-{model_name}-{percentage}-NoLetter.csv")
