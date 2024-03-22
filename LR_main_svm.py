from sklearn.ensemble import AdaBoostRegressor
from sklr.pairwise import PairwisePartialLabelRanker
from sklearn.svm import SVR, SVC
from utils import build_plottable_evaluationDataFrame,plot_evaluation_data
from utils_chain import model_evaluation, model_scores
from MORE_models import PLR_RegressorChain, PLR_MultiOutputRegressor, PLR_RegressorChainInterval

random_state = 0
data_id = 42855
from dataLinks import name_to_data_lr
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
    n_jobs =int(os.environ['SLURM_CPUS_PER_TASK'])  # important that this equals the number of cpus on the linux cluster batch job




    clas_estimator = SVC(probability=True,random_state=random_state, cache_size=4000)
    clas_model = PairwisePartialLabelRanker(clas_estimator, n_jobs=n_jobs)

    regr_estimator = SVR(cache_size=4000)
    regr_model_chain = PLR_RegressorChain(estimator=regr_estimator,
                                order=None,# the order will later on be updated to an appropriate order
                                random_state=random_state) # not really necessary but double is better

    regr_estimator = SVR(cache_size=4000)
    regr_model_singleTarget = PLR_MultiOutputRegressor(estimator=regr_estimator,
                                n_jobs=n_jobs)

    regr_estimator = AdaBoostRegressor(SVR(cache_size=4000),random_state=random_state)
    regr_model_interval = PLR_RegressorChainInterval(estimator=regr_estimator,
                                order=None,# the order will later on be updated to an appropriate order
                                random_state=random_state) # not really necessary but double is better

    regr_name_chain = "CHAIN-SVR"
    regr_name_singleTarget = "SingleTarget-SVR"
    regr_name_interval = "CHAIN-ADA_SVR-INTERVAL"
    clas_name = "SVC-JC"
    
    model = clas_model
    model_name = clas_name
    
    df = build_plottable_evaluationDataFrame(name_to_data_lr, models=[model],
                                             random_state=random_state,
                                             model_evaluation_function=model_evaluation,
                                             model_names=[model_name],
                                             model_score_function=model_scores)
    
    df.to_csv(f"../data/MultiRegression/LR-SVM/LR_{model_name}-NoLetter.csv")
    #file_name = f"benchMark_LR_SVM-vs-JC_SVM.png"
    #plot_evaluation_data(dataframe=df,
     #                    fileName=file_name,
      #                   xticks=list(name_to_data_lr.keys()))