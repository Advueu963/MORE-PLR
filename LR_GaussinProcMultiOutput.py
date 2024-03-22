import time
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, RationalQuadratic, ExpSineSquared,ConstantKernel
from sklearn.tree import DecisionTreeClassifier

from MORE_models import PLR_GaussianProcessRegressor
from sklearn.ensemble import RandomForestClassifier
from sklr.pairwise import PairwisePartialLabelRanker
from utils import build_plottable_evaluationDataFrame, plot_evaluation_data
from utils_chain import model_evaluation, model_scores
import os

#kernel = 5*RationalQuadratic(length_scale=0.9) * RationalQuadratic(length_scale=0.1) + 10*RationalQuadratic(length_scale=2) + 0.2**2*WhiteKernel() * ExpSineSquared()
#kernel = RBF(length_scale=2)*RBF(length_scale=3)*RBF(length_scale=4) + 0.1*WhiteKernel(noise_level=0.9) + DotProduct(sigma_0=0.5)*ConstantKernel(2)
k1 = 20*RBF(length_scale=0.9)*RBF(length_scale=3)*RBF(length_scale=0.1)
k2 = 0.1 * RBF(length_scale=0.01) + 0.2*WhiteKernel(noise_level=4)
k3 = 0.9 * RationalQuadratic(length_scale=0.001, alpha=10) * ExpSineSquared(length_scale=2,periodicity=0.8)
k4 =10* RationalQuadratic(length_scale=0.1, alpha=0.1) * RationalQuadratic(length_scale=4, alpha=4)
kernel= k1 + k2 + k3 + k4

random_state = 0
name_to_data_lr = {
        'LR-AUTHORSHIP': 42834,
        "LR-GLASS": 42847,
        'LR-IRIS': 42851,
        # "LR-LETTER": 45727, #For GPR not possible because to much samples
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
    n_jobs =-1 # important that this equals the number of cpus on the linux cluster batch job

    estimator = RandomForestClassifier(random_state=random_state,n_jobs=n_jobs)
    rf_model_clas = PairwisePartialLabelRanker(estimator)

    estimator = DecisionTreeClassifier(random_state=random_state)
    dt_model_clas = PairwisePartialLabelRanker(estimator)

    regr_model  = PLR_GaussianProcessRegressor(kernel=kernel,
                                               random_state=random_state,
                                               n_restarts_optimizer=0)

    regr_name = "GPR"
    rf_model_clas_name = "RF-JC"
    dt_model_clas_name = "DT-JC"

    model = regr_model
    model_name = regr_name
    df = build_plottable_evaluationDataFrame(name_to_data_lr,
                                             models=[regr_model],
                                             random_state=random_state,
                                             model_evaluation_function=model_evaluation,
                                             model_score_function=model_scores,
                                             model_names=[regr_name])

    df.to_csv(f"../data/MultiRegression/LR-GPR/LR-{model_name}.csv")
    #file_name = f"benchMark_LR_GPR-vs-JC"
    #plot_evaluation_data(df,file_name,list(name_to_data_lr.keys()))
