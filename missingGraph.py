import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
sns.set(rc={"figure.figsize":(6, 5)}) #width=6, height=5

df = pd.read_csv(filepath_or_buffer="../data/MultiRegression/missingLabels/Missing-PLRs.csv",index_col=0)
relevant = ["data","tau_x_score","algo","percentage"]
plot_df = df.loc[:,relevant]
sns_plot = sns.lineplot(df,x="percentage",y="tau_x_score",hue="algo",errorbar=None
                        ,style="algo") # confidence is 95% Confidence interval
sns_plot.figure.savefig("../data/MultiRegression/missingLabels/PLR-missingLabels",dpi=1000)
plt.show()