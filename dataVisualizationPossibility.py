import numpy as np
from sklearn.datasets import fetch_openml
from sklr.metrics import tau_x_score
from sklearn.manifold import MDS, TSNE
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
data_id = 42855
# Authorship: 42834
# IRIS: 42851
# WINE: 42872
# ECOLI: 42844
# Stock : 42862
# LIBRAS: 42855


name_to_data = {
     'IRIS': 42851,
     'WINE': 42872,
    'Stock' : 42862,
    'Authorship': 42834,
    'ECOLI': 42844,
    'LIBRAS': 42855
}
data_to_name = {y:x for x,y in name_to_data.items()}

# Options for JC data extraction
as_frame = False
return_X_y = True

X, Y = fetch_openml(data_id=data_id, as_frame=as_frame, return_X_y=return_X_y)
Y_float = Y.astype(float)
samples, n_classes = Y.shape
x_md_scaling = MDS(
    n_components=2,
    max_iter=50,
    n_init=4,
    random_state=0,
)
x_tsne = TSNE(
    n_components=2,
    learning_rate="auto",
    init="random",
    perplexity=5,
    random_state=0
)
x_points = np.round(x_tsne.fit_transform(X),2)
print(x_points)
y_labels = np.array(["-".join(x) for x in Y]).reshape(-1,1)
data = np.concatenate((x_points,y_labels),axis=1)
dataFrame = pd.DataFrame(data,columns=["X1","X2","LABEL"])
dataFrame["X1"] = dataFrame["X1"].astype(float)
dataFrame["X2"] = dataFrame["X2"].astype(float)

ax = sns.relplot(dataFrame,x='X1',y='X2',hue="LABEL",palette="Spectral")
ax.figure.savefig(f"../data/dataVisualisation/Visualisation_{data_to_name[data_id]}")
plt.show()