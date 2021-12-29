import numpy as np
import scipy.cluster._hierarchy as shc
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.cluster import AgglomerativeClustering

#define dataset
x,y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)

#define the model
model = AgglomerativeClustering(n_clusters=2)

#fit the model and predict clusters
yhat = model.fit_predict(x)

#retrieve unique clusters
clusters = np.unique(yhat)

#create scatter plot for samples from each cluster
for cluster in clusters:
    #get row indexes for samples with this cluster
    row_ix = np.where(yhat == cluster)

    #create scatter of these samples
    plt.scatter(x[row_ix, 0], x[row_ix, 1])

#show the plot
plt.show()

clusters

plt.figure(figsize=(10,8))
plt.title("Dendrograms")

dend = shc.dendrogram(shc.linkage(x, method='ward'))