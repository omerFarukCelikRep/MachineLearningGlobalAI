#DBSCAN Clustering
import matplotlib.pyplot as plt
from numpy import unique,where
from sklearn.datasets import make_classification
from sklearn.cluster import DBSCAN

#define dataset
x,y = make_classification(n_samples=1000, n_features=2, n_informative=2, n_redundant=0, n_clusters_per_class=1, random_state=4)

#define the model
model = DBSCAN(eps=0.30, min_samples=9)

#fit the model and predict clusters
yhat = model.fit_predict(x)

#retrieve unique clusters
clusters = unique(yhat)

#create scatter plot for samples from each cluster
for cluster in clusters:
    #get row indexes for samples with this cluster
    row_ix = where(yhat == cluster)

    #create scatter of these samples
    plt.scatter(x[row_ix, 0], x[row_ix, 1])

#show the plot
plt.show()