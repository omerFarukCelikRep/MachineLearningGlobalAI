import matplotlib.pyplot as plt
import seaborn as sns
from numpy import where, unique
from sklearn.datasets import make_classification,make_blobs
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_samples, silhouette_score

#K-Means clustering

#define dataset
x,y = make_blobs(n_samples=1000, random_state=5)

#define the model
model = KMeans(n_clusters=2)

#fit the model
model.fit(x)

#assign a cluster to each example
yhat = model.predict(x)

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

#Elbow Maethod for K-Means Clustering

#create a list includes inertia for different k values
#inertia : Sum of squared distances of samples to their closest cluster center

var = []

for k in range(1,10):
    model = KMeans(n_clusters=k)
    model.fit(x)

    print(f"{k} : {model.inertia_}")
    var.append(model.inertia_)

plt.plot(var)
plt.xticks(range(10), range(1,10))

plt.show()

#Silhouette Score

silhouette = []

for k in range(2,11):
    model = KMeans(n_clusters=k)
    model.fit(x)

    label = model.predict(x)

    print(f"{k} : {model.inertia_}")
    silhouette.append(silhouette_score(x,label)) 

ax = plt.plot(silhouette)
plt.xticks(range(9), range(2,11))
plt.title("Mean Silhouette score for each K")

plt.show()

print(silhouette)

#Final Run
model = KMeans(n_clusters=2, random_state=5)
model.fit(x)

clusters = model.predict(x)

sns.scatterplot(x.T[0], x.T[1], hue=clusters)

plt.scatter(x=model.cluster_centers_[0][0],
            y=model.cluster_centers_[0][1],
            color='r')

plt.scatter(x=model.cluster_centers_[1][0],
            y=model.cluster_centers_[1][1],
            color='r')