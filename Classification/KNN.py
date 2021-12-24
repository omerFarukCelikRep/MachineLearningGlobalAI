import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

wine = datasets.load_wine()

wine_data = pd.DataFrame(wine.data, columns=wine.feature_names)
wine_data['target'] = wine.target

print(wine_data)

wine_data.info()

wine_data.nunique()

plt.figure(figsize=(15,7))

sns.scatterplot(x='color_intensity', y='alcohol', hue='target', data=wine_data)

x_train, x_test, y_train, y_test = train_test_split(wine.data, wine.target, test_size=0.3)

knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(x_train,y_train)
y_pred = knn.predict(x_test)

print(y_pred)

print(accuracy_score(y_test, y_pred))

#np.arange(1,25)

knn_tuning= KNeighborsClassifier()

param_grid = {'n_neighbors': np.arange(1,25)}

knn_gs = GridSearchCV(knn_tuning, param_grid, cv=5)

knn_gs.fit(x_train, y_train)

print("Best Parameter : ", knn_gs.best_params_)
print("The mean accuracy of the scores : ", knn_gs.best_score_)