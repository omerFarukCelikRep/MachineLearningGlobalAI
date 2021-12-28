import numpy as np
import pandas as pd
from scipy.sparse import data
import seaborn as sns
import xgboost as xgb
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score

iris = datasets.load_iris()

x = iris.data
y = iris.target

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=41)

dtrain = xgb.DMatrix(x_train, label=y_train)
dtest = xgb.DMatrix(x_test, label=y_test)

params = {
    'max_depth':3,
    'eta':0.3,
    'silent':1,
    'object': 'multi:softpropb',
    'num_class':3
}

num_round = 20

bst = xgb.train(params, dtrain, num_round)

preds = bst.predict(dtest)

preds[:5]

print(np.argmax(preds[0]))
print(np.argmax(preds[1]))

best_preds = np.asarray([np.argmax(line) for line in preds])

print("Numpy array precision : ", precision_score(y_test, best_preds, average="macro"))

xgb.plot_importance(bst)

dtrain.feature_names

iris.feature_names

iris_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

iris_df['species'] = pd.Series(data=iris.target)

iris_df.head()

xgb.plot_tree(bst, num_trees=4, rankdir="LR")

fig = plt.gcf()

fig.set_size_inches(150,100)

plt.figure(figsize=(15,7))

sns.scatterplot(data=iris_df, x="petal length (cm)", y="petal width (cm)", hue="species")

plt.axvline(2.45,0,1.0)

plt.axhline(1.55,0,1.0)

plt.axvline(5.05,0,1.0)