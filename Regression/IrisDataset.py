from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns

iris = load_iris()

print(iris.feature_names)

print(iris.data[:10])

print(iris.target_names)

feature_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
label_df = pd.DataFrame(data=iris.target, columns=['species'])

print(feature_df.head())
print(label_df.head())

iris_df = pd.concat([feature_df, label_df], axis=1)

print(iris_df.head())

print(iris_df.describe())

print(feature_df.describe().T)

print(iris_df.info())

print(feature_df.info())

sns.pairplot(feature_df)