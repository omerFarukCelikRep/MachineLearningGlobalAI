import numpy as np
import pandas as pd
from scipy import stats
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split, cross_validate
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

x, y = load_breast_cancer(return_X_y=True)

df = pd.DataFrame(x, columns=load_breast_cancer().feature_names)

df.head()

df.info()

df.describe().T

df.isna().sum()

plt.figure(figsize=(16,8))
sns.distplot(df['mean area'])

z = np.abs(stats.zscore(df))

print(z)

outliers = list(set(np.where(z > 3)[0]))

len(outliers)

new_df = df.drop(outliers, axis=0).reset_index(drop=False)

print(new_df)

y_new = y[list(new_df['index'])]

x_new = new_df.drop('index', axis=1)

x_scaled = MinMaxScaler().fit_transform(x_new)

#Scaling and outlier removed
x_train, x_test, y_train, y_test = train_test_split(x_scaled,y_new, test_size=0.3)

model = LogisticRegression()
cv = cross_validate(model, x_train, y_train, cv = 3, n_jobs=-1, return_estimator=True)
print(cv["test_score"])

print("Mean training accuracy: {}".format(np.mean(cv['test_score'])))
print("Test accuracy: {}".format(cv["estimator"][0].score(x_test, y_test)))

cv['test_score'].mean()

best_estimator_index = np.argmax(cv["test_score"])
best_estimator_model = cv["estimator"][best_estimator_index]

pred = best_estimator_model.predict(x_test)
cm = confusion_matrix(y_test, pred)

#visualization
plt.figure(figsize=(5, 5))
ax = sns.heatmap(cm, square=True, annot=True, cbar=False)
sns.set(font_scale=3.4)
ax.xaxis.set_ticklabels(["False","True"], fontsize = 12)
ax.yaxis.set_ticklabels(["False","True"], fontsize = 12, rotation=0)
ax.set_xlabel('Predicted Labels', fontsize = 12)
ax.set_ylabel('True Labels', fontsize = 12)
plt.show()

target_names = load_breast_cancer().target_names
custom_prediction_index = int(input("Tahmin etmek istediğiniz değerin indexini girin: "))

custom_prediction = best_estimator_model.predict([x_test[custom_prediction_index]])
real_idx = y_test[custom_prediction_index]
print()
print("Estimated value:", target_names[custom_prediction][0])
print("Real value:", target_names[real_idx])