import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.svm import SVC

iris = datasets.load_iris()

iris_data = pd.DataFrame(data=np.c_[
                         iris['data'], iris['target']], columns=iris['feature_names'] + ['target'])

iris_data.shape

x = iris.data
y = iris.target

print(y)
print(iris.target_names)

x_train, x_test, y_train, y_test = train_test_split(x, y, random_state=0)

svm_model_linear = SVC(kernel='linear', C=1).fit(x_train, y_train)
svm_predictions = svm_model_linear.predict(x_test)

print(svm_predictions)


accuracy = svm_model_linear.score(x_test,y_test)

print(accuracy)