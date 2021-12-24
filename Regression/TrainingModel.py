from sklearn import linear_model
from sklearn.model_selection import train_test_split
from sklearn import datasets
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.metrics import mean_squared_error, r2_score,mean_absolute_error
import pandas as pd
import numpy as np

iris = datasets.load_iris()

feature_df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

print(feature_df.head())

x = feature_df.select_dtypes("float64").drop("sepal length (cm)", axis=1)
y = feature_df["sepal length (cm)"]

print(x.head())
print(y.head())

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=42)

linear_model = LinearRegression()
ridge_model = Ridge() # L2
lasso_model = Lasso() # L1

linear_model.fit(x_train,y_train)
ridge_model(x_train,y_train)
lasso_model(x_train, y_train)

print(linear_model.score(x_train,y_train))
print(ridge_model.score(x_train,y_train))
print(lasso_model.score(x_train,y_train))


lin_pred = linear_model.predict(x_test)
ridge_pred = ridge_model.predict(x_test)
lasso_pred = lasso_model.predict(x_test)

pred_dict = {"Linear": lin_pred, "Ridge":ridge_pred, "Lasso":lasso_pred}

print(pred_dict)

for key, value in pred_dict.items():
    print("Model : ", key)
    print("R2 Score", r2_score(y_test, value))
    print("Mean Absolute Error : ", mean_absolute_error(y_test, value))
    print("Mean Squared Error : ", mean_squared_error(y_test, value))
    print("Root Mean Squared Error : ", np.sqrt(mean_squared_error(y_test, value)))

value_to_predict = [[3.7, 5, 0.9]]

lin_pred= linear_model.predict(value_to_predict)
ridge_pred = ridge_model.predict(value_to_predict)
lasso_pred = lasso_model.predict(value_to_predict)

pred_dict = {"Linear": lin_pred, "Ridge":ridge_pred, "Lasso":lasso_pred}

print(pred_dict)