from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

df = pd.DataFrame(data={
    'Metrekare': [100, 150, 120, 300, 230, 175, 220, 270, 190, 220],
    'Bina Yaşı': [5, 2, 6, 10, 3, 7, 6, 8, 9, 4],
    'Fiyat': [70, 90, 95, 120, 110, 120, 95, 140, 220, 100]
})

x = df.iloc[:, :-1].values # feature values (take columns until last column)
y = df.iloc[:, -1].values # label values (take only last column)

x_train, x_test, y_train, y_test = train_test_split(x,y, test_size=0.2, random_state=7)

x_train, x_validation, y_train, y_validation = train_test_split(x_train,y_train, test_size=2)

print(x_train.shape)
print(x_test.shape)

print(y_train.shape)
print(y_test.shape)

print(x_validation.shape)
print(y_validation.shape)

print(x_train)