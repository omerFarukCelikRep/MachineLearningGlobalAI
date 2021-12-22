from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import pandas as pd

df = pd.DataFrame(data={
    'Metrekare': [100, 150, 120, 300, 230],
    'Bina Yaşı': [5, 2, 6, 10, 3],
    'Fiyat': [70, 90, 95, 120, 110]
})

x = df.iloc[:, :-1].values # feature values (take columns until last column)
y = df.iloc[:, -1].values # label values (take only last column)

reg = LinearRegression().fit(x,y)

y_pred = []

for idx in range(len(df)):
    metrekare = df.loc[idx]["Metrekare"]
    bina_yasi = df.loc[idx]["Bina Yaşı"]

    y_pred.append(reg.predict([[metrekare, bina_yasi]]))

print(mean_squared_error(y, y_pred))