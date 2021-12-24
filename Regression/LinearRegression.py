# y = wx + b
# y => label (Fiyat)
# w = Weight of x
# x = Feature (Metrekare)
# b = bias

from sklearn.linear_model import LinearRegression
import numpy as np
import pandas as pd

df = pd.DataFrame(data={
    'Metrekare': [100, 150, 120, 300, 230],
    'Fiyat': [70, 90, 95, 120, 110]
})

x = df.iloc[:, :-1].values #feature
y = df.iloc[:, -1].values #label

reg = LinearRegression().fit(x,y)

print(f"Score: {reg.score(x,y)}")
print(f"Coefficient: {reg.coef_}")
print(f"Intercept: {reg.intercept_}")

metrekare = int(input("Lütfen fiyat tahmini yapacağınız metrekareyi giriniz : "))

print(f"{metrekare} metrekare evin tahmini fiyatı : {reg.predict(np.array([[metrekare]]))}")

print(metrekare * reg.coef_[0] + float(reg.intercept_))