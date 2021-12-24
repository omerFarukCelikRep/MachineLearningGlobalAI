import pandas as pd
from scipy.sparse import data
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

df = pd.DataFrame(data={
    'Metrekare': [100, 150, 120, 300, 230],
    'Bina Yaşı': [5, 2, 6, 10, 3],
    'Fiyat': [70, 90, 95, 120, 110]
})

df.head()

x = df.iloc[:, :-1].values
y = df.iloc[:, -1].values

reg = LinearRegression().fit(x, y)

y_pred = []

for idx in range(len(df)):
    metrekare = df.loc[idx]["Metrekare"]
    bina_yasi = df.loc[idx]["Bina Yaşı"]

    y_pred.append(reg.predict([[metrekare, bina_yasi]]))

# x => Test verisi, y => Test verilerinin gerçek label değerleri
print(reg.score(x, y))

# y => Gerçek label değerleri, y_pred => model tarafından tahmin edilen
print(r2_score(y, y_pred))
