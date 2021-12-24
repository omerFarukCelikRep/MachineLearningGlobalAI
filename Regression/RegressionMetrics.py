import pandas as pd
from sklearn.linear_model import LinearRegression

df = pd.DataFrame(data={
    'Metrekare': [100, 150, 120, 300, 230],
    'Fiyat': [70, 90, 95, 120, 110]
})

x = df.iloc[:, :-1]
y = df.iloc[:, -1]

reg = LinearRegression().fit(x,y)

totalError = 0

for idx in range(len(df)):
    actualPrice = df.loc[idx]['Fiyat']
    metrekare = df.loc[idx]['Metrekare']

    expectedPrice = reg.predict([[metrekare]])

    print(f"{df.loc[idx]['Metrekare']} metrekare için gerçek fiyat : {actualPrice}. Tahmin edilen fiyat: {expectedPrice}. HATA: {actualPrice - expectedPrice}")

    totalError = actualPrice - expectedPrice

print(f"\nToplam Hata: {totalError}")

df.head()