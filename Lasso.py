import pandas as pd
from sklearn.linear_model import Lasso

df = pd.DataFrame(data={
    'Metrekare': [100, 150, 120, 300, 230],
    'Bina Yaşı': [5, 2, 6, 10, 3],
    'Fiyat': [70, 90, 95, 120, 110]
})

x = df.iloc[:, :-1].values # feature values (take columns until last column)
y = df.iloc[:, -1].values # label values (take only last column)

model = Lasso(alpha=0.1)
model.fit(x,y)

print(model.coef_)
print(model.score(x,y))
print(model.intercept_)

print(model.predict([[300,9]]))