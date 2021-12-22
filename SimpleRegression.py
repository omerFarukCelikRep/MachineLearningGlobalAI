import pandas as pd
import seaborn as sns

df = pd.DataFrame(data={
    'Metrekare': [100, 150, 120, 300, 230],
    'Fiyat': [70, 90, 95, 120, 110]
})

sns.lmplot(x='Metrekare', y='Fiyat', data=df)
