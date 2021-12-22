import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame(data={
    'Metrekare': [100, 150, 160, 200, 230],
    'Yükseklik': [50, 60, 70, 80, 90],
    'Genişlik': [10, 20, 20, 25, 40],
    'Fiyat': [70, 90, 95, 120, 110]
})

df['Alan'] = df['Yükseklik'] * df['Genişlik']
df.drop(columns=['Yükseklik', 'Genişlik'])

plt.plot(df['Alan'], df['Fiyat'])
