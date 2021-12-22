import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame(data= {
    'Metrekare' : range(1,101,3)
})

df['Fiyat'] = df['Metrekare'] ** 3

plt.plot(df['Metrekare'], df['Fiyat'])