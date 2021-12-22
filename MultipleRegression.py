import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

df = pd.DataFrame(data={
    'Metrekare':[100,150,120,300,230],
    'Bina Yaşı' : [5,2,6,10,3],
    'Fiyat':[70,90,95,120,110]
})

fig = plt.figure()
ax = Axes3D(fig)
ax.plot_trisurf(df['Metrekare'], df['Bina Yaşı'], df['Fiyat'], cmap=cm.jet, linewidth=0.2)
ax.set_xlabel('Metrekare')
ax.set_ylabel('Fiyat')
ax.set_zlabel('Bina Yaşı')

plt.show()