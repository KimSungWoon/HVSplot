import matplotlib.pyplot as plt
import numpy as np
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm


fig = plt.figure()
ax = fig.gca(projection='3d')

data = np.loadtxt('crop_data.csv',unpack=True, delimiter=',',skiprows=1)
X = data[0]
Y = data[1]
Z = data[2]

# Creating figure
fig = plt.figure(figsize=(16, 9))
ax = plt.axes(projection="3d")

# Add x, y gridlines
ax.grid(b=True, color='grey',
        linestyle='-.', linewidth=0.3,
        alpha=0.2)

# Creating color map
my_cmap = plt.get_cmap('hsv')

# Creating plot
sctt = ax.scatter3D(X, Y, Z,
                    alpha=0.8,
                    c=(X + Y + Z),
                    cmap=my_cmap,
                    marker='^')

plt.title("animation 3D")
ax.set_xlabel('R', fontweight='bold')
ax.set_ylabel('G', fontweight='bold')
ax.set_zlabel('B', fontweight='bold')
fig.colorbar(sctt, ax=ax, shrink=0.5, aspect=5)




# ax.scatter3D(X,Y,Z, color = 'green')
#
# ax.set_xlabel('R')
# ax.set_ylabel('G')
# ax.set_zlabel('B')
#
# plt.suptitle('animation_cluster',fontsize=5)
plt.show()