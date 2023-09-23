import numpy as np
import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D

x =np.linspace(-100000,100000,1000)
y =np.linspace(-100000,100000,1000)
X,Y = np.meshgrid(x,y)

Z = 100 * (X * X - Y) * (X * X - Y) + (X - 1) * (X - 1);
fig = plt.figure()
ax = fig.add_subplot(111,projection='3d')
ax.plot_surface(X,Y,Z)

# plt.show()

a= 35.6596
b= 1271.43
c= 100*(a**2-b)**2+(a-1)**2
x = 105300
y = -105300
z=  1
d = (x - 2) * (x - 3) + 2 * x + 2 * x * y + y * y + (z + 1) * (z + 1)
print(d)