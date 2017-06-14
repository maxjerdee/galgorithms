import matplotlib.pyplot as plt
import numpy as np

def gaussian2D(x,y,mean_xy,cov):
	return np.exp(-1/2*np.dot(np.dot(x - mean_xy, cov), y - mean_xy))

x = np.arange(0.,10.,0.1)
y  = np.arange(0.,10.,0.1)
x,y = np.meshgrid(x,y)
mean_xy = np.array([5.,4.])
cov = np.array([[1.,1.],[1.,2.]])**2.

print(x - mean_xy[0])
print(y)
#f = gaussian2D(x.ravel(), y.ravel(), mean_xy, cov)
#f.reshape(100,100)

#plt.imshow(f.reshape(100,100))
