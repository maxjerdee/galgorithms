import matplotlib.pyplot as plt
import numpy as np

def gaussian2D(x,y,mean_xy,cov):
	tx = np.stack([x,y], axis=1);
	return np.array([np.dot(np.dot((tx[i] - mean_xy),cov),np.transpose(tx[i] - mean_xy)) for i in range(10000)]);
	#return np.exp(-1/2*np.dot(np.dot(np.transpose(tx - mean_xy),cov),(tx - mean_xy)));

x = np.arange(0.,10.,0.1)
y  = np.arange(0.,10.,0.1)
x,y = np.meshgrid(x,y)
mean_xy = np.array([5.,5.])
cov = np.array([[1,1],[0.,1]])**2.
f = gaussian2D(x.ravel(), y.ravel(), mean_xy, cov)
f.reshape(100,100)

plt.imshow(f.reshape(100,100))
plt.show()
