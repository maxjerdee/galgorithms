from scipy.stats import multivariate_normal as mvn
import numpy as np
import matplotlib.pyplot as plt
import math

n = 50;
x = np.linspace(-1,1,n)[:,None] @ np.ones(n)[None,:]
y = np.ones(n)[:,None] @ np.linspace(-1,1,n)[None,:]
grid = np.stack((x.ravel(),y.ravel()),axis=1)
res = mvn([0,0],[[0.6,0.2],[0.2,0.2]]).pdf(grid).reshape(n,n);
res += 0.5*mvn([-0.5,0.5],[[0.2,0],[0,0.2]]).pdf(grid).reshape(n,n);
plt.imshow(res)
plt.show()