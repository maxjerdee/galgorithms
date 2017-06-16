# from scipy.stats import multivariate_normal as mvn
# print(mvn([0,0],[[1,0],[0,1]]).pdf([[0,0],[1,1]])) 
import numpy as np
import matplotlib.pyplot as plt
import math
t = np.linspace(1,10,100);
fig, ax = plt.subplots();
ax.scatter(t, t+10)
ax.plot(t, t+2)
plt.show()