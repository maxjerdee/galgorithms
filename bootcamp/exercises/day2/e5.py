import numpy as np
import matplotlib.pyplot as plt
def sim(n):
	darts = np.array([np.random.rand(2)*2 - 1 for i in range(n)])
	plt.plot(darts)
sim(10)
