import numpy as np
def go(n):
	darts = np.zeros(n,2)
	for i in range(n):
		darts[i] = np.random.rand(2);
	print(darts)
