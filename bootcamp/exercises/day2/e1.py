import numpy as np
import estimate_pi as ep
a1 = np.ones((4,4), dtype=np.int16)
a1[2][3] = 2
a1[3][1] = 6
print(a1)
a2 = np.zeros((5,5))
for i in range(6):
	a2[i - 1][i - 1] = i
print(a2)

ep.go(10)