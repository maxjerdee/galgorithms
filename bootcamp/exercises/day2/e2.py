import numpy as np
fib = np.ones((16,64), dtype=np.int16)
for x in range(2,16):
	for y in range(64):
		fib[x][y] = fib[x - 1][y] + fib[x - 2][y]
print(fib)
efib = fib[2:16:3]
print(efib)