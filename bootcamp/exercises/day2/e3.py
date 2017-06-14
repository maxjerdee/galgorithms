import numpy as np
rando = np.random.random(16)*14 + 2
print(rando)
randi = rando.astype(int)
print(randi)
booleanIndex = [(randi[x] >= 5 and randi[x] <= 10 and randi[x]%2 == 0) for x in range(16)]
print(booleanIndex)