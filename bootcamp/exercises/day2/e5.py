import numpy as np
import matplotlib.pyplot as plt
def sim(n):
    darts = np.array([np.random.rand(2)*2 - 1 for i in range(n)]);
    within = np.array([(darts[i][0]**2 + darts[i][1]**2) < 1 for i in range(n)]).sum();
    plt.plot(darts);
    plt.show();
    print(float(within*4)/n);
sim(100000)
