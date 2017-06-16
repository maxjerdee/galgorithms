import matplotlib.pyplot as plt
import numpy as np

import transit as tr

t,f = tr.read_data(7016.01)
# Inspect the first 10 elements of both t and f
print(t[:10])
print(f[:10])
tr.plot_data(7016.01);
pars = [200, 1.0, 0.2, 0] # t0 [center], T [duration], tau [ingress duration], depth
t,f = tr.read_data(7016.01)
plt.plot(t, tr.trapezoid(pars,t))
plt.ylim(-250,50)
# Vary depth
depths = np.linspace(200,2000,10)
tr.vary_depth(depths); #plots a trapezoid for each depth, with default other parameters
# Vary duration
durations = np.linspace(0.5,1.5,10)
tr.vary_duration(durations);
# Vary tau
taus = np.linspace(0.05,0.5,10)
tr.vary_tau(taus);
# Vary t0
t0s = np.linspace(-0.5,0.5,10)
tr.vary_t0(t0s);
param_guess = [200, 0.8, 0.2, 0]
tr.plot_fit(7016.01, param_guess);

fit = tr.fit_trapezoid(7016.01, method='Nelder-Mead') #this uses scipy.optimize.minimize
fit

tr.plot_fit(7016.01, fit.x);
