from IPython import display
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

# rnd = np.random.RandomState(seed=42)
# n_data = 16 # number of data points
# a_true = 1.255 # randomly chosen truth
# b_true = 4.507
# x = rnd.uniform(0,2,size=16);
# x.sort();
# # evaluate the true model at the given x values
# y = a_true*x + b_true
# # Heteroscedastic Gaussian uncertainties only in y direction
# y_err = rnd.uniform(0.1, 0.2, size=n_data) # randomly generate uncertainty for each datum
# y = rnd.normal(y, y_err) # re-sample y data with noise
# plt.errorbar(x, y, y_err, linestyle='none', marker='o', ecolor='#666666')
# plt.xlabel('$x$')
# plt.ylabel('$y$')
# plt.tight_layout()
# plt.show()
def line_model(pars, x):
	return pars[0]*x + pars[1];
def weighted_absolute_deviation(pars, x, y, y_err):
    return np.sum(np.absolute((line_model(pars, x) - y)/y_err));

def weighted_squared_deviation(pars, x, y, y_err):
    return np.sum(((line_model(pars, x) - y)/y_err)**2);
_pars = [1., -10.]
_x = np.arange(16)
_y = _x
_yerr = np.ones_like(_x)

truth = np.array([-10.,  -9.,  -8.,  -7.,  -6.,  -5.,  -4.,  -3.,  -2.,  -1.,   0., 1.,   2.,   3.,   4.,   5.])
assert np.allclose(line_model(_pars, _x), truth), 'Error in line_model() function!'
assert weighted_absolute_deviation(_pars, _x, _y, _yerr) == 160., 'Error in weighted_absolute_deviation() function!'
assert weighted_squared_deviation(_pars, _x, _y, _yerr) == 1600., 'Error in weighted_squared_deviation() function!'
x0 = [1., 1.] # starting guess for the optimizer 

result_abs = minimize(weighted_absolute_deviation, x0=x0, 
                      args=(x, y, y_err), # passed to the weighted_*_deviation function after pars 
                      method='BFGS') # similar to Newton's method

result_sq = minimize(weighted_squared_deviation, x0=x0, 
                     args=(x, y, y_err), # passed to the weighted_*_deviation function after pars
                     method='BFGS')

best_pars_abs = result_abs.x
best_pars_sq = result_sq.x