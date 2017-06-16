import numpy as np
rnd = np.random.RandomState(seed=42)
n_data = 16 # number of data points
a_true = 1.255 # randomly chosen truth
b_true = 4.507
x = rnd.uniform(0,2,size=16);
x.sort();
# evaluate the true model at the given x values
y = a_true*x + b_true
# Heteroscedastic Gaussian uncertainties only in y direction
y_err = rnd.uniform(0.1, 0.2, size=n_data) # randomly generate uncertainty for each datum
y = rnd.normal(y, y_err) # re-sample y data with noise
design = np.stack((np.ones(n_data).T, x.T), axis=1);
sigma = np.diag(y_err**2);
sigma_inv = np.diag(1/y_err**2);
cov = design.T.dot(sigma_inv.dot(design));
best = np.linalg.inv(cov).dot(design.T.dot(sigma_inv.dot(y)))
print(cov)
print(best)