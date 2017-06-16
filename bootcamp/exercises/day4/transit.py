import numpy as np
import matplotlib.pyplot as plt
import math
from scipy import optimize
# Write a function that takes the object numer (e.g. 7016.01)
# as an argument and returns two arrays: time and flux, read
# from the data file ('data/7016.01.txt').
def read_data(object_num):
        time = np.loadtxt('%s.txt'%(object_num),usecols = [0])
        flux = np.loadtxt('%s.txt'%(object_num),usecols= [1])
        return time, flux






# Write a function that takes the object number as an argument
# and plots time vs. flux of the data returned by (1).
def plot_data(object_num):
	t, f = read_data(object_num)
	plt.plot(t, f)
	plt.show()






# Write a function implementing the trapezoid model. It should
# take four parameters (delta, T, tau, t0---depth, duration,
# ingress duration, and center time) and return a trapezoid
# as a function of time.
def trapezoid(pars, t):
	# initialize variable
	delta,T,tau,t0 = pars;
	# create useful vars
	dt = np.abs(t-t0);
	ir = T/2 - tau;
	f = np.zeros(t.shape); 
    # default function array

	# create bool arrays
	bottom = [dt < ir];
	slope = [dt >= ir & dt < T/2];

	# change values of default function array based on conditions
<<<<<<< HEAD
	f[bottom] = 0 - delta # bottom of trapezoid
	f[slope] = 0 - delta + (delta/tau)*(dt[slope] - in_rad)
=======
	f[bottom] = 0 - delta; # bottom of trapezoid
	f[slope] = 0 - delta + (delta/tau)(dt[slope] - in_rad);
>>>>>>> 01d0c0cf5807e5c2536fdfe69be3ebb885c7e962

	return f

# Make four different plots that show how the trapezoid shape
# changes when you vary each parameter independently (maybe 10
# examples per plot).
def vary_depth(depths):
	delta = depths
	pars = delta, T, tau, t0
	plt.plot(t, trapezoid(pars,t))
	plt.show()

def vary_duration(durations):
	T = durations
	pars = delta, T, tau, t0
	plt.plot(t, trapezoid(pars,t))
	plt.show()

def vary_tau(taus):
	tau = taus
	pars = delta, T, tau, t0
	plt.plot(t, trapezoid(pars,t))
	plt.show()

def vary_t0(t0s):
	t0 = t0s
	pars = delta, T, tau, t0
	plt.plot(t, trapezoid(pars,t))
	plt.show()





# Eyeball the plot of the actual 7016.01 transit signal, and
# try to guess what parameters might fit the data best. Overplot
# the model on top of the data, and make a plot of the residuals
# data - model) in a subplot. Your first guess doesn't have to
# be spot-on---the residuals should tell you where you're off
# the most.
# Use what you did in (5) to write a function that takes the
# object number and a parameter vector, and then makes the data
# + models + residuals plot.
def plot_fit(object_num, param_guess):
	t, f = read_data(object_num);
	m = trapezoid(param_guess, t);
	r = f - m;
	z = np.zeros_like(r);
	fig, ax = plt.subplots();
	ax.scatter(t, f);
	ax.plot(t, m);
	plt.show();
	fig2, ax2 = plt.subplots();
	ax2.scatter(t, r);
	ax2.plot(t, z);
	plt.show();




# Use scipy.optimize.minimize to find the best-fit parameters for
# the 7016.01 data set, and display these results using (6).
def fit_trapezoid(object_num, *args):
    return scipy.optimize.minimize(trapezoid(pars,t),pars)
