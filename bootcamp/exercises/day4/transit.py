import numpy as np
<<<<<<< HEAD
import matplotlib.pyplot as plt
import math
# Write a function that takes the object numer (e.g. 7016.01) 
# as an argument and returns two arrays: time and flux, read 
=======

# Write a function that takes the object numer (e.g. 7016.01)
# as an argument and returns two arrays: time and flux, read
>>>>>>> 7e904a73b98df4ff8d273d5f289cebe690c65ddd
# from the data file ('data/7016.01.txt').
def read_data(object_num):
	pass





# Write a function that takes the object number as an argument
# and plots time vs. flux of the data returned by (1).
def plot_data(object_num):
	t, f = read_data(object_num);
	plot(t, f)	





# Write a function implementing the trapezoid model. It should
# take four parameters (delta, T, tau, t0---depth, duration,
# ingress duration, and center time) and return a trapezoid
# as a function of time.
def trapezoid(pars, t):
	# initialize variable
	delta,T,tau,t0 = pars
	# create useful vars
	dt = np.abs(t-t0)
	in_rad = T/2 - tau

	# return function based on conditions
	if dt < ir:
		return 0 - delta
	elif dt < T/2:
		return 0 - delta + (delta/tau)*(dt - in_rad)
	else:
		return 0


# Make four different plots that show how the trapezoid shape
# changes when you vary each parameter independently (maybe 10
# examples per plot).
def vary_depth(depths):
	pass

def vary_duration(depths):
	pass

def vary_tau(depths):
	pass

def vary_t0(depths):
	pass




# Eyeball the plot of the actual 7016.01 transit signal, and
# try to guess what parameters might fit the data best. Overplot
# the model on top of the data, and make a plot of the residuals
# data - model) in a subplot. Your first guess doesn't have to
# be spot-on---the residuals should tell you where you're off
# the most.
# Use what you did in (5) to write a function that takes the
# object number and a parameter vector, and then makes the data
# + models + residuals plot.
def tr.plot_fit(object_num, param_guess):
	t, f = read_data(object_num);
	m = trapezoid(param_guess, t);
	plot()




# Use scipy.optimize.minimize to find the best-fit parameters for
# the 7016.01 data set, and display these results using (6).
def fit_trapezoid(object_num, *args):
	pass
