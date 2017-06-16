
# Write a function that takes the object numer (e.g. 7016.01) 
# as an argument and returns two arrays: time and flux, read 
# from the data file ('data/7016.01.txt').
def read_data(object_num):
	pass





# Write a function that takes the object number as an argument
# and plots time vs. flux of the data returned by (1).
def plot_data(object_num):
	pass





# Write a function implementing the trapezoid model. It should 
# take four parameters (delta, T, tau, t0---depth, duration, 
# ingress duration, and center time) and return a trapezoid 
# as a function of time.
def trapezoid(pars, t):
	pass





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
	pass




# Use scipy.optimize.minimize to find the best-fit parameters for 
# the 7016.01 data set, and display these results using (6).
def fit_trapezoid(object_num, *args):
	pass

