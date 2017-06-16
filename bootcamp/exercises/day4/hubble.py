import numpy as np
import matplotlib.pyplot as plt
import math
from astropy import units as u
from astropy.coordinates import SkyCoord

# Load the data into an array with numpy.genfromtxt. You will find
# 6 columns
# CAT, NUMBER: These two combined give you the name of the galaxy.
# R: distance in Mpc
# V: radial velocity in km/s
# RA, DEC: equatorial coordinates of the galaxy
# IN THIS METHOD ALSO CONVERT TO DEGREES
# RETURN DATA AS str str float(R) float(V) float(RA) float(DEC)
data = np.genfromtxt('table1.txt',skip_header=1)
r = data[:,2]
v = data[:,3] #Max can kill, just added to test model function

def read_data_and_conv():
	data = np.genfromtxt('table1.txt',skipheader=1)
	data[:, 4] = data[:,4]*u.degree
	data[:, 5] = data[:,5]*u.degree
	return data

# Use np.linalg.lstsq to fit a linear regression function and
# determine the slope $H_0$ of the line $V=H_0 R$. For that,
# reshape R as a Nx1 matrix (the so-called design matrix) and
# solve for 1 unknown parameter. Add the best-fit line to the
# plot. Why is there scatter with respect to the best-fit curve?
# s it fair to only fit for the slope and not also for the
# intercept? How would $H_0$ change if you include an intercept
# in the fit?
def model(r, v):
	A = np.vstack([r, np.ones(len(r))])
	m, b = np.linalg.lstsq(A,v)[0]
	plt.scatter(r,v)
	plt.plot(r,m*r)
	#plt.plot(r,m*r+b) #With intercept
	plt.show()
	
	# return slope


# V is a combination of any assumed cosmic expansion and the
# motion of the sun with respect to that cosmic frame.
# Generalize the model to
# $V=H_0 R + X \cos(RA)\cos(DEC) + Y\sin(RA)\cos(DEC)+Z\sin(DEC)$
# d construct a new Nx4 design matrix for the four unknown
# parameters $H_0, X, Y, Z$ to account for the solar motion.
# Use the functions in coordinates.py to convert the coordinate
# strings RA and DEC to floating points coordinates in degrees.
# The resulting $H_0$ is Hubble's own version of the "Hubble
# constant". What do you get?
def better_model(r, v, ra, dec):
	return H_0
