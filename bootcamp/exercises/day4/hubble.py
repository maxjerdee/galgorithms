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

def read_data_and_conv():
	data = np.genfromtxt('table1.txt', dtype="S8,S8,f8,f8,S8,S8", skip_header=1)
	#print(data)
	r = np.zeros(len(data))
	v = np.zeros(len(data))
	ra = np.zeros(len(data))
	dec = np.zeros(len(data))
	for i in range(len(data)):
		r[i] = data[i][2];
		v[i] = data[i][3];
		val1 = data[i][4].decode("utf-8").split(":");
		val2 = data[i][5].decode("utf-8").split(":");
		ra[i] = float(val1[0]) + float(val1[1])/60 + float(val1[2])/360
		dec[i] = float(val2[0]) + float(val2[1])/60 + float(val2[2])/3600
	return r, v, ra, dec

# Use np.linalg.lstsq to fit a linear regression function and
# determine the slope $H_0$ of the line $V=H_0 R$. For that,
# reshape R as a Nx1 matrix (the so-called design matrix) and
# solve for 1 unknown parameter. Add the best-fit line to the
# plot. Why is there scatter with respect to the best-fit curve?
# s it fair to only fit for the slope and not also for the
# intercept? How would $H_0$ change if you include an intercept
# in the fit?
def model(r, v):
	A = np.vstack([r, np.ones(len(r))]).T
	m, b = np.linalg.lstsq(A,v)[0]
	
	return m


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
	design = np.stack((r, np.cos(ra)*np.cos(dec), np.sin(ra)*np.cos(dec), np.sin(dec)),axis=0).T
	s_inv = np.eye(len(r))
	cov = np.linalg.inv(design.T @ s_inv @ design);
	best = cov @ design.T @ s_inv @ v;
	return best[0], best[1], best[2], best[3]
