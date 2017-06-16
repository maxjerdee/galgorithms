# Hubble Project Tester
import numpy as np
import matplotlib.pyplot as plt
import math
import hubble as hb

MPC_OVER_KM= 3.086*10**19
r, v, ra, dec = hb.read_data_and_conv();
fig, ax = plt.subplots();
ax.scatter(r, v);
slope = hb.model(r,v);
ax.plot(r, slope*r);
plt.show();
H_0, X, Y, Z = hb.better_model(r,v,ra,dec);
bv = v - X*np.cos(ra)*np.cos(dec) - Y*np.sin(ra)*np.cos(dec) - Z*np.sin(dec);
fig, ax = plt.subplots();
fig2, ax2 = plt.subplots();
ax2.scatter(r, bv);
ax2.plot(r, H_0*r);
plt.show();
age = MPC_OVER_KM/H_0;
print("Predicted Age of the Universe: {0}".format(age));