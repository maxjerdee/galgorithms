import numpy as np
import matplotlib.pyplot as plt
import math

def cov(x, y):
	xbar = x.mean();
	ybar = y.mean();
	return np.sum((x - xbar)*(y - ybar))/(len(x) - 1);
def binomial(n,m,p):
	return p**m*(1-p)**(n-m)*math.factorial(n)/(math.factorial(n-m)*math.factorial(m))
def neg_loglik(thetas, *args):
	#print(args)
	return -np.sum([binomial(args[0][0],x,thetas[z]) for (x, z) in zip(args[0][1], args[0][2])]);
def bern_sim(n,p):
	return np.sum([np.random.random() < p for i in range(n)])
def grad(f, x, y, p, *args):
	#print(args)
	return [(f([x + p,y], args[0]) - f([x,y], args[0]))/p, (f([x,y + p], args[0]) - f([x,y], args[0]))/p];
def gdescend(f, x0, y0, *args):
	#print(args)
	cx = x0;
	cy = y0;
	vx = 0;
	vy = 0;
	for i in range(1000):
		ng = grad(f,cx,cy, 0.001, args);
		mag = np.linalg.norm(ng);
		if mag > 0:
			ng /= mag;
		vx -= ng[0]*0.1;
		vy -= ng[1]*0.1;
		vx *= 0.9;
		vy *= 0.9;
		cx += vx;
		cy += vy;
	return [cx,cy]
n = 10;
thetas = [0.3,0.8]
zs = [0,0,1,0,1]
xs = [bern_sim(n, thetas[zs[i]]) for i in range(len(zs))]
print(gdescend(neg_loglik, 0.1, 0.1, n, xs, zs));
print(xs)
x_grid = np.linspace(0,1,100);
y_grid = np.linspace(0,1,100);
x_grid, y_grid = np.meshgrid(x_grid, y_grid)
xy_grid = np.vstack((x_grid.ravel(), y_grid.ravel())).T
result = np.array([neg_loglik(xy_grid[i], [n, xs, zs]) for i in range(len(xy_grid))]);
plt.imshow(result.reshape(100,100))
plt.show()
#print(neg_loglik(x, n, xs, zs)