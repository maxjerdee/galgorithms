import numpy as np
import math
# Gradient Descent
INFINITY = 10**10;
def gradient_descent(f, *args, **kwargs):
	step = 0.1;
	steps = 50;
	curr_args = args;
	for i in range(steps):
		step *= 0.95
		curr_args = curr_args - step*normalize(grad(f,*curr_args));
	return curr_args

# Stopped Accelerated Gradient Descent
def sagd(f, *args, **kwargs):
	step = 0.1;
	steps = 50;
	curr_args = args;
	vel = np.zeros_like(args);
	last = INFINITY;
	for i in range(steps):
		step *= 0.95
		vel = vel - step*normalize(grad(f,*curr_args));
		curr_args = curr_args + vel;
		curr = f(*curr_args)
		if curr > last:
			vel = 0;
		last = curr
	return curr_args

#Accelerated Gradient Descent
def sagd(f, *args, **kwargs):
	step = 0.1;
	steps = 50;
	curr_args = args;
	vel = np.zeros_like(args);
	for i in range(steps):
		step *= 0.95
		vel = vel - step*normalize(grad(f,*curr_args));
		curr_args = curr_args + vel;
		vel *= 0.9
	return curr_args

def normalize(v):
	norm = l2norm(v);
	if norm != 0:
		return v/norm
	return v;

def l2norm(v):
	return math.sqrt(sum(v*v));

def grad(f, *args, **kwargs):
	step = 0.001
	res = np.zeros(len(args))
	diff = np.zeros(len(args))
	for i in range(len(args)):
		diff[i] = step;
		res[i] = (f(*(args + diff)) - f(*args))/step
		diff[i] = 0;
	return res;

def test(x, y):
	return x**2 + y**2


