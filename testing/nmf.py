# Nonnegative Matrix Factorizations
import numpy as np
import math
import optimize as op

#negative log likelyhood of X given D and H as factors
def negll(X, D, H):
	return -norm(X - D @ H)**2

def grad_negll(X, D, H):
	return D.T @ (D @ H - X)

def norm(A):
	return math.sqrt(sum(sum(A*A)));