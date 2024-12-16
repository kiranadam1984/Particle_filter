import numpy as np
from numpy.linalg import inv


def kl_divergence(p,q):
	return sum(p[i] * np.log(p[i]/q[i]) for i in range(len(p)))
	
def gaussian_kernel(x, y, A):
	return np.exp(-0.5 * (x-y).T * inv(A) * (x-y))
