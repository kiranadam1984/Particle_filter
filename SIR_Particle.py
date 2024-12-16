import numpy as np
from numpy.linalg import inv
from numpy.random import multivariate_normal, choice
from scipy import stats


def prior_sampling(x, Q, N):

	particles = multivariate_normal(mean = x, cov=Q, size=N)
	weights = np.array([1/N for i in range(N)])
	return particles, weights
	

def weight_normalization(weights):

	normal_weights = []
	sum_weights = np.sum(weights)
	for i in range(len(weights)):
		normal_weights.append(weights[i]/sum_weights)
	return np.array(normal_weights)
	
	
def resampling_particles(particles, weights):

	indices = [i for i in range(len(weights))]
	new_particles = np.zeros(particles.shape)
	for i in range(len(weights)):
		new_particles[i] = particles[choice(indices, size=1, p=weights)]
	
	new_weights = np.zeros(len(weights))
	
	for i in range(len(weights)):
		new_weights[i] = 1/len(weights)
	return new_particles, new_weights
        
        
def particles_effectivness(particles, weights):
	
	rneff = 0
	for i in range(len(weights)):
		rneff += weights[i]**2
		
	neff = 1/rneff
	N = len(weights)
	
	if neff < N/2:
		return resampling_particles(particles, weights)
	else:
		return particles, weights
		
		
def linear_gaussian_importance_distribution(xkk, z, F, Q):

	if isinstance(z, float):
		z1 = np.zeros(xkk.shape)
		z1[0] = z
		z = z1
	else:
		while xkk.shape != z.shape:
			z = np.append(z, 0)
	mu = 0.5*F@xkk + 0.5*z
	f = stats.multivariate_normal(mean=mu, cov=Q)
	return f
	
	
def linear_gaussian_resampling_particle_filter(F, Q, H, R, z, N, K):
	# Initialization
	n = F.shape[0]
	prev_x, prev_w = prior_sampling(np.zeros(n), Q, N)
	w_record = [prev_w]
	x_record = [prev_x]
	m_final = np.zeros((K, n))
	var = np.zeros((K, n))
	
	for i in range(N):
		m_final[0] += prev_w[i] * prev_x[i]
	
	# Calculate weight for each time step
	for k in range(1, K):
		x = np.zeros((N, n))
		w = np.zeros(N)
	
		for i in range(N):
			f = linear_gaussian_importance_distribution(prev_x[i], z[k], F, Q)
			x[i] = np.array(f.rvs(size=1))
			g1 = stats.multivariate_normal(mean=H@x[i], cov=R)
			g2 = stats.multivariate_normal(mean=F@prev_x[i], cov=Q)
			w[i] = (prev_w[i]*g1.pdf(z[k])*g2.pdf(x[i])/f.pdf(x[i])) + np.finfo(float).eps
	
		w = weight_normalization(w)
		x, w = particles_effectivness(x, w)
		m_final[k] = np.average(x, weights=w, axis=0)
		var[k]  = np.average((x - m_final[k])**2, weights=w, axis=0)
		prev_x = x
		prev_w = w
		x_record.append(prev_x)
		w_record.append(prev_w)
	return x_record, w_record, m_final, var

