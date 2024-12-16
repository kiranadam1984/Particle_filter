import numpy as np
from numpy.linalg import inv

def predict(xkk, Pkk, F, Q):
	xkk_1 = F @ xkk
	Pkk_1 = F @ Pkk @ F.T + Q
	return xkk_1, Pkk_1


def update(xkk_1, Pkk_1, z, H, R):
	nu = z - H @ xkk_1
	S = H @ Pkk_1 @ H.T + R
	K = Pkk_1 @ H.T @ inv(S)
	xkk = xkk_1 + K @ nu
	Pkk = Pkk+1 - K @ S @ K.T
	return xkk, Pkk
	
	
