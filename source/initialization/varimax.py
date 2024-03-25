import torch


def varimax(loadings, gamma=1.0, q=20, tol=1e-6):
	"""
	Varimax rotation
	:param loadings: (n_features, n_latent)
	:param gamma: rotation parameter
	:param q: number of iterations
	:param tol: tolerance
	:return: rotated loadings
	"""
	p, k = loadings.shape
	R = torch.eye(k)
	d = 0
	for i in range(q):
		d_old = d
		loadings = loadings @ R
		u, s, v = torch.svd(loadings.T @ (loadings ** 3 - (gamma / p) * (loadings ** 2).sum(0) * loadings))
		R = u @ v
		d = (s ** 2).sum()
		if d_old != 0 and d / d_old < 1 + tol:
			break
	return loadings @ R, R
