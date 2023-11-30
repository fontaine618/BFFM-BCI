import torch


class WFA:

	def __init__(self, latent_dim: int):
		self._w = None
		self._X = None
		self._mean = None
		self.n_features = None
		self.n_samples = None
		self.latent_dim = latent_dim

	def fit(self, X: torch.Tensor, w: torch.Tensor = None,
	        tol: float = 1e-5, max_iter: int = 1000):
		"""
		:param X: (n_samples, n_features)
		"""
		self.n_samples, self.n_features = X.shape
		self._mean = X.mean(dim=0)
		self._X = X  # - self._mean.reshape((1, -1))
		if w is None:
			w = torch.ones(self.n_samples)
		self._w = w
		self._initialize()
		self._fit(tol, max_iter)

	def _initialize(self):
		# self._loadings = torch.randn(self.n_features, self.latent_dim)
		# self._observation_variance = torch.ones(self.n_features)
		# self._e_step()

		# compute variance
		var = torch.cov(self._X.T)

		# get top eigenvectors
		evals, evecs = torch.linalg.eigh(var)
		evals = evals.flip(0)[:self.latent_dim]
		evecs = evecs.flip(1)[:, :self.latent_dim]
		evecs *= evals.reshape(1, -1).sqrt()

		# reconstructed matrix
		resid = var - evecs @ evecs.T
		diag = resid.diag()

		self._loadings = evecs
		self._observation_variance = diag
		self._e_step()

	def _fit(self, tol: float, max_iter: int):
		prev_llk = self.log_likelihood()
		for i in range(max_iter):
			self._e_step()
			self._m_step()
			llk = self.log_likelihood()
			diff = (llk - prev_llk) / abs(prev_llk)
			prev_llk = llk
			print(f"[WFA] Iteration: {i:>4}  Log-likelihood: {llk:>10.2f}  "
				  f"Relative difference: {diff:>10.2e}  "
				  f"{'(increased)' if diff < 0 else ''}")
			if abs(diff) < tol:
				break
		# post-processing
		L, _ = _varimax(self._loadings)
		self._loadings = L
		order = self._loadings.norm(2, 0).argsort(descending=True)
		self._loadings = self._loadings[:, order]
		self._observation_variance.clamp_min_(1e-5)
		self._e_step()  # run one more E-step to realign the factors
		llk = self.log_likelihood()
		print(f"[WFA] Converged        Log-likelihood: {llk:>10.2f}")

	def _e_step(self):
		var = torch.inverse(torch.einsum(
			"i, jk, j, jl -> ikl",
			self._w,
			self._loadings,
			self._observation_variance.pow(-1),
			self._loadings
		) + torch.eye(self.latent_dim).unsqueeze(0)) * self._w.reshape((-1, 1, 1))
		vec = torch.einsum(
			"jk, j, ij -> ik",
			self._loadings,
			self._observation_variance.pow(-1),
			self._X
		)
		mean = torch.einsum("ijk, ik -> ij", var, vec)
		self._m1 = mean
		self._m2 = var + torch.einsum("ij, ik -> ijk", mean, mean)

	def _m_step(self):
		num = self._X.T @ self._m1
		denum = self._m2.sum(0).inverse()
		self._loadings = num @ denum
		# TODO check this,
		Psi = self._X.T @ self._X - num @ self._loadings.T
		Psi /= self.n_samples
		self._observation_variance = torch.diag(Psi).clamp_min(1e-5)
		# self._observation_variance = self._X.pow(2.).sum(0)
		# self._observation_variance -= (self._X * (self._m1 @ self._loadings.T)).sum(0)
		# self._observation_variance /= self.n_samples

	def log_likelihood(self):
		# we return the marginal log-likelihood
		variance = torch.einsum(
			"i, jk, lk -> ijl",
			self._w,
			self._loadings,
			self._loadings
		) + torch.diag(self._observation_variance).unsqueeze(0)
		log_det = torch.logdet(variance * 2 * torch.pi)
		quad = torch.einsum(
			"ijk, ij, ik -> i",
			torch.inverse(variance),
			self._X,
			self._X
		)
		# torch.distributions.multivariate_normal.MultivariateNormal(
		# 	torch.zeros_like(self._X), variance
		# ).log_prob(self._X)
		return - 0.5 * (log_det + quad).sum().item()

	def q(self):
		logdet = self._observation_variance.log().sum()
		xPx = torch.einsum(
			"ij, j, ij -> i",
			self._X,
			self._observation_variance.pow(-1),
			self._X
		)
		xPTz = torch.einsum(
			"ij, j, jk, ik -> i",
			self._X,
			self._observation_variance.pow(-1),
			self._loadings,
			self._m1
		)
		trTPTzz = (torch.einsum(
			"jk, j, jl -> kl",
			self._loadings,
			self._observation_variance.pow(-1),
			self._loadings
		).unsqueeze(0) * self._m2).sum((1, 2))
		return - 0.5 * (logdet + xPx - 2 * xPTz + trTPTzz).sum().item()


def _varimax(loadings, gamma=1.0, q=20, tol=1e-6):
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
