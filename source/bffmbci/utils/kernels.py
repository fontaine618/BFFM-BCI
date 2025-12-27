import torch


class Kernel:

	def __init__(
			self,
			covariance_matrix,
			inverse_matrix,
			cholesky_factor,
			cholesky_inverse
	):
		self._covariance_matrix = covariance_matrix
		self._inverse_matrix = inverse_matrix
		self._cholesky_factor = cholesky_factor
		self._cholesky_inverse = cholesky_inverse

	@property
	def shape(self):
		return self.cov.shape

	@property
	def cov(self):
		return self._covariance_matrix

	@property
	def inv(self):
		return self._inverse_matrix

	@property
	def chol(self):
		return self._cholesky_factor

	@property
	def cholinv(self):
		return self._cholesky_inverse

	@classmethod
	def from_covariance_matrix(cls, cov):

		# try Cholesky, add small diagonal if matrix is singular/ill-conditioned
		jitter = 0.0
		chol = None
		for _ in range(20):
			try:
				mat = cov + torch.eye(cov.shape[-1], device=cov.device, dtype=cov.dtype) * jitter
				chol = torch.linalg.cholesky(mat)
				cov = mat  # use jittered matrix for downstream facts
				break
			except RuntimeError:
				jitter = 1e-6 if jitter == 0.0 else jitter * 2.0
		if chol is None:
			raise RuntimeError("Cholesky decomposition failed (matrix may be singular)")

		inv = torch.cholesky_inverse(chol)
		cholinv = torch.inverse(chol)
		return Kernel(cov, inv, chol, cholinv)

	@classmethod
	def from_covariance_matrix_unsafe(cls, cov):
		chol = torch.linalg.cholesky(cov)
		inv = torch.cholesky_inverse(chol)
		cholinv = torch.inverse(chol)
		return Kernel(cov, inv, chol, cholinv)

	@classmethod
	def identity_times(cls, shape, value):
		cov = torch.eye(shape) * value
		inv = torch.eye(shape) / value
		chol = torch.eye(shape) * value.sqrt()
		cholinv = torch.eye(shape) / value.sqrt()
		return Kernel(cov, inv, chol, cholinv)