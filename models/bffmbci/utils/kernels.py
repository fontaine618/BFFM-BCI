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
		chol = torch.linalg.cholesky(cov)
		inv = torch.cholesky_inverse(chol)
		cholinv = torch.inverse(chol)
		return Kernel(cov, inv, chol, cholinv)