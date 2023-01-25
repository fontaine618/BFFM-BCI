import torch
import scipy.stats
import numpy as np


class TruncatedStandardMultivariateGaussian:

	def __init__(
			self,
			rotation,
			lower,
			upper
	):
		self._rotation = rotation
		self._lower = lower
		self._upper = upper
		self._dim = lower.shape

	def sample(self, value):
		for i in range(self._dim[0]):
			self._sample_i(value, i)
		return value

	def _sample_i(self, value, i):
		r_i = self._rotation[:, i]
		which = torch.where(torch.arange(0, self._dim[0]) == i, False, True)
		r_mi = self._rotation[:, which]
		x_mi = value[which]
		a_mrx = self._lower - r_mi @ x_mi
		b_mrx = self._upper - r_mi @ x_mi
		j_pos = torch.where(r_i > 0.)[0]
		j_neg = torch.where(r_i < 0.)[0]
		l_pos, l_neg, u_pos, u_neg = -np.inf, -np.inf, np.inf, np.inf
		if len(j_pos):
			l_pos = (a_mrx[j_pos] / r_i[j_pos]).max()
			u_pos = (b_mrx[j_pos] / r_i[j_pos]).min()
		if len(j_neg):
			l_neg = (b_mrx[j_neg] / r_i[j_neg]).max()
			u_neg = (a_mrx[j_neg] / r_i[j_neg]).min()
		l = max(l_pos, l_neg)
		u = min(u_pos, u_neg)
		if l > u:
			print(f"{i}: No valid solution (l > u)")
			return
		elif l > u - 1e-10:
			value[i] = l
			return
		# TODO: find a way to do this on the gpu (CuPy?)
		# print(l.item(), u.item())
		value[i] = scipy.stats.truncnorm(l.item(), u.item(), loc=0, scale=1).rvs()


class TruncatedMultivariateGaussian(TruncatedStandardMultivariateGaussian):

	def __init__(
			self,
			mean,
			covariance=None,
			cholesky=None,
			rotation=None,
			lower=None,
			upper=None
	):
		cholesky, rotation, lower, upper = self._check_args(cholesky, covariance, lower, rotation, upper)
		rotation_ = rotation @ cholesky
		lower_ = lower - rotation @ mean
		upper_ = upper - rotation @ mean
		super().__init__(rotation=rotation_, lower=lower_, upper=upper_)
		self._mean = mean
		self._cholesky = cholesky
		self._cholesky_inverse = torch.linalg.inv(cholesky)

	def _check_args(self, cholesky, covariance, lower, rotation, upper):
		if covariance is not None:
			cholesky = torch.linalg.cholesky(covariance)
		if cholesky is None:
			raise ValueError("need at least covariance or cholesky")
		p = cholesky.shape[0]
		if rotation is None:
			rotation = torch.eye(p)
		if lower is None:
			lower = torch.zeros(p)
		if upper is None:
			upper = torch.ones(p)
		return cholesky, rotation, lower, upper

	def sample(self, value):
		r"""value is a current value to take a single step from"""
		value = self._cholesky_inverse @ (value - self._mean)
		x = super().sample(value)
		return self._mean + self._cholesky @ x
