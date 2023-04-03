import warnings

import torch
import torch.linalg
from .variable import Variable, ObservedVariable
from ..utils import Kernel, TruncatedMultivariateGaussian
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.autograd.functional import jacobian


class GaussianProcess(Variable):
	r"""
	Abstract class for n Gaussian processes all with the same kernel matrix.

	We assume the kernel to be fixed and of class Kernel.
	We allow mean to be either a fixed integer or a Variable.
	"""

	_dim_names = ["n_processes", "n_timepoints"]

	def __init__(self, n_copies, kernel: Kernel, mean=0.):
		dim = n_copies, kernel.shape[0]
		self.kernel = kernel
		if isinstance(mean, float):
			mean = torch.full(dim, mean)
			self.mean = ObservedVariable(mean)
		elif isinstance(mean, Variable):
			self.mean = mean
		super().__init__(dim=dim, store=True, init=None)
		self.parents = {"mean": self.mean}
		self.name = None
		self.superposition = None
		self.sample = self.elliptical_slice_sample
		# self.sample = self.direct_sample
		self.n_proposals = 0
		self.n_accepts = 0
		self.n_evals = 0

	def generate(self):
		dist = MultivariateNormal(loc=torch.zeros(self._dim[1]), scale_tril=self.kernel.chol)
		self._set_value(dist.sample((self._dim[0], )) + self.mean.data)

	def _parameters_from_child(self, k, value):
		r"""
		Should return child precision and means times precision to add to prior.
		"""
		L, fmk = self._get_linear_transform(k, value)

		# compute updates to precision and mtp
		Kinv = self.superposition.observations.time_kernel(k).inv
		x = self.superposition.observations.data
		xt = x - fmk
		mtp1 = torch.einsum("netu, tv, nev -> u", L, Kinv, xt)
		p1 = torch.einsum("netu, tv, nevw -> uw", L, Kinv, L)
		return p1, mtp1

	def _get_linear_transform(self, k, value):
		z = value.clone().detach()
		zk = torch.nn.Parameter(z[k, :], requires_grad=True)

		def f(zk):
			z[k, :] = zk
			s = self.superposition.compute_superposition(**{self.name: z})
			sname = self.superposition.name
			out = self.superposition.observations.mean(**{sname: s})
			return out

		L = jacobian(f, zk, strategy="forward-mode", vectorize=True)
		fmk = f(0.)
		return L.detach(), fmk.detach()

	def direct_sample(self, store=True):
		value = self._value.clone().detach()
		for k in range(self._dim[0]):
			p0 = self.kernel.inv
			mtp0 = p0 @ self.mean.data[k, :]

			p1, mtp1 = self._parameters_from_child(k, value)

			prec = p0 + p1
			mtp = mtp0 + mtp1
			c = torch.inverse(prec)
			m = c @ mtp
			dist = self._dist(mean=m, covariance=c)
			value[k, :] = self._sample_k(dist, self.data[k, :])
		self._set_value(value, store=store)

	def _dist(self, mean, covariance):
		return MultivariateNormal(loc=mean, covariance_matrix=covariance)

	def _sample_k(self, dist, value):
		return dist.sample()

	def elliptical_slice_sample(self, store=True):
		value = self._value.clone().detach()
		ogvalue = self._value.clone().detach()
		chol = self.kernel.chol
		# llk_proposed = self.get_log_likelihood(value)
		for k in range(self._dim[0]):
			m0 = self.mean.data[k, :]
			# first iteration, this will take current value
			# subsequent iteration, this will take the llk at the accepted value
			# llk_current = llk_proposed
			llk_current = self.get_log_likelihood(value)
			vk = value[k, :].clone().detach()
			nu = m0 + torch.randn(self._dim[1]) @ chol.T
			# nu = torch.distributions.multivariate_normal.MultivariateNormal(
			# 	loc=m0,
			# 	scale_tril=chol
			# ).sample()
			u = torch.rand(1)
			theta = torch.rand(1) * 2 * torch.pi
			theta_min = theta - 2 * torch.pi
			theta_max = theta
			n_proposals = 0
			n_evals = 0
			while True:
				if n_proposals > 0:  # skip first step
					if theta > 0:
						theta_max = theta
					else:
						theta_min = theta
					theta = torch.rand(1) * (theta_max - theta_min) + theta_min
				mk = torch.cos(theta) * vk + torch.sin(theta) * nu
				# print(theta, mk[0:3])
				n_proposals += 1
				if n_proposals > 30:
					# at this point, since /2 every iteration, we should be back to the original value
					# then we stop and revert to original value
					value[k, :] = ogvalue[k, :]
					# warnings.warn(f"[{self.id}] no proposal accepted after {n_proposals} proposals")
					break
				value[k, :] = mk
				if not self.check_constraints(value):
					continue
				llk_proposed = self.get_log_likelihood(value)
				n_evals += 1
				if llk_proposed > llk_current + torch.log(u).item():
					# print(f"[{self.id}] accepted after {n_proposals} proposal; "
					# 	  f"llk previous: {llk_current:.3f}, "
					# 	  f"llk accepted: {llk_proposed:.3f}")
					break  # accept current proposal
			self.n_evals += n_evals
			self.n_proposals += n_proposals
			self.n_accepts += 1
		# print(f"{self.id} ESS running acceptance rate: "
		# 	  f"{self.n_accepts / self.n_proposals:.3f}")
		self._set_value(value, store=store)

	def check_constraints(self, value):
		return True

	def get_log_likelihood(self, value):
		self._set_value(value, store=False)
		self.superposition.generate()
		return self.superposition.observations.log_density


class TruncatedGaussianProcess01(GaussianProcess):

	def __init__(self, n_copies, kernel: Kernel, mean=0.5):
		super().__init__(n_copies, kernel, mean)
		# self.sample = self.direct_sample

	def _dist(self, mean, covariance):
		return TruncatedMultivariateGaussian(mean=mean, covariance=covariance)

	def generate(self):
		value = self.mean.data.clone().detach()
		for k in range(value.shape[0]):
			dist = TruncatedMultivariateGaussian(mean=self.mean.data[k, :], covariance=self.kernel.cov)
			value[k, :] = dist.sample(self.mean.data[k, :])
		self._set_value(value)

	def _sample_k(self, dist, value):
		return dist.sample(value)

	def jitter(self, sd: float = 0.01):
		noise = torch.randn(self.shape) * sd
		self._set_value((self._value * (1 + noise)).clamp(min=0., max=1.))

	def check_constraints(self, value):
		return (value >= 0.).all() and (value <= 1.).all()


class NonnegativeGaussianProcess(GaussianProcess):

	def __init__(self, n_copies, kernel: Kernel, mean=1.):
		super().__init__(n_copies, kernel, mean)
		# self.sample = self.direct_sample

	def _dist(self, mean, covariance):
		return TruncatedMultivariateGaussian(mean=mean, covariance=covariance, lower=0., upper=100.)

	def generate(self):
		value = self.mean.data.clone().detach()
		for k in range(value.shape[0]):
			dist = TruncatedMultivariateGaussian(mean=self.mean.data[k, :], covariance=self.kernel.cov,
			                                     lower=0., upper=100.)
			value[k, :] = dist.sample(self.mean.data[k, :])
		self._set_value(value)

	def _sample_k(self, dist, value):
		return dist.sample(value)

	def jitter(self, sd: float = 0.01):
		noise = torch.randn(self.shape) * sd
		self._set_value((self._value * (1 + noise)).clamp(min=0.))

	def check_constraints(self, value):
		return (value >= 0.).all()