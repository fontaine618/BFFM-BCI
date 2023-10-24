import torch
import torch.linalg
from .variable import Variable
from ..utils import Kernel
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.autograd.functional import jacobian, hessian


class NoisyProcesses(Variable):
	r"""
	GPs around a mean process.

	Dimension equal to the mean process
	[n_sequences, n_latent, n_timepoints]

	Parent is mean, child is observation.
	"""

	_dim_names = ["n_sequences", "n_processes", "n_timepoints"]

	def __init__(self, mean: Variable, kernel: Kernel):
		dim = mean.shape
		self.kernel = kernel
		self._mean = mean
		super().__init__(dim=dim, store=False, init=None)
		self.parents = {"mean": mean}
		self.observations: Variable = None
		self._message_from_child: str = None

	def time_kernel(self, k):
		return self.kernel

	def generate(self):
		dist = MultivariateNormal(loc=torch.zeros(self.kernel.shape[0]), scale_tril=self.kernel.chol)
		z = dist.sample(self.shape[:-1])
		value = self._mean.data + z
		self._set_value(value)

	@property
	def conditional_posterior_mean(self):
		N, K, T = self.shape
		value = self.data
		for k in range(K):
			m0 = self._mean.data[:, k, :]
			p0 = self.kernel.inv
			mtp0 = m0 @ p0

			mtp1, p1 = self._parameters_from_child(k, value)

			mtp = mtp0 + mtp1
			p = p0.unsqueeze(0) + p1
			c = torch.linalg.inv(p)
			m = torch.einsum("ntu, nt -> nu", c, mtp)
			value[:, k, :] = m
		return value

	def posterior_mean_batched(self, batchsize: int = 10):
		# FIXME: this doesnt work for now
		oldvalue = self.data.clone().detach()
		newvalue = self.data.clone().detach()
		N, K, T = self.shape
		nbatches = N // batchsize
		if N % batchsize > 0:
			nbatches += 1
		for i in range(nbatches):
			which = slice(i * batchsize, min((i + 1) * batchsize, N))
			n = which.stop - which.start
			value = torch.nn.Parameter(self.data[which, ...].reshape(n, -1), requires_grad=True)
			def f(z):
				self.data[which, ...] = z.reshape(n, K, T)
				return self.log_density_per_sequence[which].sum() + \
					  self.observations.log_density_per_sequence[which].sum()
			def grad(z):
				return jacobian(f, z, create_graph=True, strategy="reverse-mode", vectorize=True).sum(0)

			g = jacobian(f, value, strategy="reverse-mode", vectorize=True)
			H = jacobian(grad, value, strategy="reverse-mode", vectorize=False).permute(1, 0, 2)

			newvalue[which, ...] = (value - torch.linalg.solve(H, g)).reshape(n, K, T)

			print(f"{i}/{nbatches} ({n}/{N}): {f(value).item()}")

		self.data = oldvalue
		return newvalue.detach()



	@property
	def posterior_mean(self):

		oldvalue = self.data.clone().detach()
		N, K, T = self.shape
		value = torch.nn.Parameter(self.data.reshape(N, -1), requires_grad=True)
		def f(z):
			self.data = z.reshape(N, K, T)
			return self.log_density_per_sequence.sum() + \
				  self.observations.log_density_per_sequence.sum()
		def grad(z):
			return jacobian(f, z, create_graph=True, strategy="reverse-mode", vectorize=False).sum(0)

		g = jacobian(f, value, strategy="reverse-mode", vectorize=True)
		H = jacobian(grad, value, strategy="reverse-mode", vectorize=False).permute(1, 0, 2)

		value = value - torch.linalg.solve(H, g)

		self.data = oldvalue
		return value.reshape(self.shape).detach()



	@property
	def posterior_mean_by_conditionals(self):

		oldvalue = self.data.clone().detach()

		prevllk = self.log_density + self.observations.log_density
		for i in range(2000):
			self.data = self.conditional_posterior_mean
			llk = self.log_density + self.observations.log_density
			print(f"{i}: {llk}")
			if abs(prevllk - llk) / abs(llk) < 1e-7:
				break
			prevllk = llk

		postmean = self.data.clone().detach()
		self.data = oldvalue
		return postmean


	def sample(self, store=False):
		N, K, T = self.shape
		value = self.data
		for k in range(K):
			m0 = self._mean.data[:, k, :]
			p0 = self.kernel.inv
			mtp0 = m0 @ p0

			mtp1, p1 = self._parameters_from_child(k, value)

			mtp = mtp0 + mtp1
			p = p0 + p1
			c = torch.linalg.inv(p)
			m = torch.einsum("ntu, nt -> nu", c, mtp)

			# sample
			for i in range(N):
				if torch.isnan(m[i, :]).any():
					raise RuntimeError("NaN in mean")
				dist = MultivariateNormal(loc=m[i, :], covariance_matrix=c[i, :, :])
				value[i, k, :] = dist.sample()

		self._set_value(value, store=store)

	def _parameters_from_child(self, k, value):
		x = self.observations.data
		siginv = 1. / self.observations.observation_variance.data
		L, fmk = self._get_linear_transform(k, value)
		# compute updates to precision and mtp
		xt = x - fmk
		mtp1 = torch.einsum("e, net, net -> nt", siginv, L, xt)
		p1 = torch.einsum("e, net, net -> nt", siginv, L, L)
		p1 = torch.diag_embed(p1)
		return mtp1, p1

	def _get_linear_transform(self, k, value):
		z = value.detach().clone()
		lk = self.observations.loadings.data[:, k] # E
		lpk = self.observations.loading_processes.data[:, k, :] # N x T
		N, K, T = self.observations.loading_processes.shape

		jac = torch.einsum(
			"e, nt -> net",
			lk, lpk
		) # N x E x T, but really should be N x E x T x T
		z[:, k, :] = 0.
		return jac, self.observations.mean(factor_processes=z)

	@property
	def log_density_per_sequence(self):
		chol = self.kernel.chol
		mean = self._mean.data
		value = self.data
		dist = torch.distributions.multivariate_normal.MultivariateNormal(loc=mean, scale_tril=chol)
		logp = dist.log_prob(value)
		return logp.sum(1)

	@property
	def log_density(self):
		return self.log_density_per_sequence.sum().item()

	@staticmethod
	def mean(factor_processes):
		# this is to satisfy the interface with GaussianProcesses sampling
		return factor_processes