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

	def sample(self, store=True):
		# value = self.mean.data.clone().detach()
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


class TruncatedGaussianProcess01(GaussianProcess):

	def __init__(self, n_copies, kernel: Kernel, mean=0.5):
		super().__init__(n_copies, kernel, mean)

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


class NonnegativeGaussianProcess(GaussianProcess):

	def __init__(self, n_copies, kernel: Kernel, mean=1.):
		super().__init__(n_copies, kernel, mean)

	def _dist(self, mean, covariance):
		return TruncatedMultivariateGaussian(mean=mean, covariance=covariance, lower=0., upper=float("inf"))

	def generate(self):
		value = self.mean.data.clone().detach()
		for k in range(value.shape[0]):
			dist = TruncatedMultivariateGaussian(mean=self.mean.data[k, :], covariance=self.kernel.cov,
			                                     lower=0., upper=float("inf"))
			value[k, :] = dist.sample(self.mean.data[k, :])
		self._set_value(value)

	def _sample_k(self, dist, value):
		return dist.sample(value)

	def jitter(self, sd: float = 0.01):
		noise = torch.randn(self.shape) * sd
		self._set_value((self._value * (1 + noise)).clamp(min=0.))