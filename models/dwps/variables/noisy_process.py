import torch
import torch.linalg
from .variable import Variable, ObservedVariable
from ..utils import Kernel
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.autograd.functional import jacobian
from functorch import vmap, vjp, jacfwd, jacrev
from functorch.experimental import functionalize


class NoisyProcesses(Variable):
	r"""
	GPs around a mean process.

	Dimension equal to the mean process
	[n_sequences, n_latent, n_timepoints]

	Parent is mean, child is observation.
	"""

	def __init__(self, mean: Variable, kernel: Kernel):
		dim = mean.shape
		self.kernel = kernel
		self.mean = mean
		super().__init__(dim=dim, store=False, init=None)
		self.parents = {"mean": mean}
		self.observations: Variable = None
		self._message_from_child: str = None

	def time_kernel(self, k):
		return self.kernel

	def generate(self):
		dist = MultivariateNormal(loc=torch.zeros(self.kernel.shape[0]), scale_tril=self.kernel.chol)
		z = dist.sample(self.shape[:-1])
		value = self.mean.data + z
		self._set_value(value)

	def sample(self, store=False):
		N, K, T = self.shape
		value = self.data
		for k in range(K):
			m0 = self.mean.data[:, k, :]
			p0 = self.kernel.inv
			mtp0 = m0 @ p0

			mtp1, p1 = self._parameters_from_child(k, value)

			mtp = mtp0 + mtp1
			p = p0 + p1
			c = torch.linalg.inv(p)
			m = torch.einsum("ntu, nt -> nu", c, mtp)

			# sample
			for i in range(N):
				dist = MultivariateNormal(loc=m[i, :], covariance_matrix=c[i, :, :])
				value[i, k, :] = dist.sample()

		self._set_value(value, store=store)

	def _parameters_from_child(self, k, value):
		x = self.observations.data
		sig = self.observations.observation_variance.data

		L, fmk = self._get_linear_transform(k, value)
		# compute updates to precision and mtp
		xt = x - fmk
		mtp1 = torch.einsum("e, netu, net -> nu", sig, L, xt)
		p1 = torch.einsum("e, netu, neut -> ntu", sig, L, L)
		return mtp1, p1

	def _get_linear_transform(self, k, value):
		z = value.detach().clone()
		zk = torch.nn.Parameter(z[:, k, :], requires_grad=True)

		def f(zk):
			z[:, k, :] = zk
			return self.observations.mean(factor_processes=z)

		L = jacobian(f, zk, strategy="forward-mode", vectorize=True)  # should be (N x E x T) x (N x T)
		# drop useless parts of L
		L = torch.stack([L[i, :, :, i, :] for i in range(L.shape[0])])  # should be (N x E x T x T)
		fmk = f(0.)
		return L.detach(), fmk.detach()

	# def _get_linear_transform_functorch(self, k, value):
	# 	z = value.detach().clone()
	# 	# get linear transform
	# 	zk = torch.nn.Parameter(z[:, k, :], requires_grad=True)
	#
	# 	def f(zk):
	# 		z[:, k, :] = zk
	# 		return self.observations.mean(factor_processes=z)
	#
	# 	compute_batch_jacobian = vmap(jacrev(f))
	# 	compute_batch_jacobian = vmap(jacfwd(f))
	# 	L = compute_batch_jacobian(zk)
	#
	# 	fmk = f(0.)
	# 	return L.detach(), fmk.detach()