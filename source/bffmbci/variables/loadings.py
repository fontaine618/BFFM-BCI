import torch
from .variable import Variable, ObservedVariable
from ..utils.inverse_gamma import InverseGamma
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.gamma import Gamma


class Loadings(Variable):
	r"""
	Loading matrix with shrikage prior.

	This variable has dimension [n_channels, n_latent]

	This variable has observations as children which should implement
	the loading_times_factor_processes method

	This variable has heterogeneities and shrinkage factors as parents:
	theta_ek ~ N(0, h_ek / tau_k)
	h_ek ~ InvGamma(gamma/2, gamma/2)
	tau_k = prod_j<=k delta_j
	delta_1 ~ Gamma(a1, 1)
	delta_j ~ Gamma(a2, 1)
	- heterogeneities: of dimension [n_channels, n_latent], phi
	- shrinkage_factor: of dimension [n_latent, ], tau
	- observation_variance: of dimension [n_channels, ], sigma
	"""

	_dim_names = ["n_channels", "latent_dim"]

	def __init__(self, heterogeneities, shrinkage_factor):
		self.heterogeneities = heterogeneities
		self.shrinkage_factor = shrinkage_factor
		super().__init__(heterogeneities.shape, store=True, init=None)
		self.parents = {
			"heterogeneities": heterogeneities,
			"shrinkage_factor": shrinkage_factor
		}
		self.observations = None
		self.observation_variance = None

	def generate(self):
		var = self.heterogeneities.data / self.shrinkage_factor.data.unsqueeze(0)
		theta = Normal(0, var.sqrt())
		self._set_value(theta.sample())

	def sample(self, store=True):
		Theta = self.data
		eta = self.observations.loading_times_factor_processes
		sig2 = self.observation_variance.data
		outer = torch.einsum("nkt, njt -> kj", eta, eta)
		phi = self.heterogeneities.data
		tau = self.shrinkage_factor.data
		x = self.observations.data
		for e in range(self.shape[0]):
			# prec = outer / sig2[e] + torch.diag(phi[e, :] / tau)
			prec = outer / sig2[e] + torch.diag(tau / phi[e, :])
			prec_times_mean = torch.einsum("nt, nkt -> k", x[:, e, :], eta) / sig2[e]
			cov = torch.inverse(prec)
			mean = prec_times_mean @ cov
			dist = MultivariateNormal(loc=mean, covariance_matrix=cov)
			Theta[e, :] = dist.sample()
		self._set_value(Theta, store=store)

	@property
	def squares_times_shrinkage(self):
		return self.data.pow(2) * self.shrinkage_factor.data.unsqueeze(0)

	@property
	def squares_by_heterogeneities(self):
		return self.data.pow(2) / self.heterogeneities.data


class Heterogeneities(Variable):
	r"""
	The variance parameters for the loadings entries.

	This variable is of dimension [n_channels, n_latent]

	This has no parents, only a prior InvGamma(gamma/2, gamma/2).

	the only children in loadings, of dimension [n_channels, n_latent]
	"""

	_dim_names = ["n_channels", "latent_dim"]

	def __init__(self, dim, gamma=3.):
		self._gamma = gamma
		self.loadings: Loadings = None
		super().__init__(dim=dim, store=True, init=None)

	def sample(self, store=True):
		b = (self._gamma + self.loadings.squares_times_shrinkage) / 2
		a = (self._gamma + 1) / 2
		dist = InverseGamma(a, b)
		self._set_value(dist.sample(), store=store)

	def generate(self):
		dist = InverseGamma(self._gamma/2, self._gamma/2)
		self._set_value(dist.sample(self.shape))


class SparseHetereogeneities(Variable):
	r"""
	The variance parameters for the loadings entries.

	This variable is of dimension [n_channels, n_latent]

	This has no parents, only a prior C+(0,1).

	the only children in loadings, of dimension [n_channels, n_latent]

	Sampling is done using auxiliary variables (Makalik and Schmidt 2015)
	"""

	_dim_names = ["n_channels", "latent_dim"]

	def __init__(self, dim, gamma=1.):
		self._gamma = gamma
		self.loadings: Loadings = None
		super().__init__(dim=dim, store=True, init=None)
		self._nu = torch.ones(dim)

	def sample(self, store=True):
		# update value
		a = 1.
		b = 1./self._nu + self.loadings.squares_times_shrinkage/2.
		data = torch.zeros(self.shape)
		for e in range(self.shape[0]):
			for k in range(self.shape[1]):
				dist = InverseGamma(a, b[e, k])
				data[e, k] = dist.sample()
		self._set_value(data, store=store)
		# update nu
		dist = InverseGamma(1., 1./self.data)
		self._nu = dist.sample()

	def generate(self, **kwargs):
		# generate nu
		dist = InverseGamma(0.5, 1.)
		self._nu = dist.sample(self.shape)
		# generate data
		a = 1./2.
		b = 1./self._nu
		data = torch.zeros(self.shape)
		for e in range(self.shape[0]):
			for k in range(self.shape[1]):
				dist = InverseGamma(a, b[e, k])
				data[e, k] = dist.sample()
		self._set_value(data)



class ShrinkageFactor(Variable):
	r"""
	Parameter shrinking loading entries.

	This variable is of dimension [n_latent,]
	We internally keep track of the multiplicative gamma.

	This variable has no parent, only the prior
	tau_k = prod_j<=k delta_j
	delta_1 ~ Gamma(a1, 1)
	delta_j ~ Gamma(a2, 1)
	"""

	_dim_names = ["latent_dim"]

	def __init__(self, n_latent, prior_parameters=(10., 10.)):
		self._a1 = prior_parameters[0]
		self._a2 = prior_parameters[1]
		self._delta = None
		self.loadings: Loadings = None
		super().__init__(dim=(n_latent, ), store=True, init=None)

	def generate(self):
		K = self.shape[0]
		a = torch.ones(K) * self._a2
		a[0] = self._a1
		dist = Gamma(a, 1)
		delta = dist.sample()
		tau = torch.cumprod(delta, 0)
		self._delta = delta
		self._set_value(tau)

	def sample(self, store=True):
		delta = self._delta
		tau = self._value
		E, K = self.loadings.shape
		sbh = self.loadings.squares_by_heterogeneities.sum(0)
		a = torch.ones(K) * self._a2
		a[0] = self._a1
		a += 0.5 * E * torch.linspace(K, 1, K)
		for k in range(K):
			tau_minus = tau / delta[k]
			tau_minus[0:k] = 0
			b = 1 + 0.5 * (tau_minus * sbh).sum()
			dist = Gamma(a[k], b)
			delta[k] = dist.sample()
			tau = torch.cumprod(delta, 0)
		self._delta = delta
		self._set_value(tau, store=store)


class IdentityLoadings(ObservedVariable):

	def __init__(self, dim):
		super(IdentityLoadings, self).__init__(value=torch.eye(dim))