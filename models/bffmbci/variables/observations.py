import torch
import math
from . import ObservedVariable
from ..utils import Kernel


class GaussianObservations(ObservedVariable):
	r"""
	Gaussian observation model using the factor decomposition.

	Dimensions:
	[n_sequences, n_channels, n_timepoints]

	This variable has no children since observed. This variable has the following parents:
	- observation_variance: ObservationVariance of dimension n_channels, sigma
	- loadings: Loadings of dimension [n_channels, n_latent], Theta
	- loading_processes: process extracting the local loadings,
		of dimension [n_sequences, n_latent, n_timepoints], xi
	- factor_processes: process creating the mean, of dimension [n_sequences, n_latent, n_timepoints]
	"""

	_dim_names = ["n_sequences", "n_channels", "n_timepoints"]

	def __init__(self, observation_variance, loadings, loading_processes, factor_processes, value=None):
		self.observation_variance = observation_variance
		self.loadings = loadings
		self.loading_processes = loading_processes
		self.factor_processes = factor_processes
		if value is not None:
			super().__init__(value=value)
		else:
			dim = factor_processes.shape[0], loadings.shape[0], factor_processes.shape[2]
			super().__init__(dim=dim)
		self.parents = {
			"observation_variance": observation_variance,
			"loadings": loadings,
			"loading_processes": loading_processes,
			"factor_processes": factor_processes
		}
		loading_processes._message_from_child = "message_to_loading_processes"
		factor_processes._message_from_child = "message_to_factor_processes"

	@property
	def residuals(self):
		r"""Message to observation variance."""
		return self.data - self.mean()

	@property
	def loading_times_factor_processes(self):
		r"""Message to loading"""
		return torch.einsum(
			"nkt, nkt -> nkt",
			self.loading_processes.data,
			self.factor_processes.data
		)

	@property
	def message_to_loading_processes(self):
		prec, mtp = 0, 0
		return prec, mtp

	@property
	def message_to_factor_processes(self):
		prec, mtp = 0, 0
		return prec, mtp

	def mean(self, loadings=None, loading_processes=None, factor_processes=None):
		if loadings is None:
			loadings = self.loadings.data
		if loading_processes is None:
			loading_processes = self.loading_processes.data
		if factor_processes is None:
			factor_processes = self.factor_processes.data
		mean = torch.einsum(
			"ek, nkt, nkt -> net",
			loadings, loading_processes, factor_processes
		)
		return mean

	@property
	def log_density(self):
		var = self.observation_variance.data.unsqueeze(0).unsqueeze(2)
		N, _, T = self.shape
		llk = -(self.data - self.mean()).pow(2.) / (2. * var)
		llk = llk.sum()
		llk -= 0.5 * N * T * torch.log(var * 2. * math.pi).sum()
		return llk.item()

	def generate(self):
		sd = self.observation_variance.data.sqrt()
		mean = self.mean()
		value = mean + torch.randn_like(mean) * sd.unsqueeze(0).unsqueeze(2)
		self._set_value(value)

	def time_kernel(self, k):
		sigk = self.observation_variance.data[k]
		return Kernel.from_covariance_matrix(cov=torch.eye(self.shape[2])*sigk)
