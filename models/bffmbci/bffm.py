from typing import Tuple, Union
import torch
import numpy as np
import scipy.linalg
import torch.nn.functional as F

from models.bffmbci.utils import Kernel
from models.bffmbci.variables import SequenceData, SMGP, Superposition
from models.bffmbci.variables import GaussianObservations
from models.bffmbci.variables import ObservationVariance
from models.bffmbci.variables import Loadings, Heterogeneities, ShrinkageFactor
from models.bffmbci.variables import NoisyProcesses


class BFFModel:

	def __init__(
			self,
			stimulus_order: torch.Tensor,
			target_stimulus: torch.Tensor,
			stimulus_window: int,
			stimulus_to_stimulus_interval: int,
			latent_dim: int,
			sequences: Union[torch.Tensor, None] = None,
			n_stimulus: Tuple[int] = (6, 6),
			n_sequences: int = 15*19,
			n_channels: int = 15,
			**kwargs
	):
		self._dimensions = {
			"n_sequences": n_sequences,
			"n_timepoints": (sum(n_stimulus) - 1) * stimulus_to_stimulus_interval + stimulus_window,
			"n_channels": n_channels,
			"n_stimulus": n_stimulus,
			"stimulus_window": stimulus_window,
			"stimulus_to_stimulus_interval": stimulus_to_stimulus_interval,
			"latent_dim": latent_dim
		}
		self.prior_parameters = {}
		self._initialize_prior_parameters(**kwargs)
		self.variables = {}
		self._prepare_model(
			sequences=sequences,
			stimulus_order=stimulus_order,
			target_stimulus=target_stimulus
		)
		self._sampling_order = [
			"factor_processes",
			"mean_factor_processes",
			"smgp_factors",

			"loading_processes",
			"smgp_loadings",

			"loadings",
			"shrinkage_factor",
			"heterogeneities",

			"observation_variance"
		]

	def _prepare_model(
			self,
			sequences: Union[torch.Tensor, None],
			stimulus_order: torch.Tensor,
			target_stimulus: torch.Tensor
	):
		parms = self.prior_parameters
		dims = self._dimensions
		observation_variance = ObservationVariance(
			n_channels=dims["n_channels"],
			prior_parameters=parms["observation_variance"]
		)
		heterogeneities = Heterogeneities(
			dim=(dims["n_channels"], dims["latent_dim"]),
			gamma=parms["heterogeneities"]
		)
		shrinkage_factor = ShrinkageFactor(
			n_latent=dims["latent_dim"],
			prior_parameters=parms["shrinkage_factor"]
		)
		loadings = Loadings(
			heterogeneities=heterogeneities,
			shrinkage_factor=shrinkage_factor
		)
		sequence_data = SequenceData(
			order=stimulus_order,
			target=target_stimulus
		)
		tmat = scipy.linalg.toeplitz(parms["kernel_gp"][0] ** np.arange(dims["stimulus_window"]))
		kernel_gp = Kernel.from_covariance_matrix(torch.Tensor(tmat) * parms["kernel_gp"][1])
		tmat = scipy.linalg.toeplitz(parms["kernel_tgp"][0] ** np.arange(dims["stimulus_window"]))
		kernel_tgp = Kernel.from_covariance_matrix(torch.Tensor(tmat) * parms["kernel_tgp"][1])
		smgp_loadings = SMGP(dims["latent_dim"], kernel_gp, kernel_tgp)
		loading_processes = Superposition(
			smgp=smgp_loadings,
			sequence_data=sequence_data,
			stimulus_to_stimulus_interval=dims["stimulus_to_stimulus_interval"],
			window_length=dims["stimulus_window"]
		)
		loading_processes.name = "loading_processes"
		smgp_factors = SMGP(dims["latent_dim"], kernel_gp, kernel_tgp)
		mean_factor_processes = Superposition(
			smgp=smgp_factors,
			sequence_data=sequence_data,
			stimulus_to_stimulus_interval=dims["stimulus_to_stimulus_interval"],
			window_length=dims["stimulus_window"]
		)
		mean_factor_processes.name = "factor_processes"
		tmat = scipy.linalg.toeplitz(parms["kernel_factor"][0] ** np.arange(dims["n_timepoints"]))
		kernel_factor = Kernel.from_covariance_matrix(torch.Tensor(tmat) * parms["kernel_factor"][1])
		factor_processes = NoisyProcesses(
			mean=mean_factor_processes,
			kernel=kernel_factor
		)
		observations = GaussianObservations(
			observation_variance=observation_variance,
			loadings=loadings,
			loading_processes=loading_processes,
			factor_processes=factor_processes,
			value=sequences
		)

		shrinkage_factor.add_children(loadings=loadings)
		heterogeneities.add_children(loadings=loadings)
		loadings.add_children(
			observations=observations,
			observation_variance=observation_variance
		)
		observation_variance.add_children(observations=observations)
		smgp_loadings.add_children(superposition=loading_processes)
		smgp_factors.add_children(superposition=mean_factor_processes)
		loading_processes.add_children(observations=observations)
		mean_factor_processes.add_children(child=factor_processes, observations=observations)
		factor_processes.add_children(observations=observations)

		self.variables = {
			"observation_variance": observation_variance,
			"heterogeneities": heterogeneities,
			"shrinkage_factor": shrinkage_factor,
			"loadings": loadings,
			"sequence_data": sequence_data,
			"smgp_loadings": smgp_loadings,
			"loading_processes": loading_processes,
			"smgp_factors": smgp_factors,
			"mean_factor_processes": mean_factor_processes,
			"factor_processes": factor_processes,
			"observations": observations
		}

	@classmethod
	def generate_from_dimensions(
			cls,
			n_sequences: int = 15*19,
			n_stimulus: Tuple[int] = (6, 6),
			n_channels: int = 15,
			stimulus_window: int = 55,
			stimulus_to_stimulus_interval: int = 10,
			latent_dim: int = 3,
			**kwargs
	):
		stimulus_order, target_stimulus = _create_sequence_data(n_sequences, n_stimulus)
		return cls(
			sequences=None,
			stimulus_order=stimulus_order,
			target_stimulus=target_stimulus,
			stimulus_window=stimulus_window,
			stimulus_to_stimulus_interval=stimulus_to_stimulus_interval,
			latent_dim=latent_dim,
			n_sequences=n_sequences,
			n_channels=n_channels,
			n_stimulus=n_stimulus,
			**kwargs
		)

	def set(self, **kwargs):
		for k, v in kwargs.items():
			self.variables[k].data = v

	def _initialize_prior_parameters(self, **kwargs):
		prior_parameters = {
			"observation_variance": (1., 10.),
			"heterogeneities": 3.,
			"shrinkage_factor": (1., 10.),
			"kernel_gp": (0.99, 1.),
			"kernel_tgp": (0.99, 0.5),
			"kernel_factor": (0.99, 0.01)
		}
		for k in prior_parameters.keys():
			if k in kwargs:
				prior_parameters[k] = kwargs[k]
		self.prior_parameters = prior_parameters

	def sample(self, sampling_order=None, random=True):
		if sampling_order is None:
			sampling_order = self._sampling_order
		if random:
			sampling_order = np.random.choice(sampling_order, len(sampling_order), replace=False)
		for var in sampling_order:
			self.variables[var].sample()


def _create_sequence_data(n_sequences, n_stimulus):
	stimulus_order = torch.hstack([
		torch.vstack([torch.randperm(n_stimulus[i]) for _ in range(n_sequences)]) +
		(n_stimulus[i - 1] if i > 0 else 0)
		for i in range(len(n_stimulus))
	])
	target_stimulus = torch.hstack([
		torch.randint(
			low=n_stimulus[i - 1] if i > 0 else 0,
			high=(n_stimulus[i - 1] if i > 0 else 0) + n_stimulus[i],
			size=(n_sequences, 1)
		)
		for i in range(len(n_stimulus))
	])
	target_stimulus = F.one_hot(target_stimulus, num_classes=sum(n_stimulus)).max(1).values
	return stimulus_order, target_stimulus