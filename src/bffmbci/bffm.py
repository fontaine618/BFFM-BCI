from typing import Tuple, Union
import torch
import pickle
import numpy as np
import scipy.linalg
import torch.nn.functional as F

from .utils import Kernel
from .variables import SequenceData, SMGP, Superposition, IndependentSMGP, NonnegativeSMGP
from .variables import GaussianObservations
from .variables import ObservationVariance
from .variables import Loadings, Heterogeneities, ShrinkageFactor
from .variables import NoisyProcesses
from .bffm_init import bffm_initializer
from ..results_old.mcmc_results import MCMCResults


class BFFModel:

	def __init__(
			self,
			stimulus_order: torch.Tensor,
			target_stimulus: torch.Tensor,
			stimulus_window: int,
			stimulus_to_stimulus_interval: int,
			latent_dim: int,
			sequences: Union[torch.Tensor, None] = None,
			n_stimulus: Tuple[int, int] = (6, 6),
			n_sequences: int = 15*19,
			n_channels: int = 15,
			independent_smgp: bool = False,
			nonnegative_smgp: bool = True,
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
			target_stimulus=target_stimulus,
			independent_smgp=independent_smgp,
			nonnegative_smgp=nonnegative_smgp
		)
		self._sampling_order = [
			"factor_processes",
			"mean_factor_processes",
			"smgp_factors",

			"loading_processes",
			"smgp_scaling",

			"loadings",
			"shrinkage_factor",
			"heterogeneities",

			"observation_variance"
		]

	def _prepare_model(
			self,
			sequences: Union[torch.Tensor, None],
			stimulus_order: torch.Tensor,
			target_stimulus: torch.Tensor,
			independent_smgp: bool = False,
			nonnegative_smgp: bool = True
	):
		parms = self.prior_parameters
		dims = self._dimensions

		# Observation variance
		observation_variance = ObservationVariance(
			n_channels=dims["n_channels"],
			prior_parameters=parms["observation_variance"]
		)

		# Loadings
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

		# Sequence data
		sequence_data = SequenceData(
			order=stimulus_order,
			target=target_stimulus
		)

		# Loading processes prior
		p = parms["kernel_gp_loading_processes"]
		tmat = scipy.linalg.toeplitz(p[0] ** (np.arange(dims["stimulus_window"])*p[2]))
		kernel_gp_loading_processes = Kernel.from_covariance_matrix(torch.Tensor(tmat) * p[1])
		p = parms["kernel_tgp_loading_processes"]
		tmat = scipy.linalg.toeplitz(p[0] ** (np.arange(dims["stimulus_window"])*p[2]))
		kernel_tgp_loading_processes = Kernel.from_covariance_matrix(torch.Tensor(tmat) * p[1])
		smgp_scaling = SMGP(
			dims["latent_dim"],
			kernel_gp_loading_processes,
			kernel_tgp_loading_processes,
			0.5,
			1.
		)
		if independent_smgp:
			smgp_scaling = IndependentSMGP(
				dims["latent_dim"],
				kernel_gp_loading_processes,
				kernel_tgp_loading_processes,
				0.5,
				1.
			)
		if nonnegative_smgp:
			smgp_scaling = NonnegativeSMGP(
				dims["latent_dim"],
				kernel_gp_loading_processes,
				kernel_tgp_loading_processes,
				0.5,
				1.
			)

		# Loading processes
		loading_processes = Superposition(
			smgp=smgp_scaling,
			sequence_data=sequence_data,
			stimulus_to_stimulus_interval=dims["stimulus_to_stimulus_interval"],
			window_length=dims["stimulus_window"]
		)
		loading_processes.name = "loading_processes"

		# Mean factor processes prior
		p = parms["kernel_gp_factor_processes"]
		tmat = scipy.linalg.toeplitz(p[0] ** (np.arange(dims["stimulus_window"])*p[2]))
		kernel_gp_factor_processes = Kernel.from_covariance_matrix(torch.Tensor(tmat) * p[1])
		p = parms["kernel_tgp_factor_processes"]
		tmat = scipy.linalg.toeplitz(p[0] ** (np.arange(dims["stimulus_window"])*p[2]))
		kernel_tgp_factor_processes = Kernel.from_covariance_matrix(torch.Tensor(tmat) * p[1])
		smgp_factors = SMGP(
			dims["latent_dim"],
			kernel_gp_factor_processes,
			kernel_tgp_factor_processes,
			0.5,
			0.
		)
		if independent_smgp:
			smgp_factors = IndependentSMGP(
				dims["latent_dim"],
				kernel_gp_factor_processes,
				kernel_tgp_factor_processes,
				0.5,
				0.
			)

		# Mean factor processes
		mean_factor_processes = Superposition(
			smgp=smgp_factors,
			sequence_data=sequence_data,
			stimulus_to_stimulus_interval=dims["stimulus_to_stimulus_interval"],
			window_length=dims["stimulus_window"]
		)
		mean_factor_processes.name = "factor_processes"

		# Factor processes
		p = parms["kernel_gp_factor"]
		tmat = scipy.linalg.toeplitz(p[0] ** (np.arange(dims["n_timepoints"])*p[2]))
		kernel_factor = Kernel.from_covariance_matrix(torch.Tensor(tmat) * p[1])
		factor_processes = NoisyProcesses(
			mean=mean_factor_processes,
			kernel=kernel_factor
		)

		# Observations
		observations = GaussianObservations(
			observation_variance=observation_variance,
			loadings=loadings,
			loading_processes=loading_processes,
			factor_processes=factor_processes,
			value=sequences
		)

		# Link backwards
		shrinkage_factor.add_children(loadings=loadings)
		heterogeneities.add_children(loadings=loadings)
		loadings.add_children(
			observations=observations,
			observation_variance=observation_variance
		)
		observation_variance.add_children(observations=observations)
		smgp_scaling.add_children(superposition=loading_processes)
		smgp_factors.add_children(superposition=mean_factor_processes)
		loading_processes.add_children(observations=observations)
		mean_factor_processes.add_children(child=factor_processes, observations=observations)
		factor_processes.add_children(observations=observations)

		# TODO: maybe this should be a plate?
		self.variables = {
			"observation_variance": observation_variance,
			"heterogeneities": heterogeneities,
			"shrinkage_factor": shrinkage_factor,
			"loadings": loadings,
			"sequence_data": sequence_data,
			"smgp_scaling": smgp_scaling,
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
			n_stimulus: Tuple[int, int] = (6, 6),
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

	@classmethod
	def load_dict(cls, data: dict):
		# data should be a dict with at least
		# - variables
		# - dimensions
		# - prior
		# For example, this could be the output of the result method
		obj = cls.generate_from_dimensions(**data["dimensions"], **data["prior"])
		obj.data = data["variables"]
		obj.clear_history() # we make sure we do not repeat the last value
		return obj

	@classmethod
	def load_file(cls, filename: str):
		with open(filename, "rb") as f:
			data = pickle.load(f)
		return cls.load_dict(data)

	def set(self, **kwargs):
		for k, v in kwargs.items():
			self.variables[k].data = v

	def _initialize_prior_parameters(self, **kwargs):
		prior_parameters = {
			"observation_variance": (3., 1.),
			"heterogeneities": 3.,
			"shrinkage_factor": (1., 10.),
			"kernel_gp_factor_processes": (0.99, 1., 1.),
			"kernel_tgp_factor_processes": (0.99, 0.5, 1.),
			"kernel_gp_loading_processes": (0.99, 1., 1.),
			"kernel_tgp_loading_processes": (0.99, 0.5, 1.),
			"kernel_gp_factor": (0.99, 0.1, 1.)
		}
		for k in prior_parameters.keys():
			if k in kwargs:
				prior_parameters[k] = kwargs[k]
		self.prior_parameters = prior_parameters

	def sample(self, sampling_order=None, random=True):
		if sampling_order is None:
			sampling_order = self._sampling_order
		if random:
			random_order = torch.randperm(len(sampling_order)).tolist()
			sampling_order = [sampling_order[i] for i in random_order]
		for var in sampling_order:
			try:
				if "." in var:
					v1, v2 = var.split(".")
					self.variables[v1][v2].sample()
				else:
					self.variables[var].sample()
			except Exception as e:
				print(f"Error sampling {var}: {e}")
		self.variables["observations"].store_log_density()

	def jitter_values(self, which=None, sd=0.01):
		if which is None:
			which = self._sampling_order
		for var in which:
			if "." in var:
				v1, v2 = var.split(".")
				self.variables[v1][v2].jitter(sd=sd)
			else:
				self.variables[var].jitter(sd=sd)

	def initialize_chain(self):
		# use WFA to find loadings and variance
		# the estimated factors will be used to initialize the processes below
		loadings, observation_variance, factors = bffm_initializer(
			target_stimulus=self.variables["sequence_data"].target.data,
			stimulus_order=self.variables["sequence_data"].order.data,
			sequences=self.variables["observations"].data,
			latent_dim=self._dimensions["latent_dim"],
			stimulus_window=self._dimensions["stimulus_window"],
			stimulus_to_stimulus_interval=self._dimensions["stimulus_to_stimulus_interval"],
		)
		# smooth out the factors
		smat = scipy.linalg.toeplitz(0.5 ** np.arange(factors.shape[2]))
		smat = torch.Tensor(smat)
		smat = smat / smat.sum(0)
		sfactors = factors @ smat
		# put values into variables
		self.variables["loadings"].data = loadings
		self.sample(["shrinkage_factor", "heterogeneities"])
		self.variables["observation_variance"].data = observation_variance
		self.variables["factor_processes"].data = sfactors.clone()
		# update smgp_factors: first set everything to constant, then update with the factors
		self.variables["smgp_factors"].nontarget_process.data.zero_()
		self.variables["smgp_factors"].target_process.data.zero_()
		self.variables["smgp_factors"].mixing_process.data.fill_(0.5)
		self.variables["smgp_scaling"].nontarget_process.data.fill_(1.)
		self.variables["smgp_scaling"].target_process.data.fill_(1.)
		self.variables["smgp_scaling"].mixing_process.data.fill_(0.5)
		self.sample(["mean_factor_processes"])
		self.sample(["smgp_factors"])
		self.sample(["mean_factor_processes"])
		self.sample(["factor_processes"])
		# compute the loadings processes by dividing and clipping above 0
		z = self.variables["factor_processes"].data
		lprocesses = 1. + (sfactors - z) / torch.where(z.abs() > 0.1, z, torch.ones_like(z))
		lprocesses = torch.clamp(lprocesses, min=0., max=5.)
		lprocesses = lprocesses @ smat
		self.variables["loading_processes"].data = lprocesses
		self.sample(["smgp_scaling"])
		self.sample(["loading_processes"])
		self.clear_history()

	@property
	def data(self):
		return {k: v.data for k, v in self.variables.items()}

	@data.setter
	def data(self, value: dict[str: torch.Tensor]):
		for k, v in value.items():
			self.variables[k].data = v

	def chain(self, start=0, end=None, thin=1):
		return {
			k: v.chain(start=start, end=end, thin=thin)
			for k, v in self.variables.items()
			if v._store
		}

	def results(self, start=0, end=None, thin=1):
		chain = self.chain(start, end, thin)  # preprocess first to reduce memory
		llk = self.variables["observations"].log_density_history
		if end is None:
			end = len(llk)
		llk = llk[start:end:thin]
		out = {
			"chain": chain,
			"log_likelihood": {"observations": llk},
			"prior": self.prior_parameters,
			"dimensions": self._dimensions,
			"thinned": thin,
			"warmed_up": start,
			"variables": self.data
		}
		return out

	def save(self, filename):
		results = self.results()
		with open(filename, "wb") as f:
			pickle.dump(results, f)

	def clear_history(self):
		for v in self.variables.values():
			v.clear_history()
		self.variables["observations"].log_density_history = []


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