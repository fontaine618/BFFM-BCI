from typing import Tuple, Union
import torch
import numpy as np
import scipy.linalg
import torch.nn.functional as F

from src.models.bffmbci.utils import Kernel
from src.models.bffmbci.variables import SequenceData, SMGP, Superposition
from src.initialization.wfa import WFA


def bffm_initializer(
		target_stimulus: torch.Tensor,
		stimulus_order: torch.Tensor,
		sequences: torch.Tensor,
		latent_dim: int,
		stimulus_window: int,
		stimulus_to_stimulus_interval: int,
	):
		N, E, T = sequences.shape
		w = stimulus_window
		d = stimulus_to_stimulus_interval
		K = latent_dim
		# we create a dummy SMGP object and put the variance
		dummy_kernel = Kernel.from_covariance_matrix(torch.eye(w))
		smgp = SMGP(K, dummy_kernel, dummy_kernel)
		# target and nontarget has variance 1, mixing has variance 0.733
		# hence nontarget has variance 1 and target has variance 1.733
		smgp.nontarget_process._value = torch.ones(K, w)
		smgp.target_process._value = torch.ones(K, w) * 1.733
		smgp.mixing_process._value = torch.ones(K, w)
		sequence_data = SequenceData(
			order=stimulus_order,
			target=target_stimulus
		)
		superposition = Superposition(
			smgp=smgp,
			sequence_data=sequence_data,
			stimulus_to_stimulus_interval=d,
			window_length=w
		)
		# get variance at each timepoint, we just need one slice
		variance = superposition.compute_superposition()[:, 0, :]
		# to long format
		W = variance.reshape(-1)  # inverse is .reshape(N, T)
		# X = sequences.reshape(-1, E)  # inverse is .reshape(N, T, E)
		X = torch.vstack([sequences[i, :, :].T for i in range(sequences.shape[0])])
		fa = WFA(K)
		fa.fit(X, W)
		loadings = fa._loadings
		observation_variance = fa._observation_variance
		factors = fa._m1
		# we need to reshape the factors
		T = sequences.shape[-1]
		factors = torch.stack(
			[factors[(T*i):(T*(i+1)), :] for i in range(N)],
			dim=0
		).permute(0, 2, 1)
		return loadings, observation_variance, factors