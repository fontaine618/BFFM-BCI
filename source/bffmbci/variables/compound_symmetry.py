import torch
from .variable import Variable, ObservedVariable
from ..utils.inverse_gamma import InverseGamma
from torch.distributions.normal import Normal
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.distributions.gamma import Gamma
from .smpg import SMGP

"""
For compatibility with the current infrastructure, we reformulate
the Ma (2022) model in a different way.

The loading matrix is [I, 1] of dimension E x (E+1) so that the
first E components capture the mean structure and the last component capture the 
compound symmetry structure. Correspondingly, 
the scaling process for the first E components will be identically equal to 1
and an (almost) constant value for the last component, which will capture the correlation.
The mean factor processes for the first E components corresponds directly to the
mean; the last one is fixed to 0.

Of note, the covariance contribution of the first E component is 
exactly I so the observation variance will be shifted by one. 
Fortunately, the posterior values are far away from 0, so this is not an issue.
"""


class CompoundSymmetryLoadings(Variable):
	r"""
	Loadings
	"""

	_dim_names = ["n_channels", "latent_dim"]

	def __init__(self, n_channels: int):
		super().__init__((n_channels, n_channels+1), store=True, init=None)

	def generate(self):
		self._set_value(torch.hstack([
			torch.eye(self.shape[0]),
			torch.ones(self.shape[0], 1)
		]))

	def sample(self, store=True):
		self._set_value(torch.hstack([
			torch.eye(self.shape[0]),
			torch.ones(self.shape[0], 1)
		]), store=store)