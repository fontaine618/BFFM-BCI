from .variable import Variable
from .observations import GaussianObservations
from models.bffmbci.utils.inverse_gamma import InverseGamma


class ObservationVariance(Variable):
	r"""
	Observation variance with Inverse Gamma prior.

	This variable does not have parents, only a prior (InvGamma)
	parameterized by (a, b). The children are the observations,
	which is a Gaussian observation model. The channels are independent
	given the mean process so only the diagonal is stored.

	We expect the observation to have dimensions
	[n_sequences, n_channels, n_timepoints]

	observations needs to implement residuals

	Initialization is sampling from prior.
	"""

	_dim_names = ["n_channels"]

	def __init__(self, n_channels=15, prior_parameters=(1, 0.1)):
		self._a = prior_parameters[0]
		self._b = prior_parameters[1]
		self.observations: GaussianObservations = None
		super().__init__(dim=(n_channels, ), store=True)

	def sample(self, store=True):
		residuals = self.observations.residuals
		dim = residuals.shape
		a = self._a + 0.5 * dim[0] * dim[2]
		b = self._b + 0.5 * residuals.pow(2).sum((0, 2))
		dist = InverseGamma(a, b)
		self._set_value(dist.sample(), store=store)

	def generate(self):
		dist = InverseGamma(self._a, self._b)
		self._set_value(dist.sample(self.shape))