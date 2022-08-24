import torch
import torch.linalg
from .variable import Variable, ObservedVariable
# from .superposition import Superposition
from .gaussian_process import GaussianProcess, TruncatedGaussianProcess01
from .plate import Plate
from ..utils import Kernel
from torch.distributions.multivariate_normal import MultivariateNormal


class SMGP(Plate):
	r"""
	Split-merge Gaussian Process.

	We only store a0, a1 and zeta, not beta.

	Children must be a Superposition so we can access its children to get values.
	"""

	def __init__(
			self,
			n_latent,
			kernel_gp: Kernel,
			kernel_tgp: Kernel,
			mean_tgp=0.5
	):
		self.nontarget_process: GaussianProcess = None
		self.target_process: GaussianProcess = None
		self.mixing_process: TruncatedGaussianProcess01 = None
		self.superposition = None
		super().__init__(
			nontarget_process=GaussianProcess(n_copies=n_latent, kernel=kernel_gp, mean=0.),
			target_process=GaussianProcess(n_copies=n_latent, kernel=kernel_gp, mean=0.),
			mixing_process=TruncatedGaussianProcess01(
				n_copies=n_latent, kernel=kernel_tgp, mean=mean_tgp
			)
		)
		self.nontarget_process.name = "nontarget_process"
		self.target_process.name = "target_process"
		self.mixing_process.name = "mixing_process"

	@property
	def processes(self):
		zeta = self.mixing_process.data
		alpha0 = self.nontarget_process.data
		alpha1 = self.target_process.data
		beta1 = (1. - zeta) * alpha0 + zeta * alpha1
		return alpha0, beta1

	# def sample(self, store=True):
	# 	self.nontarget_process.sample(store)
	# 	self.target_process.sample(store)
	# 	self.mixing_process.sample(store)
	# 	# for which in ["nontarget_process", "target_process", "missing_process"]:
	# 	# 	process = self.__getattribute__(which)
	# 	# 	for k in range(self.nontarget_process.shape[0]):
	# 	# 		prec, mtp = self.superposition.parameter_update_for_sampling(which, k)
	# 	# 		process.sample_from_parameter_update(k, prec, mtp)
	# 	# 	process.store_new_value(store)