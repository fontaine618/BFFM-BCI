import torch
import torch.linalg
from .variable import ObservedVariable, ConstantVariable
from .gaussian_process import GaussianProcess, TruncatedGaussianProcess01, NonnegativeGaussianProcess
from .plate import Plate
from ..utils import Kernel


class SMGP(Plate):
	r"""
	Split-merge Gaussian Process.

	We only store a0, a1 and zeta, not beta.
	Note that when zeta (mixing_process) is zero we get nontarget_process,
	and target_process when equal to 1.

	Children must be a Superposition, so we can access its children to get values.
	"""

	_store = True
	_dim_names = ["n_processes", "n_timepoints"]

	def __init__(
			self,
			n_latent,
			kernel_gp: Kernel,
			kernel_tgp: Kernel,
			mean_tgp=0.5,
			mean_gp=0.,
			fixed_components: list[int] | None = None
	):
		self.nontarget_process: GaussianProcess | None = None
		self.target_process: GaussianProcess | None = None
		self.mixing_process: TruncatedGaussianProcess01 | None = None
		self.superposition = None
		self._fixed_components = fixed_components if fixed_components is not None else []
		super().__init__(
			nontarget_process=GaussianProcess(
				n_copies=n_latent, kernel=kernel_gp, mean=mean_gp, fixed_components=fixed_components
			),
			target_process=GaussianProcess(
				n_copies=n_latent, kernel=kernel_gp, mean=mean_gp, fixed_components=fixed_components
			),
			mixing_process=TruncatedGaussianProcess01(
				n_copies=n_latent, kernel=kernel_tgp, mean=mean_tgp, fixed_components=fixed_components
			)
		)
		self.nontarget_process.name = "nontarget_process"
		self.target_process.name = "target_process"
		self.mixing_process.name = "mixing_process"
		self._dim = n_latent, kernel_gp.shape[0]

	@property
	def processes(self):
		zeta = self.mixing_process.data
		alpha0 = self.nontarget_process.data
		alpha1 = self.target_process.data
		beta1 = (1. - zeta) * alpha0 + zeta * alpha1
		return alpha0, beta1


class ConstantSMGP(SMGP):

	def __init__(
			self,
			n_latent,
			kernel_gp: Kernel,
			kernel_tgp: Kernel,
			mean_tgp=0.5,
			mean_gp=0.,
	):
		super(ConstantSMGP, self).__init__(n_latent, kernel_gp, kernel_tgp, mean_tgp, mean_gp)
		self.mixing_process = ConstantVariable(torch.zeros_like(self.mixing_process.data))
		self.nontarget_process = ConstantVariable(torch.zeros_like(self.mixing_process.data))
		self.target_process = ConstantVariable(torch.zeros_like(self.mixing_process.data))
		self.mixing_process.name = "mixing_process"
		self.nontarget_process.name = "nontarget_process"
		self.target_process.name = "target_process"
		self.variables["mixing_process"] = self.mixing_process
		self.variables["nontarget_process"] = self.nontarget_process
		self.variables["target_process"] = self.target_process


class SingleSMGP(SMGP):

	def __init__(
			self,
			n_latent,
			kernel_gp: Kernel,
			kernel_tgp: Kernel,
			mean_tgp=0.5,
			mean_gp=0.,
			fixed_components: list[int] | None = None
	):
		super(SingleSMGP, self).__init__(n_latent, kernel_gp, kernel_tgp, mean_tgp, mean_gp, fixed_components)
		self.mixing_process = ConstantVariable(torch.zeros_like(self.mixing_process.data))
		self.target_process = ConstantVariable(torch.zeros_like(self.mixing_process.data))
		self.mixing_process.name = "mixing_process"
		self.target_process.name = "target_process"
		self.variables["mixing_process"] = self.mixing_process
		self.variables["target_process"] = self.target_process


# class IndependentSMGP(SMGP):
#
# 	def __init__(
# 			self,
# 			n_latent,
# 			kernel_gp: Kernel,
# 			kernel_tgp: Kernel,
# 			mean_tgp=0.5,
# 			mean_gp=0.,
# 	):
# 		super(IndependentSMGP, self).__init__(n_latent, kernel_gp, kernel_tgp, mean_tgp, mean_gp)
# 		# replace the children with independent ones
# 		self.mixing_process = ObservedVariable(torch.ones_like(self.mixing_process.data))
# 		self.mixing_process.name = "mixing_process"
# 		self.variables["mixing_process"] = self.mixing_process
#
#
# class NonnegativeSMGP(SMGP):
#
# 	def __init__(
# 			self,
# 			n_latent,
# 			kernel_gp: Kernel,
# 			kernel_tgp: Kernel,
# 			mean_tgp=0.5,
# 			mean_gp=1.,
# 	):
# 		super(NonnegativeSMGP, self).__init__(n_latent, kernel_gp, kernel_tgp, mean_tgp, mean_gp)
# 		# replace the children with nonnegative ones
# 		self.nontarget_process = NonnegativeGaussianProcess(n_copies=n_latent, kernel=kernel_gp, mean=mean_gp)
# 		self.target_process = NonnegativeGaussianProcess(n_copies=n_latent, kernel=kernel_gp, mean=mean_gp)
# 		self.nontarget_process.name = "nontarget_process"
# 		self.target_process.name = "target_process"
# 		self.variables["nontarget_process"] = self.nontarget_process
# 		self.variables["target_process"] = self.target_process
