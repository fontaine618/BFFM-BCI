import torch
import torch.linalg
from .variable import ObservedVariable
from .gaussian_process import GaussianProcess, TruncatedGaussianProcess01, NonnegativeGaussianProcess
from .plate import Plate
from ..utils import Kernel


class SMGP(Plate):
	r"""
	Split-merge Gaussian Process.

	We only store a0, a1 and zeta, not beta.
	Note that when zeta (mixing_process) is zero we get nontarget_process,
	and target_process when equal to 1.

	Children must be a Superposition so we can access its children to get values.
	"""

	_store = True

	def __init__(
			self,
			n_latent,
			kernel_gp: Kernel,
			kernel_tgp: Kernel,
			mean_tgp=0.5,
			mean_gp=0.,
	):
		self.nontarget_process: GaussianProcess = None
		self.target_process: GaussianProcess = None
		self.mixing_process: TruncatedGaussianProcess01 = None
		self.superposition = None
		super().__init__(
			nontarget_process=GaussianProcess(
				n_copies=n_latent, kernel=kernel_gp, mean=mean_gp
			),
			target_process=GaussianProcess(
				n_copies=n_latent, kernel=kernel_gp, mean=mean_gp
			),
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

	@property
	def data(self):
		proc = self.processes
		return {
			"nontarget_process": proc[0],
			"target_process": proc[1],
			"mixing_process": self.mixing_process.data
		}

	def chain(self, start=0, end=None, thin=1):
		zeta = self.mixing_process.chain(start=start, end=end, thin=thin)
		alpha0 = self.nontarget_process.chain(start=start, end=end, thin=thin)
		alpha1 = self.target_process.chain(start=start, end=end, thin=thin)
		beta1 = (1. - zeta) * alpha0 + zeta * alpha1
		return {
			"nontarget_process": alpha0,
			"target_process": beta1,
			"mixing_process": zeta
		}


class IndependentSMGP(SMGP):

	def __init__(
			self,
			n_latent,
			kernel_gp: Kernel,
			kernel_tgp: Kernel,
			mean_tgp=0.5,
			mean_gp=0.,
	):
		super(IndependentSMGP, self).__init__(n_latent, kernel_gp, kernel_tgp, mean_tgp, mean_gp)
		# replace the children with independent ones
		self.mixing_process = ObservedVariable(torch.ones_like(self.mixing_process.data))
		self.mixing_process.name = "mixing_process"
		self.variables["mixing_process"] = self.mixing_process


class NonnegativeSMGP(SMGP):

	def __init__(
			self,
			n_latent,
			kernel_gp: Kernel,
			kernel_tgp: Kernel,
			mean_tgp=0.5,
			mean_gp=1.,
	):
		super(NonnegativeSMGP, self).__init__(n_latent, kernel_gp, kernel_tgp, mean_tgp, mean_gp)
		# replace the children with nonnegative ones
		self.nontarget_process = NonnegativeGaussianProcess(n_copies=n_latent, kernel=kernel_gp, mean=mean_gp)
		self.target_process = NonnegativeGaussianProcess(n_copies=n_latent, kernel=kernel_gp, mean=mean_gp)
		self.nontarget_process.name = "nontarget_process"
		self.target_process.name = "target_process"
		self.variables["nontarget_process"] = self.nontarget_process
		self.variables["target_process"] = self.target_process