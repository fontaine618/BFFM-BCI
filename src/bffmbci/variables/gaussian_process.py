import warnings
import scipy
import math
import torch
import torch.linalg
from .variable import Variable, ObservedVariable
from ..utils import Kernel, TruncatedMultivariateGaussian
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.autograd.functional import jacobian


class GaussianProcess(Variable):
	r"""
	Abstract class for n Gaussian processes all with the same kernel matrix.

	We assume the kernel to be fixed and of class Kernel.
	We allow mean to be either a fixed integer or a Variable.
	"""

	_dim_names = ["n_processes", "n_timepoints"]

	def __init__(self, n_copies, kernel: Kernel, mean=0.):
		dim = n_copies, kernel.shape[0]
		self.kernel = kernel
		if isinstance(mean, float):
			mean = torch.full(dim, mean)
			self.mean = ObservedVariable(mean)
		elif isinstance(mean, Variable):
			self.mean = mean
		super().__init__(dim=dim, store=True, init=None)
		self.parents = {"mean": self.mean}
		self.name = None
		self.superposition = None
		# self.sample = self.elliptical_slice_sample_prior
		# self.sample = self.direct_sample
		self.n_proposals = 0
		self.n_accepts = 0
		self.n_evals = 0

		self.sample = self.rwmh_sample
		self._rwmh = {
			"n_proposals": [0 for _ in range(self._dim[0])],
			"n_accepts": [0 for _ in range(self._dim[0])],
			"log_scale": [-5. for _ in range(self._dim[0])],
			"target_rate": [0.234 for _ in range(self._dim[0])],
			"step_size": [1. for _ in range(self._dim[0])],
			"power": [0.5 for _ in range(self._dim[0])],
		}

		self._initialize_rwmh()

		# self.sample = self.elliptical_slice_sampler_posterior
		self._ess = {
			"n_proposals": [0 for _ in range(self._dim[0])],
			"n_accepts": [0 for _ in range(self._dim[0])],
			"n_evals": [0 for _ in range(self._dim[0])],
		}
		self._prior_dist = MultivariateNormal(
			loc=self.mean.data,
			covariance_matrix=self.kernel.cov
		)

		# self.sample = self.mala_sample
		self._mala = {
			"n_proposals": [0 for _ in range(self._dim[0])],
			"n_accepts": [0 for _ in range(self._dim[0])],
			"step_size": [0.001 for _ in range(self._dim[0])],
		}


	def generate(self):
		dist = MultivariateNormal(loc=torch.zeros(self._dim[1]), scale_tril=self.kernel.chol)
		self._set_value(dist.sample((self._dim[0], )) + self.mean.data)

	def _parameters_from_child(self, k, value):
		r"""
		Should return child precision and means times precision to add to prior.
		"""
		L, fmk = self._get_linear_transform(k, value)

		# compute updates to precision and mtp
		Kinv = self.superposition.observations.time_kernel(k).inv
		x = self.superposition.observations.data
		xt = x - fmk
		mtp1 = torch.einsum("netu, tv, nev -> u", L, Kinv, xt)
		p1 = torch.einsum("netu, tv, nevw -> uw", L, Kinv, L)
		return p1, mtp1

	def _get_linear_transform(self, k, value):
		z = value.clone().detach()
		zk = torch.nn.Parameter(z[k, :], requires_grad=True)

		def f(zk):
			z[k, :] = zk
			s = self.superposition.compute_superposition(**{self.name: z})
			sname = self.superposition.name
			out = self.superposition.observations.mean(**{sname: s})
			return out

		L = jacobian(f, zk, strategy="forward-mode", vectorize=True)
		fmk = f(0.)
		return L.detach(), fmk.detach()

	def direct_sample(self, store=True):
		value = self._value.clone().detach()
		for k in range(self._dim[0]):
			c, m = self._get_posterior(k, value)
			dist = self._dist(mean=m, covariance=c)
			value[k, :] = self._sample_k(dist, self.data[k, :])
		self._set_value(value, store=store)

	def _dist(self, mean, covariance):
		return MultivariateNormal(loc=mean, covariance_matrix=covariance)

	def _sample_k(self, dist, value):
		return dist.sample()

	# def _get_log_prob(self, k, value):
	# 	llk = self.get_log_likelihood(value)
	# 	log_prior = self._prior_dist.log_prob(value)[k]
	# 	return llk + log_prior
	#
	# def _get_log_prob_and_grad(self, k, value):
	# 	vk = torch.nn.Parameter(value[k, :], requires_grad=True)
	# 	value[k, :] = vk
	# 	logpi = self._get_log_prob(k, value)
	# 	logpi.backward(retain_graph=True)
	# 	grad = vk.grad
	# 	return logpi.detach(), grad.detach()

	def _get_log_prob_and_grad(self, k, value):
		z = value.clone().detach()
		vk = torch.nn.Parameter(z[k, :], requires_grad=True)

		def f(zk):
			z[k, :] = zk
			llk = self.get_log_likelihood(value)
			log_prior = self._prior_dist.log_prob(value)[k]
			return llk + log_prior

		L = jacobian(f, vk, strategy="forward-mode", vectorize=True)
		fmk = f(vk)
		return fmk.detach(), L.detach()

	def mala_sample(self, store=True):
		value = self._value.clone().detach()
		for k in range(self._dim[0]):
			logpi, grad = self._get_log_prob_and_grad(k, value)
			step_size = self._mala["step_size"][k]
			vk = value[k, :].clone().detach()
			step = step_size * grad
			noise = torch.randn_like(vk) * (2 * step_size) ** 0.5
			proposal = vk + step + noise
			value[k, :] = proposal.detach()
			if not self.check_constraints(value):
				acc_rate = 0.
			else:
				logpi_proposal, grad_proposal = self._get_log_prob_and_grad(k, value)
				step_proposal = step_size * grad_proposal
				num = logpi_proposal - (vk - proposal - step_proposal).pow(2.).sum() / (4*step_size)
				den = logpi - (proposal - vk - step).pow(2.).sum() / (4*step_size)
				acc_rate = min(1., torch.exp(num - den))
			if torch.rand(1) < acc_rate:
				acc = 1
			else:
				acc = 0
				value[k, :] = vk
			self._mala["n_accepts"][k] += acc
			self._mala["n_proposals"][k] += 1
			self._mala["step_size"][k] *= (1. + 0.02 * (acc - 0.574))
			# print(f"[{self.id}.{k}] Acceptance rate: {acc_rate:.3f} ({acc})")
		self._set_value(value.detach(), store=store)

	def elliptical_slice_sample_prior(self, store=True):
		# TODO I think this is a bit wrong on how I deal with mean
		value = self._value.clone().detach()
		ogvalue = self._value.clone().detach()
		chol = self.kernel.chol
		# llk_proposed = self.get_log_likelihood(value)
		for k in range(self._dim[0]):
			m0 = self.mean.data[k, :]
			# first iteration, this will take current value
			# subsequent iteration, this will take the llk at the accepted value
			# llk_current = llk_proposed
			llk_current = self.get_log_likelihood(value)
			vk = value[k, :].clone().detach()
			nu = m0 + torch.randn(self._dim[1]) @ chol.T
			# nu = torch.distributions.multivariate_normal.MultivariateNormal(
			# 	loc=m0,
			# 	scale_tril=chol
			# ).sample()
			u = torch.rand(1)
			theta = torch.rand(1) * 2 * torch.pi
			theta_min = theta - 2 * torch.pi
			theta_max = theta
			n_proposals = 0
			n_evals = 0
			while True:
				if n_proposals > 0:  # skip first step
					if theta > 0:
						theta_max = theta
					else:
						theta_min = theta
					theta = torch.rand(1) * (theta_max - theta_min) + theta_min
				mk = torch.cos(theta) * vk + torch.sin(theta) * nu
				# print(theta, mk[0:3])
				n_proposals += 1
				if n_proposals > 30:
					# at this point, since /2 every iteration, we should be back to the original value
					# then we stop and revert to original value
					value[k, :] = ogvalue[k, :]
					# warnings.warn(f"[{self.id}] no proposal accepted after {n_proposals} proposals")
					break
				value[k, :] = mk
				if not self.check_constraints(value):
					continue
				llk_proposed = self.get_log_likelihood(value)
				n_evals += 1
				if llk_proposed > llk_current + torch.log(u).item():
					# print(f"[{self.id}] accepted after {n_proposals} proposal; "
					# 	  f"llk previous: {llk_current:.3f}, "
					# 	  f"llk accepted: {llk_proposed:.3f}")
					break  # accept current proposal
			self.n_evals += n_evals
			self.n_proposals += n_proposals
			self.n_accepts += 1
		# print(f"{self.id} ESS running acceptance rate: "
		# 	  f"{self.n_accepts / self.n_proposals:.3f}")
		self._set_value(value, store=store)

	def _get_posterior_natural(self, k, value):
		p0 = self.kernel.inv
		mtp0 = p0 @ self.mean.data[k, :]
		p1, mtp1 = self._parameters_from_child(k, value)
		prec = p0 + p1
		mtp = mtp0 + mtp1
		return prec, mtp

	def _get_posterior(self, k, value):
		prec, mtp = self._get_posterior_natural(k, value)
		c = torch.inverse(prec)
		m = c @ mtp
		return c, m

	def _ess_log_prob(self, proposal_dist, value, llk, mean, k):
		prior_log_prob = self._prior_dist.log_prob(value)[k] # TODO this works, but could be improved
		proposal_log_prob = proposal_dist.log_prob(value - mean)
		return prior_log_prob + llk - proposal_log_prob

	def elliptical_slice_sampler_posterior(self, store=True):
		value = self._value.clone().detach()
		ogvalue = self._value.clone().detach()
		for k in range(self._dim[0]):
			# get posterior (will be the proposal distribution)
			c, m = self._get_posterior(k, value)
			chol = torch.linalg.cholesky(c)
			# get proposal (note we don't use _dist, since we want normal proposal, not TGN)
			proposal_dist = MultivariateNormal(loc=torch.zeros_like(m), covariance_matrix=c)
			# get comparison value
			mk = self.mean.data[k, :]
			vk = value[k, :].clone().detach()
			llk_current = self.get_log_likelihood(value)
			log_prob = self._ess_log_prob(proposal_dist, vk, llk_current, mk, k)
			logy = torch.rand(1).log().item() + log_prob
			# get initial proposal
			nu = torch.randn(self._dim[1]) @ chol.T
			theta = torch.rand(1) * 2 * torch.pi
			theta_min = theta - 2 * torch.pi
			theta_max = theta
			n_proposals = 0
			n_evals = 0
			fk = vk - mk
			while True:
				if n_proposals > 0:  # skip first step
					if theta > 0:
						theta_max = theta
					else:
						theta_min = theta
					theta = torch.rand(1) * (theta_max - theta_min) + theta_min
				fkp = torch.cos(theta) * fk + torch.sin(theta) * nu
				n_proposals += 1
				if n_proposals > 30:
					value[k, :] = ogvalue[k, :]
					break
				vk = fkp + mk
				value[k, :] = vk
				if not self.check_constraints(value):
					continue
				llk_proposed = self.get_log_likelihood(value)
				log_prob = self._ess_log_prob(proposal_dist, vk, llk_proposed, mk, k)
				n_evals += 1
				if log_prob > logy:
					break  # accept current proposal
			# print(f"[{self.id}.{k}] accepted after {n_proposals} proposal ({n_evals} llk evaluations) ")
			self._ess["n_evals"][k] += n_evals
			self._ess["n_proposals"][k] += n_proposals
			self._ess["n_accepts"][k] += 1
		self._set_value(value, store=store)

	def _initialize_rwmh(self):
		# Equation (7) in Garthwaite et al. (2016)
		m = self.kernel.shape[0]
		for k in range(self._dim[0]):
			pstar = self._rwmh["target_rate"][k]
			alpha = - scipy.stats.norm.ppf(pstar / 2)
			cstar = (2 * torch.pi) ** (0.5) * math.exp(alpha * alpha / 2) / (2 * alpha)
			cstar *= (1 - 1/m)
			cstar += 1 / (m*pstar*(1-pstar))
			self._rwmh["step_size"][k] = cstar

	def rwmh_sample(self, store=True):
		# our proposal will be the prior scale by a parameter
		# that we adaptively optimize using a Robbins-Monro update
		# proposed by Garthwaite et al. (2016)
		value = self._value.clone().detach()
		ogvalue = self._value.clone().detach()
		chol = self.kernel.chol
		for k in range(self._dim[0]):
			vk = value[k, :].clone().detach()
			sk = self._rwmh["log_scale"][k]
			proposal = vk + math.exp(sk / 2) * torch.randn(self._dim[1]) @ chol.T
			llk_current = self.get_log_likelihood(value)
			value[k, :] = proposal
			if not self.check_constraints(value):
				acc_prob = -float("inf")
				acc = 0
				value[k, :] = vk
			else:
				llk_proposed = self.get_log_likelihood(value)
				acc_prob = min(0, llk_proposed - llk_current)
				u = torch.rand(1).log().item()
				if u < acc_prob:
					acc = 1
				else:
					value[k, :] = vk # revert to previous value
					acc = 0
			# print(k, acc_prob, acc)
			self._rwmh["n_accepts"][k] += acc
			self._rwmh["n_proposals"][k] += 1
			self._rwmh_update_k(k, acc)
		self._set_value(value, store=store)

	def _rwmh_update_k(self, k, acc):
		step_size = self._rwmh["step_size"][k] / max(self._rwmh["n_proposals"][k] ** self._rwmh["power"][k], 1)
		self._rwmh["log_scale"][k] += (acc - self._rwmh["target_rate"][k]) * step_size

	def check_constraints(self, value):
		return True

	def get_log_likelihood(self, value):
		self._set_value(value, store=False)
		self.superposition.generate()
		return self.superposition.observations.log_density


class TruncatedGaussianProcess01(GaussianProcess):

	def __init__(self, n_copies, kernel: Kernel, mean=0.5):
		super().__init__(n_copies, kernel, mean)
		# self.sample = self.direct_sample

	def _dist(self, mean, covariance):
		return TruncatedMultivariateGaussian(mean=mean, covariance=covariance)

	def generate(self):
		value = self.mean.data.clone().detach()
		for k in range(value.shape[0]):
			dist = TruncatedMultivariateGaussian(mean=self.mean.data[k, :], covariance=self.kernel.cov)
			value[k, :] = dist.sample(self.mean.data[k, :])
		self._set_value(value)

	def _sample_k(self, dist, value):
		return dist.sample(value)

	def jitter(self, sd: float = 0.01):
		noise = torch.randn(self.shape) * sd
		self._set_value((self._value * (1 + noise)).clamp(min=0., max=1.))

	def check_constraints(self, value):
		return (value >= 0.).all() and (value <= 1.).all()


class NonnegativeGaussianProcess(GaussianProcess):

	def __init__(self, n_copies, kernel: Kernel, mean=1.):
		super().__init__(n_copies, kernel, mean)
		# self.sample = self.direct_sample

	def _dist(self, mean, covariance):
		return TruncatedMultivariateGaussian(mean=mean, covariance=covariance, lower=0., upper=100.)

	def generate(self):
		value = self.mean.data.clone().detach()
		for k in range(value.shape[0]):
			dist = TruncatedMultivariateGaussian(mean=self.mean.data[k, :], covariance=self.kernel.cov,
			                                     lower=0., upper=100.)
			value[k, :] = dist.sample(self.mean.data[k, :])
		self._set_value(value)

	def _sample_k(self, dist, value):
		return dist.sample(value)

	def jitter(self, sd: float = 0.01):
		noise = torch.randn(self.shape) * sd
		self._set_value((self._value * (1 + noise)).clamp(min=0.))

	def check_constraints(self, value):
		return (value >= 0.).all()