import torch
from .variable import Variable
from .sequence_data import SequenceData
from .smpg import SMGP
from torch.autograd.functional import jacobian


class Superposition(Variable):
	r"""
	Takes a SMGP prior and sequence data to generate the superposition of signals.

	Children should be either observation or a noisy version of this.
	and should implement message_to_parent.

	Subclasses will implement methods helping parent's sample.
	"""

	_stochastic = False
	_dim_names = ["n_sequences", "n_processes", "n_timepoints"]

	def __init__(
			self,
			smgp: SMGP,
			sequence_data: SequenceData,
			stimulus_to_stimulus_interval: int,
			window_length: int
	):
		self.smgp = smgp
		self.sequence_data = sequence_data
		self._stimulus_to_stimulus_interval = stimulus_to_stimulus_interval
		self._window_length = window_length
		self.child: Variable = None
		n_sequences, n_stimuli = sequence_data.order.shape
		n_latent, _ = smgp.nontarget_process.shape
		n_timepoints = (n_stimuli - 1) * stimulus_to_stimulus_interval + window_length
		dim = n_sequences, n_latent, n_timepoints
		super().__init__(dim, store=False, init=None)
		self.parents = {
			"smpg": smgp,
			"sequence_data": sequence_data
		}
		self.name = None

	def superposition(self, nontarget_process=None, target_process=None, mixing_process=None):
		N, J = self.sequence_data.order.shape
		K, W = self.smgp.nontarget_process.shape
		d = self._stimulus_to_stimulus_interval
		T = (J - 1) * d + W
		value = torch.zeros((N, K, T))
		if nontarget_process is None:
			nontarget_process = self.smgp.nontarget_process.data
		if target_process is None:
			target_process = self.smgp.target_process.data
		if mixing_process is None:
			mixing_process = self.smgp.mixing_process.data
		time = torch.arange(0, T)
		# for n in range(N):
		# 	wn = self.sequence_data.order.data[n, :]
		# 	yn = self.sequence_data.target.data[n, :]
		# 	for j in range(J):
		# 		ynj = yn[j]
		# 		p_inj = (1 - ynj * mixing_process) * nontarget_process + \
		# 		        ynj * mixing_process * target_process
		# 		wnj = wn[j]
		# 		shift = (time - wnj * d).long()
		# 		which = (shift >= 0) * (shift < W)
		# 		value[n, :, time[which]] += p_inj[:, shift[which]]

		w = self.sequence_data.order.data
		y = self.sequence_data.target.data
		nontarget_process = nontarget_process.unsqueeze(-1).unsqueeze(0)
		target_process = target_process.unsqueeze(-1).unsqueeze(0)
		mixing_process = mixing_process.unsqueeze(-1).unsqueeze(0)
		yy = y.unsqueeze(1).unsqueeze(1)
		p_in = ((1 - yy * mixing_process) * nontarget_process + \
		        yy * mixing_process * target_process).movedim(3, 2)
		shift_n = (time - w.unsqueeze(-1) * d).long() \
			.unsqueeze(1).expand(p_in.shape[0], p_in.shape[1], -1, -1)
		which_n = (shift_n >= 0) * (shift_n < W)
		shift_n = torch.where(which_n, shift_n, W)
		p_in = torch.cat([p_in, torch.zeros(N, K, J, 1)], 3)
		value = torch.gather(p_in, 3, shift_n).sum(2)
		return value

	def generate(self):
		self._set_value(self.superposition(), store=False)

	# def parameter_update_for_sampling(self, which: str, k: int):
	# 	# so we can have the same class for both
	# 	process_k = torch.nn.Parameter(
	# 		torch.zeros_like(self.smgp.__getattribute__(which).data[k, :]),
	# 		requires_grad=True
	# 	)
	#
	# 	def f(x):
	# 		process = self.smgp.__getattribute__(which).data
	# 		process[k, :] = x
	# 		self.smgp.__getattribute__(which)._set_value(process)
	# 		self.generate()
	# 		return self.child.mean
	#
	# 	# this could more efficient if done incrementally so we dont have to store the whole thing
	# 	m0 = f(process_k)
	# 	J = jacobian(f, process_k)
	# 	mt = self.child.data - m0
	# 	sig_inv = self.child.observation_variance.data.pow(-1)
	# 	prec = torch.einsum("netu, e, netv -> uv", J, sig_inv, J)
	# 	mtp = torch.einsum("netu, e, net -> u", J, sig_inv, mt)
	# 	return prec, mtp

	def sample(self, store=False):
		# ensure we satisfy equalities, no need to save
		self.generate()
