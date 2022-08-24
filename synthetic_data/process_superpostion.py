import torch


def process_superposition(
		nontarget_process: torch.Tensor,
		target_process: torch.Tensor,
		stimuli_order: torch.Tensor,
		target_indicator: torch.Tensor,
		window_length: int = 200,
		stimulus_to_stimulus_interval: int = 40
) -> torch.Tensor:
	"""

	:param nontarget_process: (K, T)
	:param target_process: (K, T)
	:param stimuli_order: (N, J)
	:param target_indicator: (N, J)
	:param window_length: int
	:param stimulus_to_stimulus_interval: int
	:return:
	"""
	N, J = stimuli_order.shape
	K, T = nontarget_process.shape
	d = stimulus_to_stimulus_interval
	out = torch.zeros((N, K, T))
	for n in range(N):
		wn = stimuli_order[n, :]
		yn = target_indicator[n, :]
		for j in range(J):
			wnj = wn[j]
			ynj = yn[j]
			p_inj = target_process * ynj + nontarget_process * (1-ynj)
			time = torch.arange(0, T)
			shift = (time - wnj * d).long()
			which = (shift >= 0) * (shift < window_length)
			out[n, :, time[which]] = p_inj[:, shift[which]]
	return out
