import torch


def extract_subsequences(
		sequences: torch.Tensor,
		stimuli_order: torch.Tensor,
		target_indicator: torch.Tensor,
		window_length: int = 200,
		stimulus_to_stimulus_interval: int = 40
) -> tuple[torch.Tensor, torch.Tensor]:
	"""

	:param sequences: (N, E, T)
	:param stimuli_order: (N, J)
	:param target_indicator: (N, J)
	:param window_length:
	:param stimulus_to_stimulus_interval:
	:return:
		- (NJ, E, T)
		- (NJ, )
	"""
	N, E, T = sequences.shape
	d = stimulus_to_stimulus_interval
	t = window_length
	J = stimuli_order.shape[1]
	out = torch.zeros((N*J, E, t)) * torch.nan
	y = torch.zeros((N*J,))
	for n in range(N):
		for j in range(J):
			time = torch.arange(j*d, min(j*d+t, T))
			out[n*J+j, :, 0:len(time)] = sequences[n, :, time]
			y[n*J+j] = target_indicator[n, j]
	return out, y