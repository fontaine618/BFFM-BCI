import torch


def gaussian_observation_model(
		mean_process: torch.Tensor,
		covariance_cholesky_process: torch.Tensor,
		error_kernel_cholesly: torch.Tensor
) -> torch.Tensor:
	"""

	:param mean_process: (N, E, T)
	:param covariance_cholesky_process: (N, E, E, T)
	:param error_kernel_cholesly: (T, T)
	:return:
	"""
	N, E, T = mean_process.shape
	out = torch.zeros_like(mean_process)
	z = torch.randn_like(out)
	z = torch.einsum("tu, net->neu", error_kernel_cholesly, z)
	e = torch.einsum("neft, net->net", covariance_cholesky_process, z)
	return mean_process + e