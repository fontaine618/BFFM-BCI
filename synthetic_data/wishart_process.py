import torch


def wishart_process_dictionary(
		latent_process: torch.Tensor,
		loadings: torch.Tensor,
		base_covariance: torch.Tensor,
		return_cholesky_lower: bool = True
) -> torch.Tensor:
	"""

	:param latent_process: (N, K, T)
	:param loadings: (E, K)
	:param base_covariance: (E, E)
	:return:
	"""
	N, K, T = latent_process.shape
	loadings.shape[0]
	out = base_covariance.unsqueeze(0).repeat(N, 1, 1).unsqueeze(3).repeat(1, 1, 1, T)
	for n in range(N):
		for k in range(K):
			outer = torch.outer(loadings[:, k], loadings[:, k]).unsqueeze(2)
			out[n, :, :, :] += outer * latent_process[n, k, :].pow(2.)
	if return_cholesky_lower:
		out = torch.linalg.cholesky(out.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)
	return out