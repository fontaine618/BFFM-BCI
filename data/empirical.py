import torch


def mean(
		subsequences: torch.Tensor,
		target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
	"""

	:param subsequences: (NJ, E, T)
	:param target: (NJ, )
	:return: (E, T)
	"""
	target_mean = subsequences[target == 1., :, :].mean(0)
	nontarget_mean = subsequences[target == 0., :, :].mean(0)
	return nontarget_mean, target_mean


def batch_cov(points):
	B, N, D = points.size()
	mean = points.mean(dim=1).unsqueeze(1)
	diffs = (points - mean).reshape(B * N, D)
	prods = torch.bmm(diffs.unsqueeze(2), diffs.unsqueeze(1)).reshape(B, N, D, D)
	bcov = prods.sum(dim=1) / (N - 1)  # Unbiased estimate
	return bcov  # (B, D, D)


def covariance(
		subsequences: torch.Tensor,
		target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
	"""

	:param subsequences: (NJ, E, T)
	:param target: (NJ, )
	:return: 2x(E, E, T)
	"""
	target_cov = batch_cov(subsequences[target == 1., :, :].permute(2, 0, 1)).permute(1, 2, 0)
	nontarget_cov = batch_cov(subsequences[target == 0., :, :].permute(2, 0, 1)).permute(1, 2, 0)
	return nontarget_cov, target_cov


def precision(
		subsequences: torch.Tensor,
		target: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
	"""

	:param subsequences: (NJ, E, T)
	:param target: (NJ, )
	:return: 2x (E, E, T)
	"""
	nontarget_cov, target_cov = covariance(subsequences, target)
	nontarget_precison = torch.stack([
		torch.linalg.inv(nontarget_cov[:, :, i]) for i in range(nontarget_cov.shape[2])
	], 2)
	target_precison = torch.stack([
		torch.linalg.inv(target_cov[:, :, i]) for i in range(target_cov.shape[2])
	], 2)
	return nontarget_precison, target_precison
