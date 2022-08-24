import torch


def squared_exponential_time_kernel(
		n_steps: int = 640,
		scale: float = 100.,
		power: float = 0.5,
		return_lower_cholesky: bool = True,
		tikhonov_regularization: float = 1.e-10
) -> torch.Tensor:
	"""
	Computes the squared exponential kernel for regularly spaced time points.

	exp{-((t-t')^2/scale)^{power}}

	:param n_steps:
	:param scale:
	:param power:
	:param return_lower_cholesky:
	:param tikhonov_regularization:
	:return:
	"""
	T = n_steps
	kernel = torch.arange(0, T).reshape((-1, 1)) - torch.arange(0, T).reshape((1, -1))
	kernel = kernel.pow(2.).mul(1./scale).pow(power).mul(-1.).exp()
	kernel = kernel + torch.eye(T) * tikhonov_regularization
	if return_lower_cholesky:
		return torch.linalg.cholesky(kernel)
	else:
		return kernel


def split_merge_gaussian_process(
		n_out: int,
		kernel_gp: torch.Tensor,
		kernel_mixing: torch.Tensor,
		return_gp_and_mixing: bool = False
) -> tuple[torch.Tensor]:
	"""

	:param n_out:
	:param kernel_gp:
	:param kernel_mixing:
	:param return_gp_and_mixing:
	:return:
		- a0 the GPs for nontarget signal (n_out, n_steps)
		- a1 the GPs for target signal (n_out, n_steps)
		- zeta the [0,1] process] for mixing signal (n_out, n_steps)
		- b0 the process for nontarget signal (n_out, n_steps)
		- b1 theprocess for target signal (n_out, n_steps)
	"""
	T = kernel_gp.shape[0]
	a0 = torch.randn((n_out, T))
	a0 = torch.matmul(a0, kernel_gp.T)
	a1 = torch.randn((n_out, T))
	a1 = torch.matmul(a1, kernel_gp.T)
	zeta = torch.randn((n_out, T))
	zeta = torch.matmul(zeta, kernel_mixing.T) * 5.
	zeta = 1. / (1. + zeta.neg().exp())
	b0 = a0
	b1 = zeta * a1 + (1 - zeta) * a0
	if return_gp_and_mixing:
		return a0, a1, zeta, b0, b1
	else:
		return b0, b1