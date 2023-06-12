import torch
import math
import warnings

ROOT_TWO_PI = math.sqrt(2 * math.pi)
ROOT_TWO_OVER_PI = math.sqrt(2 / math.pi)
STD_NORMAL = torch.distributions.normal.Normal(0, 1)

COUNTS = { # first is the number of proposals, second is the number of accepted proposals
	"uniform": [0, 0],
	"normal": [0, 0],
	"halfnormal": [0, 0],
	"exponential": [0, 0],
}


class TruncatedStandardMultivariateGaussian:

	def __init__(
			self,
			rotation,
			lower,
			upper
	):
		self._rotation = rotation
		self._lower = lower
		self._upper = upper
		self._dim = lower.shape

	def sample(self, value):
		eps = ((self._upper - self._lower) * 1e-6).clamp(max=1e-6)
		value.clamp_(min=self._lower + eps, max=self._upper - eps)
		for i in range(self._dim[0]):
			self._sample_i(value, i)
		return value

	def _sample_i(self, value, i):
		r_i = self._rotation[:, i]
		which = torch.where(torch.arange(0, self._dim[0]) == i, False, True)
		r_mi = self._rotation[:, which]
		x_mi = value[which]
		a_mrx = self._lower - r_mi @ x_mi
		b_mrx = self._upper - r_mi @ x_mi
		j_pos = torch.where(r_i > 0.)[0]
		j_neg = torch.where(r_i < 0.)[0]
		# the case r_ij == 0 is considered by default implicitly
		l_pos, l_neg, u_pos, u_neg = -torch.inf, -torch.inf, torch.inf, torch.inf
		if len(j_pos):
			l_pos = (a_mrx[j_pos] / r_i[j_pos]).max()
			u_pos = (b_mrx[j_pos] / r_i[j_pos]).min()
		if len(j_neg):
			l_neg = (b_mrx[j_neg] / r_i[j_neg]).max()
			u_neg = (a_mrx[j_neg] / r_i[j_neg]).min()
		l = max(l_pos, l_neg)
		u = min(u_pos, u_neg)
		if l > u:
			# print(f"{i}: No valid solution (l > u) l={l:<10.3f}, u={u:<10.3f}")
			return # we do not update value[i] in this case
		elif l > u - 1e-10:
			value[i] = l
			return
		value[i] = _truncated_standard_normal_rv(l, u)


class TruncatedMultivariateGaussian(TruncatedStandardMultivariateGaussian):

	def __init__(
			self,
			mean,
			covariance=None,
			cholesky=None,
			rotation=None,
			lower=None,
			upper=None
	):
		cholesky, rotation, lower, upper = self._check_args(cholesky, covariance, lower, rotation, upper)
		rotation_ = rotation @ cholesky
		lower_ = lower - rotation @ mean
		upper_ = upper - rotation @ mean
		super().__init__(rotation=rotation_, lower=lower_, upper=upper_)
		self._mean = mean
		self._cholesky = cholesky
		self._original_lower = lower
		self._original_upper = upper
		# self._cholesky_inverse = torch.linalg.inv(cholesky)

	def _check_args(self, cholesky, covariance, lower, rotation, upper):
		if covariance is not None:
			cholesky = torch.linalg.cholesky(covariance)
		if cholesky is None:
			raise ValueError("need at least covariance or cholesky")
		p = cholesky.shape[0]
		if rotation is None:
			rotation = torch.eye(p)
		if lower is None:
			lower = torch.zeros(p)
		if upper is None:
			upper = torch.ones(p)
		return cholesky, rotation, lower, upper

	def sample(self, value):
		r"""value is a current value to take a single step from"""
		value = torch.linalg.solve_triangular(
			self._cholesky,
			(value - self._mean).reshape(-1, 1),
			upper=False
		).reshape(-1)

		value = super().sample(value)
		out = self._mean + self._cholesky @ value
		if (out < self._original_lower).any() or (out > self._original_upper).any():
			warnings.warn("TG sampling outside limits")
		out.clamp_(min=self._original_lower, max=self._original_upper)
		return out


def _truncated_standard_normal_rv_icdf(a: torch.Tensor, b: torch.Tensor):
	u = torch.rand(1)
	a_normal_cdf = STD_NORMAL.cdf(a)
	b_normal_cdf = STD_NORMAL.cdf(b)
	p = a_normal_cdf + (b_normal_cdf - a_normal_cdf) * u
	return STD_NORMAL.icdf(p).clamp(a, b)


def _truncated_standard_normal_rv_normal_rejection(a: torch.Tensor, b: torch.Tensor):
	i = 0
	while True:
		i = i + 1
		COUNTS["normal"][0] += 1
		z = STD_NORMAL.sample()
		if a <= z <= b:
			# print(f"Normal rejection took {i} iterations [{a.item():.2f},{b.item():.2f}]")
			COUNTS["normal"][1] += 1
			return z
		if i > 10:
			print(f"Normal rejection {i} iterations [{a.item():.2f},{b.item():.2f}]")


def _truncated_standard_normal_rv_halfnormal_rejection(a: torch.Tensor, b: torch.Tensor):
	# if a < 0:
	# 	raise ValueError("a must be non-negative")
	i = 0
	while True:
		i = i + 1
		COUNTS["halfnormal"][0] += 1
		z = abs(STD_NORMAL.sample())
		if a <= z <= b:
			# print(f"Halfnormal rejection took {i} iterations [{a.item():.2f},{b.item():.2f}]")
			COUNTS["halfnormal"][1] += 1
			return z
		if i > 10:
			print(f"Halfnormal rejection {i} iterations [{a.item():.2f},{b.item():.2f}]")


def _truncated_standard_normal_rv_uniform_rejection(a: torch.Tensor, b: torch.Tensor):
	if a > 0:
		M = STD_NORMAL.log_prob(a)
	elif b < 0:
		M = STD_NORMAL.log_prob(b)
	else:
		M = STD_NORMAL.log_prob(torch.Tensor([0]))
	i = 0
	while True:
		i = i + 1
		COUNTS["uniform"][0] += 1
		z = torch.rand(1) * (b - a) + a
		u = torch.rand(1)
		if u <= (STD_NORMAL.log_prob(z) - M).exp().clamp(min=0.01):
			# print(f"Uniform rejection took {i} iterations [{a.item():.2f},{b.item():.2f}]")
			COUNTS["uniform"][1] += 1
			return z
		if i > 10:
			print(f"Uniform rejection {i} iterations [{a.item():.2f},{b.item():.2f}]")


def _truncated_standard_normal_rv_exponential_rejection(a: torch.Tensor, b: torch.Tensor):
	# if a < 0:
	# 	raise ValueError("a must be non-negative")
	if a > 1e10:
		lam = a
	else:
		lam = 0.5 * (a + torch.sqrt(a ** 2 + 4))
	i = 0
	while True:
		i = i + 1
		COUNTS["exponential"][0] += 1
		z = torch.distributions.Exponential(lam).sample() + a
		u = torch.rand(1)
		if z <= b and u <= (-0.5 * (z - lam) ** 2).exp().clamp(min=0.01):
			# print(f"Exponential rejection took {i} iterations [{a.item():.2f},{b.item():.2f}]")
			COUNTS["exponential"][1] += 1
			return z
		if i > 10:
			print(f"Exponential rejection {i} iterations [{a.item():.2f},{b.item():.2f}]")


def _truncated_standard_normal_rv(a: torch.Tensor, b: torch.Tensor):
	# dispatch
	if b == torch.inf:
		if a <= 0:
			return _truncated_standard_normal_rv_normal_rejection(a, b)
		if a < ROOT_TWO_OVER_PI:
			return _truncated_standard_normal_rv_halfnormal_rejection(a, b)
		return _truncated_standard_normal_rv_exponential_rejection(a, b)
	if a == -torch.inf:
		return -_truncated_standard_normal_rv(-b, -a)
	if a <= 0 <= b:
		if b - a < ROOT_TWO_PI:
			return _truncated_standard_normal_rv_uniform_rejection(a, b)
		return _truncated_standard_normal_rv_normal_rejection(a, b)
	if a > 0:
		if a < ROOT_TWO_OVER_PI:
			if b <= a + ROOT_TWO_OVER_PI / (a**2 / 2).exp():
				return _truncated_standard_normal_rv_uniform_rejection(a, b)
			return _truncated_standard_normal_rv_halfnormal_rejection(a, b)
		else:
			b2 = a + 2 * ((a*a - a * (a*a+4).sqrt()) / 4. + 0.5).exp() / (a + (a*a+4).sqrt())
			if b <= b2:
				return _truncated_standard_normal_rv_uniform_rejection(a, b)
			return _truncated_standard_normal_rv_exponential_rejection(a, b)
	if b < 0:
		return -_truncated_standard_normal_rv(-b, -a)
	raise ValueError(f"Trunacted normal smapling error: invalid interval [{a},{b}]")

