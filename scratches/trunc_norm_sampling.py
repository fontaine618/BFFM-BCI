import torch
from src.bffmbci.utils.truncated_gaussian import \
    _truncated_standard_normal_rv_normal_rejection, \
    _truncated_standard_normal_rv_icdf, \
    _truncated_standard_normal_rv_halfnormal_rejection, \
    _truncated_standard_normal_rv_uniform_rejection, \
    _truncated_standard_normal_rv_exponential_rejection, \
    _truncated_standard_normal_rv
import matplotlib.pyplot as plt
import cProfile
a, b = 4742907400770082045952, 5067027525302262693888
normal = torch.distributions.normal.Normal(0, 1)
a, b = torch.Tensor([a]), torch.Tensor([b])
dcdf = normal.cdf(b) - normal.cdf(a)


sample = [
    _truncated_standard_normal_rv(a, b).item()
    for _ in range(10000)
]

plt.cla()
plt.hist(sample, bins=20, density=True)
plt.plot(
    torch.linspace(a.item(), b.item(), 100),
    normal.log_prob(torch.linspace(a.item(), b.item(), 100)).exp() * 1e23,
    # normal.log_prob(torch.linspace(a.item(), b.item(), 100)).exp() / dcdf
)
# plt.yscale('log')
plt.show()

cProfile.run('[_truncated_standard_normal_rv_icdf(a, b) for _ in range(10000)]')
cProfile.run('[_truncated_standard_normal_rv_normal_rejection(a, b) for _ in range(10000)]')
cProfile.run('[_truncated_standard_normal_rv_halfnormal_rejection(a, b) for _ in range(10000)]')
cProfile.run('[_truncated_standard_normal_rv_uniform_rejection(a, b) for _ in range(10000)]')
cProfile.run('[_truncated_standard_normal_rv_exponential_rejection(a, b) for _ in range(10000)]')


# MVT
import torch
from src.bffmbci.utils.truncated_gaussian import TruncatedMultivariateGaussian
from matplotlib import pyplot as plt
from torch.distributions import MultivariateNormal

mean = torch.Tensor([0, 0])
covariance = torch.Tensor([[1, 0.8], [0.8, 1]])
lower = torch.Tensor([-1, -3])
upper = torch.Tensor([1, 3])

self = TruncatedMultivariateGaussian(
    mean=mean,
    covariance=covariance,
    lower=lower,
    upper=upper,
)

value = torch.Tensor([0.5, 0.5])
samples = value.reshape(1, -1)
for _ in range(10000):
    value = self.sample(value)
    samples = torch.cat([samples, value.reshape(1, -1)], dim=0)

# get contour plot
x = torch.linspace(lower[0], upper[0], 100)
y = torch.linspace(lower[1], upper[1], 100)
X, Y = torch.meshgrid(x, y)
Z = torch.cat([X.reshape(-1, 1), Y.reshape(-1, 1)], dim=1)

mvt_norm = MultivariateNormal(mean, covariance)
log_pdf = mvt_norm.log_prob(Z).exp().reshape(100, 100)

plt.cla()
samples = samples[1:, :]
plt.plot(samples.cpu()[:, 0], samples.cpu()[:, 1], 'o', alpha=0.1)
plt.contour(X.cpu(), Y.cpu(), log_pdf.cpu())
# plt.gca().set_aspect('equal')
plt.show()