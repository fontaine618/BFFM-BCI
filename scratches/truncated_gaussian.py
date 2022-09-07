import torch
from models.bffmbci.utils import TruncatedMultivariateGaussian

torch.manual_seed(0)

# parameters
p = 5
mean = torch.randn(p)
Sigma = torch.randn((p, p))
covariance = Sigma @ Sigma.T
rotation = torch.eye(p)
lower = torch.randn(p)
upper = lower + torch.rand(p) * 3

value = lower + torch.rand(p) * (upper - lower)

print(lower)
print(value)
print(upper)

self = TruncatedMultivariateGaussian(
	mean=mean,
	covariance=covariance,
	rotation=rotation,
	lower=lower,
	upper=upper
)

print(lower)
print(self.sample(value))
print(upper)

