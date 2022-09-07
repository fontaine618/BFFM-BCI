import torch
from models.bffmbci.variables import ObservedVariable
from models.bffmbci.variables import GaussianObservations
from models.bffmbci.variables import ObservationVariance
from models.bffmbci.variables import Loadings, Heterogeneities, ShrinkageFactor

# GaussianObservations

N, E, K, T = 10, 15, 5, 100
loadings = ObservedVariable(torch.randn((E, K)))
loading_processes = ObservedVariable(torch.randn((N, K, T)))
factor_processes = ObservedVariable(torch.randn((N, K, T)))
observation_variance = ObservedVariable(torch.randn((E, )).exp())
sd = observation_variance.data.sqrt()

mean = torch.einsum(
	"ek, nkt, nkt -> net",
	loadings.data,
	loading_processes.data,
	factor_processes.data
)
value = mean + torch.randn_like(mean) * sd.unsqueeze(0).unsqueeze(2)

observations = GaussianObservations(
	value=value,
	observation_variance=observation_variance,
	loadings=loadings,
	loading_processes=loading_processes,
	factor_processes=factor_processes
)

observations.residuals.pow(2).mean((0, 2)).sqrt()
sd

# ObservationVariance
import torch

observation_variance = ObservationVariance(n_channels=E, prior_parameters=(1., 10.))
observation_variance.add_children(observations)
observation_variance.data

for _ in range(100):
	observation_variance.sample()

observation_variance.chain.mean(0)
sd.pow(2)

# Loadings
heterogeneities = ObservedVariable(torch.randn((E, K)).exp()+1.)
shrinkage_factor = ObservedVariable(torch.linspace(1., 100., K))

loadings = Loadings(heterogeneities, shrinkage_factor)
loadings.add_children(observations, observation_variance)
loadings.data

for _ in range(100):
	loadings.sample()

loadings.chain.mean(0)
loadings.data

# Heterogeneities
self = Heterogeneities(dim=(E, K), gamma=3.)
self.add_children(loadings)
self.data

for _ in range(100):
	self.sample()

self.chain.mean(0)
heterogeneities.data

# ShrinkageFactor
shrinkage_factor = ShrinkageFactor(n_latent=K)
shrinkage_factor.data
shrinkage_factor.add_children(loadings)

for _ in range(100):
	shrinkage_factor.sample()

shrinkage_factor.chain.mean(0)
shrinkage_factor.data



# =============================================================================
# GPs
import torch
import torch.linalg
import scipy.linalg
import matplotlib.pyplot as plt
from models.bffmbci.variables import GaussianProcess, TruncatedGaussianProcess01
from models.bffmbci.utils import Kernel
from models.bffmbci.variables import SequenceData, SMGP, Superposition
from models.bffmbci.variables import ObservedVariable
from models.bffmbci.variables import GaussianObservations
from models.bffmbci.variables import ObservationVariance
from models.bffmbci.variables import Loadings, Heterogeneities, ShrinkageFactor
import torch.nn.functional as F

E = 15
J = 12
K, W, d = 3, 55, 10
T = (J-1) * d + W
N = 4

cov = torch.Tensor(scipy.linalg.toeplitz(0.99**torch.arange(W)))
kernel_gp = Kernel.from_covariance_matrix(cov)
cov = torch.Tensor(scipy.linalg.toeplitz(0.99**torch.arange(W)) * 0.5)
kernel_tgp = Kernel.from_covariance_matrix(cov)
order = torch.vstack([torch.randperm(12) for _ in range(N)])
target = torch.hstack([
	torch.randint(0, 6, (N, 1)),
	torch.randint(6, 12, (N, 1))
])
target = F.one_hot(target, num_classes=J).max(1).values
sequence_data = SequenceData(order, target)
smgp = SMGP(K, kernel_gp, kernel_tgp)
superposition = Superposition(
	smgp=smgp,
	sequence_data=sequence_data,
	stimulus_to_stimulus_interval=d,
	window_length=W
)

loadings = ObservedVariable(torch.randn((E, K)))
factor_processes = ObservedVariable(torch.randn((N, K, T)))
observation_variance = ObservedVariable(torch.randn((E, )).exp())
sd = observation_variance.data.sqrt()

mean = torch.einsum(
	"ek, nkt, nkt -> net",
	loadings.data,
	superposition.data,
	factor_processes.data
)
value = mean + torch.randn_like(mean) * sd.unsqueeze(0).unsqueeze(2)

observations = GaussianObservations(
	value=value,
	observation_variance=observation_variance,
	loadings=loadings,
	loading_processes=superposition,
	factor_processes=factor_processes
)

superposition.child = observations
smgp.superposition = superposition

# for updates
k = 0
which = "nontarget_process"
smgp.sample()

plt.plot(self.data[:, 0, :].T)
plt.show()

sequence_data.order.data
sequence_data.target.data

smgp.mixing_process.data








# =============================================================================
# everything together
import torch
import cProfile
import numpy as np
import torch.linalg
import scipy.linalg
import matplotlib.pyplot as plt
from models.bffmbci.variables import GaussianProcess, TruncatedGaussianProcess01
from models.bffmbci.utils import Kernel
from models.bffmbci.variables import SequenceData, SMGP, Superposition
from models.bffmbci.variables import ObservedVariable
from models.bffmbci.variables import GaussianObservations
from models.bffmbci.variables import ObservationVariance
from models.bffmbci.variables import Loadings, Heterogeneities, ShrinkageFactor
from models.bffmbci.variables import NoisyProcesses
import torch.nn.functional as F

import matplotlib.pyplot as plt
plt.style.use("seaborn-white")



np.random.seed(0)
torch.manual_seed(0)

E = 15
J = 12
K, W, d = 3, 55, 10
T = (J-1) * d + W
N = 19*15

# Generative model
observation_variance = ObservationVariance(n_channels=E, prior_parameters=(1., 10.))
heterogeneities = Heterogeneities(dim=(E, K), gamma=3.)
shrinkage_factor = ShrinkageFactor(n_latent=K, prior_parameters=(1., 5.))
loadings = Loadings(heterogeneities, shrinkage_factor)
order = torch.vstack([torch.randperm(12) for _ in range(N)])
target = torch.hstack([
	torch.randint(0, 6, (N, 1)),
	torch.randint(6, 12, (N, 1))
])

sequence_data = SequenceData(order, target)
cov = torch.Tensor(scipy.linalg.toeplitz(0.99**np.arange(W)))
kernel_gp = Kernel.from_covariance_matrix(cov)
cov = torch.Tensor(scipy.linalg.toeplitz(0.99**np.arange(W)) * 0.5)
kernel_tgp = Kernel.from_covariance_matrix(cov)
smgp_loadings = SMGP(K, kernel_gp, kernel_tgp)
loading_processes = Superposition(
	smgp=smgp_loadings,
	sequence_data=sequence_data,
	stimulus_to_stimulus_interval=d,
	window_length=W
)
loading_processes.name = "loading_processes"
smgp_factors = SMGP(K, kernel_gp, kernel_tgp)
mean_factor_processes = Superposition(
	smgp=smgp_factors,
	sequence_data=sequence_data,
	stimulus_to_stimulus_interval=d,
	window_length=W
)
mean_factor_processes.name = "factor_processes"
cov = torch.Tensor(scipy.linalg.toeplitz(0.99**torch.arange(T))*0.01)
kernel_factor = Kernel.from_covariance_matrix(cov)
factor_processes = NoisyProcesses(
	mean=mean_factor_processes,
	kernel=kernel_factor
)

observations = GaussianObservations(
	observation_variance=observation_variance,
	loadings=loadings,
	loading_processes=loading_processes,
	factor_processes=factor_processes,
	value=None,
)

# Link backwards
shrinkage_factor.add_children(loadings=loadings)
heterogeneities.add_children(loadings=loadings)
loadings.add_children(
	observations=observations,
    observation_variance=observation_variance
)
observation_variance.add_children(observations=observations)
# sequence data has children, but we don't care
smgp_loadings.add_children(superposition=loading_processes)
smgp_factors.add_children(superposition=mean_factor_processes)
loading_processes.add_children(observations=observations)
mean_factor_processes.add_children(child=factor_processes, observations=observations)
factor_processes.add_children(observations=observations)

print(smgp_factors)
print(observations)



# sampling order
observations.sample()
print("sampling loadings")
loadings.sample()
heterogeneities.sample()
shrinkage_factor.sample()
print("sampling observation variance")
observation_variance.sample()
print("sampling factor processes")
factor_processes.sample()
mean_factor_processes.sample()
smgp_factors.sample()
print("sampling loadings processes")
loading_processes.sample()
smgp_loadings.sample()
sequence_data.sample()


cProfile.run("smgp_loadings.sample()")


from torch.autograd.functional import jacobian
self = smgp_loadings.nontarget_process
value = self.data
k = 0


self = loading_processes
n = 0
j = 3

ynj = yn[j]


p_inj = (1 - ynj * mixing_process) * nontarget_process + \
        ynj * mixing_process * target_process

ynn = yn.unsqueeze(0).unsqueeze(0)
p_in = (1 - ynn * mixing_process.unsqueeze(-1)) * nontarget_process.unsqueeze(-1) + \
        ynn * mixing_process.unsqueeze(-1) * target_process.unsqueeze(-1)



# noisy process
from torch.autograd.functional import jacobian
from functorch import vmap, vjp, jacfwd, jacrev, make_fx
from functorch.experimental import functionalize
from functorch import make_functional_with_buffers
from functools import partial
import torch.nn
self = factor_processes
value = self.data
k = 0


zk = self.observations.factor_processes.data[:, k, :]

def f(l, lp, fp, zk):
	fp[:, k, :] = zk
	return torch.einsum(
		"ek, nkt, nkt -> net",
		l, lp, fp
	)

compute_batch_jacobian = vmap(
	jacrev(f, argnums=3), (None, 0, 0, 0), 0
)
L = compute_batch_jacobian(
	self.observations.loadings.data,
	self.observations.loading_processes.data,
	self.observations.factor_processes.data,
	zk
)

compute_batch_jacobian = vmap(jacfwd(fk))


class TmpModel(torch.nn.Module):

	def __init__(self, loadings, loading_processes, factor_processes, k):
		super().__init__()
		self.loadings = torch.nn.Parameter(loadings.data, requires_grad=False)
		self.loading_processes = torch.nn.Parameter(loading_processes.data, requires_grad=False)
		self.factor_processes = torch.nn.Parameter(factor_processes.data, requires_grad=False)
		self.k = k

	def forward(self, zk):
		factor_processes = self.factor_processes
		factor_processes[:, self.k, :] = zk
		return torch.einsum(
			"ek, nkt, nkt -> net",
			self.loadings, self.loading_processes, factor_processes
		)

model = TmpModel(loadings, loading_processes, factor_processes, k)
model.forward(zk)
fmodel, params, buffers = make_functional_with_buffers(model)

f = functionalize(model.forward)
compute_batch_jacobian = vmap(jacrev(f))
compute_batch_jacobian = vmap(jacrev(model.forward))
compute_batch_jacobian = vmap(jacfwd(model.forward))
L = compute_batch_jacobian(zk)

print(make_fx(f)(zk).code)




for i in range(100):
	print(i)
	smgp_loadings.mixing_process.sample()

cProfile.run("smgp_loadings.sample()")


self = factor_processes
self = smgp_loadings.target_process

plt.imshow(prec)
plt.colorbar()
plt.show()



self = smgp_loadings.mixing_process
value = torch.Tensor(self.mean.data)
k= 1
p0 = self.kernel.inv
mtp0 = p0 @ self.mean.data[k, :]
p1, mtp1 = self._message_from_child(value, k)
with torch.no_grad():
	prec = p0 + p1
	mtp = mtp0 + mtp1
	c = torch.inverse(prec)
	m = c @ mtp

smgp_loadings.mixing_process.sample()

x = smgp_loadings.mixing_process.chain[:, k, :].detach().numpy()
plt.plot(x.T, alpha=0.2)
plt.plot(x[0, :], color="black")
plt.plot(x[-1, :], color="blue")
plt.plot(m, color="red")
plt.plot(self.mean.data[k, :].detach().numpy(), color="purple")
plt.plot(x.mean(0), color="green")
plt.show()





for i in range(100):
	print(i)
	smgp_loadings.mixing_process.sample()


k = 2


x = smgp_loadings.mixing_process.chain[:, k, :].detach().numpy()
plt.plot(x.T, alpha=0.2)
plt.plot(x[0, :], color="black")
plt.plot(x.mean(0), color="green")
plt.show()



# Sampling test
for _ in range(100):
	observation_variance.sample()
	loadings.sample_recursively()

shrinkage_factor.chain.mean(0)
shrinkage_factor.chain[0, :]

heterogeneities.chain.mean(0)
heterogeneities.chain[0, :, :]

loadings.chain.mean(0)
loadings.chain[0, :, :]

observation_variance.chain.mean(0)
observation_variance.chain[0, :]


