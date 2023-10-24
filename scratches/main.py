import torch
import math
import pandas as pd
from source.bffmbci.bffm import BFFModel

from source.bffmbci.bffm import DynamicRegressionCovarianceRegressionMean
from source.bffmbci.bffm import DynamicCovarianceRegressionMean
from source.bffmbci.bffm import StaticCovarianceRegressionMean
from source.bffmbci.bffm import DynamicRegressionCovarianceStaticMean

from source.results_old.mcmc_results import MCMCResults
import matplotlib.pyplot as plt
from torch.autograd.functional import jacobian, hessian
plt.style.use("seaborn-v0_8-whitegrid")

torch.set_default_tensor_type(torch.cuda.FloatTensor)

latent_dim = 5
n_iter = 100
n_chains = 5
batchsize = 10
torch.manual_seed(0)
model = BFFModel.generate_from_dimensions(
	latent_dim=latent_dim,
	n_channels=15,
	stimulus_to_stimulus_interval=10,
	stimulus_window=55,
	n_characters=17,
	n_repetitions=15,
	sparse=False,
	shrinkage="none",
	covariance="dynamic",
	mean_regression=False
)
model = DynamicRegressionCovarianceStaticMean.generate_from_dimensions(
	latent_dim=latent_dim,
	n_channels=15,
	stimulus_to_stimulus_interval=10,
	stimulus_window=55,
	n_characters=17,
	n_repetitions=15,
	sparse=False,
	shrinkage="none"
)

model.variables["loading_processes"].data
model.variables["factor_processes"].data
model.variables["mean_factor_processes"].data

self = model.variables["factor_processes"]
self.generate()
model.variables["observations"].generate()

self.data = self.posterior_mean
self.data = self.posterior_mean_by_conditionals
print(self.log_density + model.variables["observations"].log_density)




self = model.variables["smgp_factors"].mixing_process



true_values = model.data
true_llk = model.variables["observations"].log_density
true_values["observation_log_likelihood"] = true_llk
self = model

# torch.manual_seed(0)
model.initialize_chain()

# run for a bit
# torch.manual_seed(0)
for i in range(1000):
	model.sample()
	llk = model.variables['observations'].log_density_history[-1]
	print(f"[MCMC] Iteration: {i:>4} Log-likelihood: "
		  f"{llk:>20.2f}")
	if abs(llk) > 1e8:
		break




import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")

plt.cla()
plt.plot(
	model.variables["mean_factor_processes"].data[11, :, :].cpu().T
)
# plt.yscale("log")
plt.show()


plt.cla()
plt.plot(
	model.variables["loading_processes"].data[10, :, :].cpu().T
)
# plt.yscale("log")
plt.show()

model.sample()
plt.cla()
plt.plot(
	model.variables["smgp_scaling"].target_process.data.cpu().T
)
plt.show()


from torch.autograd.functional import jacobian


self = model.variables["smgp_scaling"].target_process
k = 0
value = self._value.data







self = model.variables["smgp_scaling"].nontarget_process
print(self.n_evals, self.n_proposals, self.n_accepts, self.n_accepts / self.n_proposals)
self = model.variables["smgp_scaling"].target_process
print(self.n_evals, self.n_proposals, self.n_accepts, self.n_accepts / self.n_proposals)
self = model.variables["smgp_scaling"].mixing_process
print(self.n_evals, self.n_proposals, self.n_accepts, self.n_accepts / self.n_proposals)

self = model.variables["smgp_factors"].nontarget_process
print(self.n_evals, self.n_proposals, self.n_accepts, self.n_accepts / self.n_proposals)
self = model.variables["smgp_factors"].target_process
print(self.n_evals, self.n_proposals, self.n_accepts, self.n_accepts / self.n_proposals)
self = model.variables["smgp_factors"].mixing_process
print(self.n_evals, self.n_proposals, self.n_accepts, self.n_accepts / self.n_proposals)

self = model.variables["smgp_factors"].target_process

self = self.superposition.observations





chain_id = 1


for chain_id in range(n_chains):
	torch.manual_seed(chain_id)
	status = False
	while not status:
		try:
			model.initialize_chain()
			status = True
		except Exception as e:
			print(e)
	for i in range(n_iter):
		model.sample()
		print(chain_id, i, model.variables["observations"].log_density_history[-1])
	model.results().save(f"/home/simon/Documents/BCI/experiments/test/chains/seed{chain_id}.chain")


results = {
	chain_id: MCMCResults.load(
		f"/home/simon/Documents/BCI/experiments/test/chains/seed{chain_id}.chain"
	) for chain_id in range(n_chains-1)
}


self = results[0]

self.moving_metrics(true_values, 10)


# compute metrics every 100 iterations
metrics100 = {}
for chain_id in range(n_chains):
	metrics100[chain_id] = {}
	for i in range(0, n_iter, 100):
		variables = results[chain_id].variables
		for name, var in variables.items():
			which = torch.arange(i, i+100)
			variables[name] = var.index_select(0, which)
		result = MCMCResults(variables)
		metrics100[chain_id][i] = result.metrics(true_values)


# for var in metrics[0].keys():
# 	for metric in metrics[0][var].keys():
# 		filename = f"{dir}{var}.{metric}.pdf"







# Sampling
results = {}
for i in range(1, n_iter+1):
	model.sample()
	lkki = model.variables["observations"].log_density
	print(i, lkki)
	if i%100 == 0:
		print(i, " stored")
		results[i%100] = model.results()
		model.clear_history()

self = model.results()
m0 = self.metrics(true_values)

self._compute_posterior_mean()
post = self.posterior_mean


# Metrics
metrics = {}
llks = {}
for i in range(n_iter):
	model.sample()
	lkki = model.variables["observations"].log_density
	llks[i] = lkki
	print(i, lkki)
	if i % 100 == 0 and i > 0:
		results = model.results()
		metrics[i] = results.metrics(true_values)
		model.clear_history()

dir = "./source/figures/metrics10k/"

for var in metrics[0].keys():
	for metric in metrics[0][var].keys():
		filename = f"{dir}{var}.{metric}.pdf"
		iter = metrics.keys()
		m = [metrics[i][var][metric] for i in iter]
		df = pd.DataFrame(m, index=iter)

		plt.figure(figsize=(6, 4))
		df.plot()
		plt.title(var)
		plt.ylabel(metric)
		plt.xlabel("iteration")
		plt.tight_layout()
		plt.savefig(filename)

filename = f"{dir}log_likelihood.pdf"
df = pd.DataFrame(llks.values(), index=llks.keys())
plt.figure(figsize=(6, 4))
df.plot()
plt.title("observation")
plt.ylabel("log likelihood")
plt.xlabel("iteration")
plt.tight_layout()
plt.savefig(filename)













# initialization
model.variables["observation_variance"].data
true_values["observation_variance"]

model.variables["loadings"].data
true_values["loadings"]

model.variables["heterogeneities"].data
true_values["heterogeneities"]

model.variables["shrinkage_factor"].data
true_values["shrinkage_factor"]


L1 = model.variables["loadings"].data
L0 = true_values["loadings"]
L1n = L1 / L1.norm(2, 0).unsqueeze(0)
L0n = L0 / L0.norm(2, 0).unsqueeze(0)
L1n.T @ L0n




model.variables["observations"].data.pow(2.)._mean((0, 2))

for iter in range(n_iter):
	model.sample()
	if iter % 10 == 0:
		print(iter)

import torch
from source.initialization.wfa import WFA
torch.manual_seed(0)
N = 10000
K, p = 3, 7
w = torch.randn(N, ).pow(2.)
# w = torch.ones_like(w)
Z = torch.randn(N, K) * w.sqrt().unsqueeze(1)
L = torch.randn(K, p)
L = L / L.norm(2, 1).unsqueeze(1)
L = L * torch.linspace(1., 0.1, K).unsqueeze(1)
sig2 = torch.randn(p, ).pow(2.)
X = Z @ L + torch.randn(N, p) * sig2.sqrt().unsqueeze(0)
self = WFA(K)
self.fit(X, w, tol=1e-6)

self._loadings
L.T

L.T @ L

# check alignment
L1n = self._loadings / self._loadings.norm(2, 0).unsqueeze(0)
L0n = L.T / L.T.norm(2, 0).unsqueeze(0)
L1n.T @ L0n

sig2
self._observation_variance