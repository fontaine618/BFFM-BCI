

# singularity of precision matrices

import torch
import pandas as pd
from models.bffmbci.bffm import BFFModel
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
plt.style.use("seaborn-whitegrid")

torch.set_default_tensor_type(torch.cuda.FloatTensor)

order = [
	"loading_processes",
	"smgp_loadings",
	"loadings",
	"shrinkage_factor",
	"heterogeneities",
]
latent_dim = 3
n_iter = 100
torch.manual_seed(0)
model = BFFModel.generate_from_dimensions(
	latent_dim=latent_dim,
	observation_variance=(3., 1.),
	independent_smgp=False
)
for _ in range(n_iter):
	model.sample(order)
	print(model.variables["observations"].log_density)


k = 0
var = "smgp_loadings.mixing_process"
# var = "loadings"

fig = plt.figure(figsize=(6, 4))
ax = fig.add_subplot(111)

if "." in var:
	v1, v2 = var.split(".")
	varobj = model.variables[v1][v2]
else:
	varobj = model.variables[var]
chain = varobj.chain[:, k].cpu().detach()

ax.plot(chain)
# ax.set_xscale("log")
ax.set_xlabel("MCMC iteration")
ax.set_ylabel(var)
fig.tight_layout()
fig.savefig(f"./code/figures/chains_smgp/{var}_k{k}_dim{latent_dim}.pdf")




# singularity of precision matrices

import torch
import pandas as pd
from models.bffmbci.bffm import BFFModel
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
plt.style.use("seaborn-whitegrid")

torch.set_default_tensor_type(torch.cuda.FloatTensor)

order = [
	"loading_processes",
	"smgp_loadings",
]
latent_dim = 3
n_iter = 10
torch.manual_seed(0)
model = BFFModel.generate_from_dimensions(
	latent_dim=latent_dim,
	observation_variance=(3., 1.)
)

for _ in range(n_iter):
	model.sample(order)
	print(model.variables["observations"].log_density)


self = model.variables["smgp_loadings"].target_process
k = 0

torch.linalg.slogdet(p0)
torch.linalg.slogdet(p1)
torch.linalg.slogdet(Kinv)
K = self.compute_superposition.observations.time_kernel(k).cov
torch.linalg.det(prec)
K

plt.imshow(self.kernel.cov.cpu().detach())

plt.imshow(self.kernel.inv.cpu().detach())

plt.imshow(c.cpu().detach())



plt.figure(figsize=(5,4))
plt.imshow(p1.log().cpu().detach())
plt.colorbar()
plt.title("Precision update matrix (log)")
plt.tight_layout()
plt.savefig(f"./code/figures/matrices/precision_update_log.pdf")

torch.linalg.eigh(prec)

k = 0
model.variables["smgp_loadings"].processes[1][k, :]
true_model.variables["smgp_loadings"].processes[1][k, :]