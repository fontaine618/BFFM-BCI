import torch
from src.models.bffmbci.bffm import BFFModel
import matplotlib.pyplot as plt
plt.style.use("seaborn-whitegrid")

torch.set_default_tensor_type(torch.cuda.FloatTensor)

n_iter = 100
latent_dim = 3
k = 2
xs = range(0, n_iter+1)
dir = f"./src/figures/nonnegative/"

order = None

torch.manual_seed(0)
self = BFFModel.generate_from_dimensions(
	latent_dim=latent_dim,
	observation_variance=(3., 1.),

	kernel_gp=(0.99, 0.5, 0.5),
	kernel_tgp=(0.99, 0.5, 0.5),
	kernel_factor=(0.99, 0.01, 0.5),

	nonnegative_smgp=True
)

X = self.variables["smgp_loadings"].nontarget_process.data
plt.plot(X.cpu().T)
plt.show()

llk = [self.variables["observations"].log_density]
for i in range(n_iter):
	self.sample(order)
	llki = self.variables["observations"].log_density
	print(i + 1, llki)
	llk += [llki]


# Loadings
plt.figure(figsize=(6, 4))
plt.plot(
	xs,
	self.variables["loadings"].chain[:, :, k].cpu().numpy()
)
plt.title(f"Loadings (k={k})")
plt.tight_layout()
plt.savefig(dir + f"loadings_{k}.pdf")
plt.cla()


# Nontarget loadings process
plt.figure(figsize=(6, 4))
plt.plot(
	range(self.variables["smgp_loadings"]["nontarget_process"].shape[1]),
	self.variables["smgp_loadings"]["nontarget_process"].chain[:, k, :].cpu().numpy().T
)
plt.title(f"Loadings nontarget processes (k={k})")
plt.tight_layout()
plt.savefig(dir + f"smgp_loadings_nontarget_{k}.pdf")
plt.cla()


# Target loadings process
plt.figure(figsize=(6, 4))
plt.plot(
	range(self.variables["smgp_loadings"]["target_process"].shape[1]),
	self.variables["smgp_loadings"]["target_process"].chain[:, k, :].cpu().numpy().T
)
plt.title(f"Loadings target processes (k={k})")
plt.tight_layout()
plt.savefig(dir + f"smgp_loadings_target_{k}.pdf")
plt.cla()


# Target mixing process
plt.figure(figsize=(6, 4))
plt.plot(
	range(self.variables["smgp_loadings"]["mixing_process"].shape[1]),
	self.variables["smgp_loadings"]["mixing_process"].chain[:, k, :].cpu().numpy().T
)
plt.title(f"Loadings mixing processes (k={k})")
plt.tight_layout()
plt.savefig(dir + f"smgp_loadings_mixing_{k}.pdf")
plt.cla()


# Nontarget factor process
plt.figure(figsize=(6, 4))
plt.plot(
	range(self.variables["smgp_factors"]["nontarget_process"].shape[1]),
	self.variables["smgp_factors"]["nontarget_process"].chain[:, k, :].cpu().numpy().T
)
plt.title(f"Factors nontarget processes (k={k})")
plt.tight_layout()
plt.savefig(dir + f"smgp_factors_nontarget_{k}.pdf")
plt.cla()


# Target factor process
plt.figure(figsize=(6, 4))
plt.plot(
	range(self.variables["smgp_factors"]["target_process"].shape[1]),
	self.variables["smgp_factors"]["target_process"].chain[:, k, :].cpu().numpy().T
)
plt.title(f"Factors target processes (k={k})")
plt.tight_layout()
plt.savefig(dir + f"smgp_factors_target_{k}.pdf")
plt.cla()


# Target factor process
plt.figure(figsize=(6, 4))
plt.plot(
	range(self.variables["smgp_factors"]["mixing_process"].shape[1]),
	self.variables["smgp_factors"]["mixing_process"].chain[:, k, :].cpu().numpy().T
)
plt.title(f"Factors mixing processes (k={k})")
plt.tight_layout()
plt.savefig(dir + f"smgp_factors_mixing_{k}.pdf")
plt.cla()