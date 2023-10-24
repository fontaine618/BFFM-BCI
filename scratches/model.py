import torch
from source.bffmbci.bffm import BFFModel
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
plt.style.use("seaborn-whitegrid")

torch.set_default_tensor_type(torch.cuda.FloatTensor)



from source.bffmbci.variables.loadings import SparseHetereogeneities, Loadings, ShrinkageFactor

self = ShrinkageFactor(6, prior_parameters=(1., 2.))

for _ in range(20):
	self.generate()
	print(self.data)



n_iter = 100

def plot_llk(self, order, name, filename):
	llk = [self.variables["observations"].log_density]
	for i in range(n_iter):
		self.sample(order)
		llki = self.variables["observations"].log_density
		print(i+1, llki)
		llk += [llki]

	plt.figure(figsize=(6, 4))
	plt.plot(range(0, n_iter+1), llk)
	plt.gca().yaxis.set_major_formatter(FormatStrFormatter('%.0f'))
	plt.title(f"{name}: data log-likelihood")
	plt.tight_layout()
	plt.savefig(f"./code/figures/partial_updates3/llk_{filename}.pdf")


experiments = {
	"Factor mixing processes": {
		"filename": "factor_mixing_processes",
		"order": [
			"factor_processes",
			"mean_factor_processes",
			"smgp_factors.mixing_process"
		]
	},
	"Factor non-mixing processes": {
		"filename": "factor_nonmixing_processes",
		"order": [
			"factor_processes",
			"mean_factor_processes",
			"smgp_factors.target_process",
			"smgp_factors.nontarget_process"
		]
	},
	"Loadings mixing processes": {
		"filename": "loading_mixing_processes",
		"order": [
			"loading_processes",
			"smgp_loadings.mixing_process"
		]
	},
	"Loadings non-mixing processes": {
		"filename": "loading_nonmixing_processes",
		"order": [
			"loading_processes",
			"smgp_loadings.target_process",
			"smgp_loadings.nontarget_process"
		]
	},
	"Observation variance": {
		"filename": "observation_variance",
		"order": [
			"observation_variance"
		]
	},
	"Loadings": {
		"filename": "loadings",
		"order": [
			"loadings",
			"shrinkage_factor",
			"heterogeneities",
		]
	},
	"Loading processes": {
		"filename": "loading_processes",
		"order": [
			"loading_processes",
			"smgp_loadings",
		]
	},
	"Factor processes": {
		"filename": "factor_processes",
		"order": [
			"factor_processes",
			"mean_factor_processes",
			"smgp_factors",
		]
	},
	"Factor processes and loadings": {
		"filename": "factor_processes_loadings",
		"order": [
			"factor_processes",
			"mean_factor_processes",
			"smgp_factors",
			"loadings",
			"shrinkage_factor",
			"heterogeneities",
		]
	},
	"Loadings processes and loadings": {
		"filename": "loading_processes_loadings",
		"order": [
			"loading_processes",
			"smgp_loadings",
			"loadings",
			"shrinkage_factor",
			"heterogeneities",
		]
	},
	"All": {
		"filename": "all",
		"order": None
	},
	"Factor processes only": {
		"filename": "factor_processes_only",
		"order": [
			"factor_processes",
		]
	},
}

for name, ex in experiments.items():
	print(name)
	torch.manual_seed(0)
	self = BFFModel.generate_from_dimensions(
		latent_dim=3,
		observation_variance=(3., 1.)
	)
	filename = ex["filename"]
	order = ex["order"]
	plot_llk(self, order, name, filename)



import torch
from source.bffmbci.variables.loadings import SparseHetereogeneities, Loadings, ShrinkageFactor

dim = (6, 4)

heterogeneities = SparseHetereogeneities(dim)
heterogeneities.generate()
heterogeneities.data

shrinking = ShrinkageFactor(dim[1], prior_parameters=(2., 3.))
shrinking.generate()
shrinking.data

loadings = Loadings(heterogeneities, shrinking)
loadings.generate()
loadings.data