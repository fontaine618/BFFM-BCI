import torch
from models.bffmbci.bffm import BFFModel
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
plt.style.use("seaborn-whitegrid")

torch.set_default_tensor_type(torch.cuda.FloatTensor)

n_iter = 1000

def plot_llk(order, name, filename):
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
	plt.savefig(f"./code/figures/sampling/{filename}_llk.pdf")


experiments = {
	# "Observation variance": {
	# 	"filename": "observation_variance",
	# 	"order": [
	# 		"observation_variance"
	# 	]
	# },
	"Loadings": {
		"filename": "loadings",
		"order": [
			"loadings",
			"shrinkage_factor",
			"heterogeneities",
		]
	},
	# "Loading processes": {
	# 	"filename": "processes",
	# 	"order": [
	# 		"loading_processes",
	# 		"smgp_loadings",
	# 	]
	# },
	# "Factor processes": {
	# 	"filename": "factor_processes",
	# 	"order": [
	# 		"factor_processes",
	# 		"mean_factor_processes",
	# 		"smgp_factors",
	# 	]
	# },
	# "Factor processes and loadings": {
	# 	"filename": "factor_processes_loadings",
	# 	"order": [
	# 		"factor_processes",
	# 		"mean_factor_processes",
	# 		"smgp_factors",
	# 		"loadings",
	# 		"shrinkage_factor",
	# 		"heterogeneities",
	# 	]
	# },
	# "Loadings processes and loadings": {
	# 	"filename": "loading_processes_loadings",
	# 	"order": [
	# 		"loading_processes",
	# 		"smgp_loadings",
	# 		"loadings",
	# 		"shrinkage_factor",
	# 		"heterogeneities",
	# 	]
	# },
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
	self = BFFModel.generate_from_dimensions()
	filename = ex["filename"]
	order = ex["order"]
	plot_llk(order, name, filename)
