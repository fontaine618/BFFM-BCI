import torch
import copy
import pandas as pd
from src.models.bffmbci.bffm import BFFModel
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.style.use("seaborn-whitegrid")

torch.set_default_tensor_type(torch.cuda.FloatTensor)

latent_dim = 3
n_iter = 1000
sd = 0.01

dir = "./src/figures/wrong_start_nonnegative/"

orders = {
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

sds = {
	0.: "green",
	0.01: "yellow",
	0.1: "orange",
	1.: "red"
}

n_repetitions = 3

torch.manual_seed(0)
true_model = BFFModel.generate_from_dimensions(
	latent_dim=latent_dim,
	n_channels=7,
	stimulus_to_stimulus_interval=20,
	stimulus_window=50,
	n_stimulus=(3, 3),
	n_sequences=51,
	nonnegative_smgp=True,
)
true_llk = true_model.variables["observations"].log_density

for exname, ex in orders.items():
	filename = ex["filename"]
	order = ex["order"]
	llks = {}

	for sd in sds.keys():
		llks[sd] = {}
		for seed in range(1, n_repetitions+1):
			print(exname, sd, seed)
			llk = []
			model = copy.deepcopy(true_model)
			torch.manual_seed(seed)
			model.jitter_values(sd=sd)

			for i in range(n_iter):
				try:
					model.sample()
				except Exception:
					print("found error, skipped")
				lkki = model.variables["observations"].log_density
				print(i, lkki)
				llk += [lkki]

			llks[sd][seed] = llk

	# plot
	file_path = dir + filename + ".pdf"
	fig, ax = plt.subplots(figsize=(6, 4))
	ax.set_xlabel("Iteration")
	ax.set_ylabel("|nllk - true nllk|/true nllk")
	ax.set_title(exname)
	for sd, color in sds.items():
		llk_sd = llks[sd]
		for seed, llk in llk_sd.items():
			ax.plot(-(true_llk - torch.Tensor(llk)).abs().cpu() / true_llk, color=color, alpha=0.5)
		llk = pd.DataFrame().mean(axis=1)
		ax.plot(llk, label=sd, color=color)
	legend_items = [mpatches.Patch(color=color, label=sd) for sd, color in sds.items()]
	ax.legend(title="Jitter SD", handles=legend_items)
	ax.set_yscale("log")
	ax.set_xscale("log")
	fig.tight_layout()
	fig.savefig(file_path)
	plt.close(fig)

