import torch
import copy
import pandas as pd
from src.models.bffmbci.bffm import BFFModel
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
plt.style.use("seaborn-whitegrid")

torch.set_default_tensor_type(torch.cuda.FloatTensor)

latent_dim = 3
n_iter = 2000
n_repetitions = 3

dir = "./src/figures/initialization/"

orders = {
	"All": {
		"filename": "all2000",
		"order": None
	},
}

sds = {
	"init": "black",
	0.: "green",
	0.01: "yellow",
	0.1: "orange",
	1.: "red"
}


torch.manual_seed(0)
true_model = BFFModel.generate_from_dimensions(
	latent_dim=latent_dim,
	n_channels=7,
	stimulus_to_stimulus_interval=10,
	stimulus_window=55,
	n_stimulus=(3, 3),
	n_sequences=51,
	nonnegative_smgp=True,
)
true_values = true_model.current_values()
true_llk = true_model.variables["observations"].log_density

for exname, ex in orders.items():
	filename = ex["filename"]
	order = ex["order"]
	llks = {}
	metrics = {}

	for sd in sds.keys():
		llks[sd] = {}
		metrics[sd] = {}

		for seed in range(1, n_repetitions+1):
			print(exname, sd, seed)
			llk = []
			metric = []
			model = copy.deepcopy(true_model)
			torch.manual_seed(seed)
			if isinstance(sd, float):
				model.jitter_values(sd=sd)
			else:
				status = False
				while not status:
					try:
						model.initialize_chain()
						status = True
					except Exception as e:
						print(e)

			for i in range(n_iter):
				model.sample()
				lkki = model.variables["observations"].log_density
				print(i, lkki)
				llk += [lkki]

			llks[sd][seed] = llk
			metrics[sd][seed] = metric

	# plot
	file_path = dir + filename + "_llk.pdf"
	fig, ax = plt.subplots(figsize=(6, 4))
	ax.set_xlabel("Iteration")
	ax.set_ylabel("nllk / true nllk")
	ax.set_title(exname)
	for sd, color in sds.items():
		if sd not in llks.keys():
			continue
		llk_sd = llks[sd]
		for seed, llk in llk_sd.items():
			ax.plot(torch.Tensor(llk).cpu() / true_llk, color=color, alpha=0.5)
		llk = pd.DataFrame().mean(axis=1)
		ax.plot(llk, label=sd, color=color)
	legend_items = [mpatches.Patch(color=color, label=sd) for sd, color in sds.items()]
	ax.legend(title="Jitter SD", handles=legend_items)
	ax.set_yscale("log")
	ax.set_xscale("log")
	fig.tight_layout()
	fig.savefig(file_path)
	plt.close(fig)


