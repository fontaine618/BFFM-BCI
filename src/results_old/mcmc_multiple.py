import torch
import arviz as az
import xarray as xr
import numpy as np

from .mcmc_results import MCMCResults


class MCMCMultipleResults:

	def __init__(self, chains: dict[str, MCMCResults]):
		self._vars: list[str] | None = None
		self._check_chains(chains)
		self._n = {
			chain_id: result._length
			for chain_id, result in chains.items()
		}
		self.chains = chains
		# concatenate chains
		chains_cat = {
			var: torch.cat([
				chains[chain_id].variables[var].unsqueeze(0)
				for chain_id in chains.keys()
			], dim=0).detach().cpu().numpy()
			for var in self._vars
		}
		chains_cat = _compute_transformed_variables(chains_cat)
		self.chains_cat = az.dict_to_dataset(chains_cat)

	def _check_chains(self, results: dict[str, MCMCResults]):
		vars = None
		dims = None
		for chain_id, result in results.items():
			if vars is None:
				vars = set(result.variables.keys())
			if dims is None:
				dims = {
					k: v.shape[1:]
					for k, v in result.variables.items()
				}
			if set(result.variables.keys()) != vars:
				raise ValueError(f"Chain {chain_id} has different variables than the first chain")
			for k, v in result.variables.items():
				if v.shape[1:] != dims[k]:
					raise ValueError(f"Chain {chain_id} has different shape for variable {k} "
					                 f"than the first chain")
		self._vars = vars


def _compute_transformed_variables(chains_cat: dict[str, torch.Tensor]):
	# loading inner products
	loadings = chains_cat["loadings"]
	chains_cat["loadings_inner_product"] = \
		(np.expand_dims(loadings, -2) * np.expand_dims(loadings, -3)).sum(-1)
	# target signals
	chains_cat["smgp_factors.target_signal"] = \
		(1 - chains_cat["smgp_factors.mixing_process"]) * \
		chains_cat["smgp_factors.nontarget_process"] + \
		chains_cat["smgp_factors.mixing_process"] * \
		chains_cat["smgp_factors.target_process"]
	chains_cat["smgp_scaling.target_signal"] = \
		(1 - chains_cat["smgp_scaling.mixing_process"]) * \
		chains_cat["smgp_scaling.nontarget_process"] + \
		chains_cat["smgp_scaling.mixing_process"] * \
		chains_cat["smgp_scaling.target_process"]
	return chains_cat