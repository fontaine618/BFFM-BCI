import torch

from src.results.mcmc_results import MCMCResults


class MCMCMultipleResults:

	def __init__(self, results: dict[str, MCMCResults]):
		self._vars = None
		self._check_chains(results)
		self.results = results
		self._n = {
			chain_id: result._length
			for chain_id, result in results.items()
		}

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

	def gelman_rubin_diagnostic(self):
		gr = dict()
		N = sum(self._n.values())
		for var in self._vars:
			means = {chain_id: result.posterior_mean[var] for chain_id, result in self.results.items()}
			variances = {chain_id: result.posterior_variance[var] for chain_id, result in self.results.items()}
			mean = sum([
				m * n
				for m, n in zip(means.values(), self._n.values())
			]) / N

