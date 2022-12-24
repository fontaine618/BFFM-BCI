from typing import Tuple, Union, Any

import torch
import pickle


class MCMCResults:

	_metrics = {
		"loadings": [
			"frobenius",
			"projection_frobenius",
			"columnwise_cosine_similarity",
			"columnwise_2norm",
		],
		"shrinkage_factor": ["2norm"],
		"observation_variance": ["2norm"],
		"smgp_factors.nontarget_process": ["rowwise_2norm"],
		"smgp_factors.target_process": ["rowwise_2norm"],
		"smgp_factors.mixing_process": ["rowwise_2norm", "rowwise_binary_cross_entropy"],
		"smgp_scaling.nontarget_process": ["rowwise_2norm"],
		"smgp_scaling.target_process": ["rowwise_2norm"],
		"smgp_scaling.mixing_process": ["rowwise_2norm", "rowwise_binary_cross_entropy"],
		"observation_log_likelihood": ["difference"],
	}

	_diagnostics = {

	}

	def __init__(
		self,
		variables: dict[str, Union[torch.Tensor, dict[str, torch.Tensor]]],
		llk: list[float] = None,
		warmup: int = 0,
		thin: int = 1
	):
		if llk is not None:
			variables["observation_log_likelihood"] = torch.tensor(llk)
		self.warmup = warmup
		self.thin = thin
		self.variables = self._flatten(variables)
		self._length = self._check_length()
		self.posterior_mean = dict()
		self.posterior_variance = dict()
		self._align_chain()
		self._compute_posterior_summaries()

	# def __getitem__(self, item: Union[str, tuple[str, str]]):
	# 	if isinstance(item, str):
	# 		return self.variables[item]
	# 	elif isinstance(item, tuple):
	# 		if len(item) != 2:
	# 			raise ValueError(f"Expected tuple of length 2, but got {item}")
	# 		return self.variables[item[0]][item[1]]
	# 	else:
	# 		raise TypeError(f"Expected str or tuple, but got {type(item)}")

	def keys(self):
		return self.variables.keys()

	def _flatten(self, v: dict[str, Union[torch.Tensor, float, dict[str, Any]]]):
		out = dict()
		for k, v in v.items():
			if isinstance(v, (torch.Tensor, float)) or v is None:
				out[k] = v
			elif isinstance(v, dict):
				for kk, vv in self._flatten(v).items():
					out[k + "." + kk] = vv
			else:
				raise TypeError(f"Expected torch.Tensor or dict, but got {type(v)}")
		return out

	def _check_length(self):
		n = None
		for k, v in self.variables.items():
			if v is None:
				continue
			nk = v.shape[0]
			if n is None:
				n = nk
			elif n != nk:
				raise ValueError(f"Chain for variable {k} has length {nk}, "
				                 f"but previous chains had length {n}")
		return n

	def _compute_posterior_summaries(self):
		which = torch.arange(self.warmup, self._length, self.thin)
		for k, v in self.variables.items():
			self.posterior_mean[k] = v.index_select(0, which).mean(0)
			self.posterior_variance[k] = v.index_select(0, which).var(0)

	def align_posterior_mean(self, true_values: dict[str, dict[str, Union[torch.Tensor, dict[str, Any]]]]):
		true_values = self._flatten(true_values)
		value = self.posterior_mean["loadings"]
		true_value = true_values["loadings"]
		# fix ordering first
		ip = value.T @ true_value
		_, indices = torch.sort(ip.abs(), dim=1, descending=True)
		order = []
		for k in range(value.shape[1]):
			# take first one not already selected
			for j in range(value.shape[1]):
				if indices[k, j] not in order:
					order.append(indices[k, j].item())
					break
		value = value[:, order]
		self.posterior_mean["loadings"] = \
			self.posterior_mean["loadings"][:, order]
		self.posterior_mean["heterogeneities"] = \
			self.posterior_mean["heterogeneities"][:, order]
		self.posterior_mean["shrinkage_factor"] = \
			self.posterior_mean["shrinkage_factor"][order]
		self.posterior_mean["smgp_factors.target_process"] = \
			self.posterior_mean["smgp_factors.target_process"][order, :]
		self.posterior_mean["smgp_factors.nontarget_process"] = \
			self.posterior_mean["smgp_factors.nontarget_process"][order, :]
		# now fix signs
		for k in range(value.shape[1]):
			if (value[:, k] * true_value[:, k]).sum() < 0:
				self.posterior_mean["loadings"][:, k] *= -1
				self.posterior_mean["smgp_factors.target_process"][k, :] *= -1
				self.posterior_mean["smgp_factors.nontarget_process"][k, :] *= -1

	def metrics(self, true_values: dict[str, torch.Tensor]):
		true_values = self._flatten(true_values)
		metrics = {
			v: {
				m: self._metric(v, m, true_values)
				for m in ms
			}
			for v, ms in self._metrics.items()
		}
		return metrics

	def moving_metrics(self, true_values: dict[str, torch.Tensor], window: int = 100):
		n = self._length
		cuts = torch.arange(0, n, window).tolist()
		n_bins = len(cuts)
		cuts.append(n)
		moving_metrics = {
			f"[{cuts[i]}-{cuts[i+1]})": self[cuts[i]:cuts[i+1]].metrics(true_values)
			for i in range(n_bins)
		}
		meta = {
			f"[{cuts[i]}-{cuts[i+1]})": {"lower": cuts[i], "upper": cuts[i+1]}
			for i in range(n_bins)
		}
		return moving_metrics, meta

	def _metric(self, variable: str, metric: str, true_values: dict[str, torch.Tensor]):
		if variable not in true_values:
			raise ValueError(f"Variable {variable} not found in true values")
		if variable not in self.variables:
			raise ValueError(f"Variable {variable} not found in MCMC results")
		value = self.posterior_mean[variable]
		true_value = true_values[variable]

		if "projection" in metric:
			value = value @ value.T
			true_value = true_value @ true_value.T
		if "cosine_similarity" in metric:
			# this is implicitely columnwise so we return directly
			out = (value * true_value).sum(0)
			out /= value.norm(2, 0) * true_value.norm(2, 0)
			return out.tolist()
		elif "2norm" in metric or "frobenius" in metric:
			out = (value - true_value).pow(2.)
		elif "binary_cross_entropy" in metric:
			out = -(true_value * value.log() + (1 - true_value) * (1 - value).log())
		elif "difference" in metric:
			out = value - true_value
		else:
			raise ValueError(f"Unknown metric {metric}")
		if "columnwise" in metric:
			return out.mean(0).tolist()
		elif "rowwise" in metric:
			return out.mean(1).tolist()
		else:
			return out.mean().item()

	def _align_chain(self):
		# we align on the last iteration
		last_loadings = self.variables["loadings"][-1, :, :]
		swaps = []
		flips = []
		for i in range(self._length):
			loadings = self.variables["loadings"][i, :, :]
			# fix ordering first
			ip = loadings.T @ last_loadings
			_, indices = torch.sort(ip.abs(), dim=1, descending=True)
			order = []
			for k in range(loadings.shape[1]):
				# take first one not already selected
				for j in range(loadings.shape[1]):
					if indices[k, j] not in order:
						order.append(indices[k, j].item())
						break
			swaps.append(order)
			loadings = loadings[:, order]
			self.variables["loadings"][i, :, :] = \
				self.variables["loadings"][i, :, order]
			self.variables["heterogeneities"][i, :, :] = \
				self.variables["heterogeneities"][i, :, order]
			self.variables["shrinkage_factor"][i, :] = \
				self.variables["shrinkage_factor"][i, order]
			self.variables["smgp_factors.target_process"][i, :, :] = \
				self.variables["smgp_factors.target_process"][i, order, :]
			self.variables["smgp_factors.nontarget_process"][i, :, :] = \
				self.variables["smgp_factors.nontarget_process"][i, order, :]
			# fix direction
			flip = [0 for _ in range(loadings.shape[1])]
			for k in range(loadings.shape[1]):
				if (loadings[:, k] * last_loadings[:, k]).sum() < 0:
					flip[k] = 1
					self.variables["loadings"][i, :, k] *= -1
					self.variables["smgp_factors.target_process"][i, k, :] *= -1
					self.variables["smgp_factors.nontarget_process"][i, k, :] *= -1
			flips.append(flip)

	# save and load could be done only using variables to have better
	def save(self, filename: str):
		with open(filename, "wb") as f:
			pickle.dump(self, f)

	@classmethod
	def load(cls, filename: str):
		with open(filename, "rb") as f:
			return pickle.load(f)

	def __len__(self):
		return self._length

	def __getitem__(self, item):
		variables = dict()
		if isinstance(item, int):
			item = [item]
		elif isinstance(item, slice):
			item = list(range(*item.indices(len(self))))
		item = torch.tensor(item)
		for k, v in self.variables.items():
			variables[k] = v.index_select(0, item)
		return MCMCResults(variables)
