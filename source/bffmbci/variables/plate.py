import torch


class Plate:
	r"""
	Abstract container for a plate, which acts as a black-box Variable.

	Built like a dict, but also puts variables as attributes.

	Generate and sample can be overidden to control the behavior (i.e. the ordering).
	The default behavior is that the original ordering in **kwargs is used for generate
	and the reverse for sample.
	"""

	def __init__(self, **kwargs):
		self.variables = dict()
		for k, v in kwargs.items():
			self.__setattr__(k, v)
			self.variables[k] = v
		self.true_values = None

	def filter(self, ids: torch.Tensor, dim: int = 0):
		r"""Filter out ids from all variables in the plate."""
		for v in self.values():
			v.filter(ids, dim=dim)

	def __getitem__(self, item):
		return self.variables[item]

	def __len__(self):
		return len(self.variables)

	def __iter__(self):
		return iter(self.variables)

	def __contains__(self, item):
		return item in self.variables

	def keys(self):
		return self.variables.keys()

	def values(self):
		return self.variables.values()

	def items(self):
		return self.variables.items()

	@property
	def parents(self):
		ids_seen = []
		parents = dict()
		for n, variable in self.items():
			for k, v in variable.parents.items():
				if v.id in ids_seen:
					continue
				ids_seen += [v.id]
				parents[n + "." + k] = v
		return parents

	@property
	def children(self):
		# ids_seen = []
		children = dict()
		for n, variable in self.items():
			for k, v in variable.children.items():
				# if v.id in ids_seen:
				# 	continue
				# ids_seen += [v.id]
				children[n + "." + k] = v
		return children

	def add_children(self, **kwargs):
		r"""Default behaviour is to add them to all inner nodes.

		For other behaviors, overwrite this."""
		for v in self.values():
			v.add_children(**kwargs)

	def generate(self):
		for v in self.values():
			v.generate()

	def sample(self, store=True):
		for v in list(self.values())[::-1]:
			v.sample(store=store)

	def __repr__(self):
		return f"{self.__class__.__name__}()"

	def __str__(self):
		out = repr(self) + "\n"
		out += "- Parents:\n"
		for n, c in self.parents.items():
			out += f"    {n}: {repr(c)}\n"
		out += "- Variables:\n"
		for n, c in self.items():
			out += f"    {n}: {repr(c)}\n"
		out += "- Children:\n"
		for n, c in self.children.items():
			out += f"    {n}: {repr(c)}\n"
		return out[:-2]

	def jitter(self, sd: float = 0.1):
		for v in self.values():
			v.jitter(sd=sd)

	def chain(self, **kwargs):
		return {
			k: v.chain(**kwargs)
			for k, v in self.items()
		}

	def clear_history(self):
		for v in self.values():
			v.clear_history()

	@property
	def data(self):
		return {k: v.data for k, v in self.variables.items()}

	@data.setter
	def data(self, value: dict[str: torch.Tensor]):
		for k, v in value.items():
			self.variables[k].data = v

	def store_new_value(self, store=False):
		for v in self.values():
			v.store_new_value(store)

	@property
	def log_density(self):
		return {
			k: v.log_density
			for k, v in self.items()
		}

	@property
	def _log_density_history(self):
		return {
			k: v.log_density_history
			for k, v in self.items()
		}

	def log_density_history(self, start=0, end=None, thin=1):
		return {
			k: v.log_density_history(start=start, end=end, thin=thin)
			for k, v in self.items()
		}

	def store_log_density(self):
		for v in self.values():
			v.store_log_density()