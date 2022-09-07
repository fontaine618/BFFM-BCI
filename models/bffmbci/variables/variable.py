import torch
import torch.nn
import itertools


class Variable:
	r"""
	A template for Bayes net variable.

	[Parents] => Factor => Variable => [Children]

	Some notes:
	- priors should be implemented in generate and sample
	- generate should only be a function of parents and therefore should not require children to be set
	- sample should depend on both parents and children
	"""

	_observed = False
	_stochastic = True
	_dim_names = []
	new_id = itertools.count()

	def __init__(self, dim=None, store=True, init=None, **kwargs):
		self.id = next(Variable.new_id)
		self._dim = torch.Size(dim)
		self._store = store
		if store:
			self._history = torch.zeros((0, *dim))
		self._value = None
		self._init(dim, init)
		self.parents = dict()
		self.children = dict()

	@property
	def data(self):
		return self._value

	@property
	def shape(self):
		return self._dim

	@property
	def chain(self):
		return self._history

	def _set_value(self, value, store=True):
		if value.shape != self._dim:

			raise ValueError(f"Trying to set a Variable value to an incorrect size:\n"
			                 f"got {tuple(value.shape)}, expected: {tuple(self._dim)}")
		self._value = value
		self.store_new_value(store)

	def _init(self, dim, init=None):
		if self._value is not None:
			raise RuntimeError("Trying to initialize a Variable that was already set.")
		if init is None:
			self.generate()
		else:
			if callable(init):
				self._set_value(init(dim))
			else:
				self._set_value(init)

	def store_new_value(self, store=False):
		if store and self._store:
			self._history = torch.cat([self._history, self.data.unsqueeze(0)])

	def generate(self, **kwargs):
		pass

	def sample(self, store=True):
		pass

	def add_children(self, **kwargs):
		for k, v in kwargs.items():
			self.__setattr__(k, v)
			self.children[k] = v

	def generate_recursively(self):
		r"""
		Generate from a hierarchical prior.
		"""
		for parent in self.parents.values():
			parent.generate_recursively()
		self.generate()

	def sample_recursively(self):
		r"""
		Sample in a hierarchical prior with a single call.

		It samples from the bottom node and works its way up to propagate the information.
		"""
		self.sample()
		for parent in self.parents.values():
			parent.sample_recursively()

	def __repr__(self):
		if len(self._dim_names) > 0:
			dim_str = ', '.join([f"{n}={x}" for n, x in zip(self._dim_names, self.shape)])
		else:
			dim_str = ', '.join([f"{x}" for x in self.shape])
		return f"[{self.id}] {self.__class__.__name__}({dim_str})"

	def __str__(self):
		out = repr(self) + "\n"
		out += "- Parents:\n"
		for n, c in self.parents.items():
			out += f"    {n}: {repr(c)}\n"
		out += "- Children:\n"
		for n, c in self.children.items():
			out += f"    {n}: {repr(c)}\n"
		return out[:-2]


class ObservedVariable(Variable):
	r"""
	Some notes:
	- generate and sample should have no effects for these variables so keep default
	- we might want to define generate if we want to sample from the model, though
	"""

	_observed = True

	def __init__(self, value=None, dim=None):
		if value is not None:
			super().__init__(value.size(), store=False, init=value)
		elif dim is not None:
			super().__init__(dim, store=False, init=None)
		else:
			raise ValueError("Cannot instantiate an ObservedVariable without its value or its dimension")

	def _set_value(self, value, store=False):
		if self._value is not None:
			raise RuntimeError("Trying to set the value of an observed variable.")
		super()._set_value(value, store)
