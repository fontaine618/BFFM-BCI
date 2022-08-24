import torch
from .variable import Variable


class Plate:
	r"""
	Abstract container for a plate, which acts as a black-box Variable.

	Built like a dict, but also puts variables as attributes.

	Generate and sample can be overidden to control the behavior (i.e. the ordering).
	The default behavior is that the original ordering in **kwargs is used for generate
	and the reverse for sample.
	"""

	def __init__(self, **kwargs):
		self.__dict__ = dict()
		for k, v in kwargs.items():
			self.__setattr__(k, v)
			self.__dict__[k] = v

	def __getitem__(self, item):
		return self.__dict__[item]

	def __len__(self):
		return len(self.__dict__)

	def __iter__(self):
		return iter(self.__dict__)

	def __contains__(self, item):
		return item in self.__dict__

	def keys(self):
		return self.__dict__.keys()

	def values(self):
		return self.__dict__.values()

	@property
	def parents(self):
		ids_seen = []
		parents = dict()
		for variable in self.values():
			for k, v in variable.parents.items():
				if v.id in ids_seen:
					continue
				ids_seen += v.id
				parents[k] = v
		return parents

	@property
	def children(self):
		ids_seen = []
		parents = dict()
		for variable in self.values():
			for k, v in variable.children.items():
				if v.id in ids_seen:
					continue
				ids_seen += v.id
				parents[k] = v
		return parents

	def add_children(self, **kwargs):
		r"""Default behaviour is to add them to add inner nodes.

		For other behaviors, overwrite this."""
		for v in self.values():
			v.add_children(**kwargs)

	def generate(self):
		for v in self.values():
			v.generate()

	def sample(self, store=True):
		for v in list(self.values())[::-1]:
			v.sample(store=store)
