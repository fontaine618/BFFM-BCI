from . import ObservedVariable
from .plate import Plate


class SequenceData(Plate):
	"""
	A container for sequence data: ordering and target/nontarget indicator.

	Assumes the following format:
	- order: [n_sequences, n_stimuli] in 0:n_stimuli
	- target: [n_sequences, n_stimuli] in {0, 1}
	"""

	_store = False
	_stochastic = False

	def __init__(self, order, target):
		self.order: ObservedVariable = None
		self.target: ObservedVariable = None
		super().__init__(
			order=ObservedVariable(order),
			target=ObservedVariable(target)
		)