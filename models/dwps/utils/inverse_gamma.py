from abc import ABC

from torch.distributions import constraints
from torch.distributions.transformed_distribution import TransformedDistribution
from torch.distributions.gamma import Gamma
from torch.distributions.transforms import PowerTransform


class InverseGamma(TransformedDistribution, ABC):
    r"""
    Creates an inverse-gamma distribution parameterized by
    `concentration` and `rate`.
        X ~ Gamma(concentration, rate)
        Y = 1/X ~ InverseGamma(concentration, rate)
    :param torch.Tensor concentration: the concentration parameter (i.e. alpha).
    :param torch.Tensor rate: the rate parameter (i.e. beta).

    Taken from pyro.distributions.inverse_gamma.py
    """
    arg_constraints = {
        "concentration": constraints.positive,
        "rate": constraints.positive,
    }
    support = constraints.positive
    has_rsample = True

    def __init__(self, concentration: float, rate: float, validate_args=None):
        base_dist = Gamma(concentration, rate)
        super().__init__(
            base_dist,
            PowerTransform(-base_dist.rate.new_ones(())),
            validate_args=validate_args,
        )

    def expand(self, batch_shape, _instance=None):
        new = self._get_checked_instance(InverseGamma, _instance)
        return super().expand(batch_shape, _instance=new)

    @property
    def concentration(self):
        return self.base_dist.concentration

    @property
    def rate(self):
        return self.base_dist.rate