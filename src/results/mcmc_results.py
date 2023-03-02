from typing import Tuple, Union, Any
import torch
import numpy as np
import arviz as az
import pickle
import copy
from ..bffmbci import BFFMPredict
import warnings


suffix = "[MCMCResults] "
padding = " " * len(suffix)


def _flatten_dict(d: dict[str: Any]) -> dict[str: Any]:
    out = dict()
    for key, value in d.items():
        if isinstance(value, dict):
            for subkey, subvalue in _flatten_dict(value).items():
                out[f"{key}.{subkey}"] = torch.Tensor(subvalue)
        else:
            out[key] = value
    return out


class MCMCResults:
    """Results of an MCMC run.

    Indexing:
    - first dimension is the chain index
    - second dimension is the sample index
    - all other dimensions are the dimensions of the variable/llk

    """

    _sufficient_for_prediction = [
        "loadings",
        "observation_variance",
        "smgp_factors.mixing_process",
        "smgp_factors.nontarget_process",
        "smgp_factors.target_process",
        "smgp_scaling.mixing_process",
        "smgp_scaling.nontarget_process",
        "smgp_scaling.target_process",
    ]

    def __init__(
        self,
        prior: dict[str: Any],
        dimensions: dict[str: int],
        chains: dict[str: torch.Tensor],
        warmup: int = 0,
        thin: int = 1,
        warmed_up: int = 0,
        thinned: int = 1,
        **kwargs
    ):
        """This assumes data is already in the correct format.
        Most likely, the other class methods should be used to create an instance of this class."""
        self.prior = prior
        self.dimensions = dimensions
        self.chains = chains
        self.warmup = warmed_up
        self.thin = thinned
        self.drop_warmup(warmup)
        self.thin_chains(thin)
        self._validate()
        _add_transformed_variables(self.chains)
        print(f"{suffix}Created {repr(self)}.")
        print(padding + f"- thinning: {self.thin}")
        print(padding + f"- warmup: {self.warmup}")

    @classmethod
    def from_files(
            cls,
            files: list[list[str]] | list[str],
            **kwargs
    ) -> "MCMCResults":
        """Each element is understood to be a chain. Then, each element can be multiple
        files for a single chain"""
        return cls.concat([cls.from_files_single_chain(f, **kwargs) for f in files], 0)

    @classmethod
    def from_files_single_chain(
            cls,
            files: list[str] | str,
            **kwargs
    ) -> "MCMCResults":
        """Each element is understood to be part of a single chain."""
        if isinstance(files, str):
            files = [files]
        return cls.concat([cls._read_file_single_chain(f, **kwargs) for f in files], 1)

    @classmethod
    def _read_file_single_chain(cls, file: str, **kwargs) -> "MCMCResults":
        """Each element is understood to be part of a single chain."""
        print(suffix + f"Reading file")
        print(padding + f"'{file}'")
        with open(file, "rb") as f:
            result = pickle.load(f)
        print(padding + "done!")
        return cls.single_chain(**result, **kwargs)

    @classmethod
    def single_chain(cls, prior, dimensions, chain, log_likelihood=None, **kwargs) -> "MCMCResults":
        """We assume the input does not have a first dimension for chain id."""
        # we create the new chain id dimension
        chain["log_likelihood"] = log_likelihood
        chain = _flatten_dict(chain)
        for v in chain.values():
            v.unsqueeze_(0)
        return cls(
            prior=prior,
            dimensions=dimensions,
            chains=chain,
            **kwargs
        )

    @classmethod
    def concat(cls, results: list["MCMCResults"], dim: int) -> "MCMCResults":
        """dim=0 means multiple chains, dim=1 means concatenate"""
        out = results[0]
        for r in results[1:]:
            out = out.append(r, dim)
        return out

    def append(self, other: "MCMCResults", dim: int) -> "MCMCResults":
        """Append other to self. Assumes that the other has the same prior, dimensions, etc."""
        repr_pre = repr(self)
        self._check_parameters(other)
        for k, v in self.chains.items():
            self.chains[k] = torch.cat([v, other.chains[k]], dim=dim)
        print(f"{suffix}Created {repr(self)} \n"
              f"{padding}by appending {repr(other)} \n"
              f"{padding}into {repr_pre}.")
        return self

    def _validate(self):
        dims = {k: v.shape[0:2] for k, v in self.chains.items()}
        dims0 = next(iter(dims.values()))
        if not all([d == dims0 for d in dims.values()]):
            warnings.warn(f"Variables have different first two dimensions. \n"
                             f"Expected {dims0}, got {dims}.\n"
                          f"The chain was patched.", UserWarning)
            max_length = max([d[1] for d in dims.values()])
            for k, dim in dims.items():
                if dim[1] < max_length:
                    self.chains[k] = torch.cat([
                        self.chains[k], self.chains[k][:, (dim[1]-max_length):, ...]
                    ], 1)

    def _check_parameters(self, other: "MCMCResults"):
        """Check that the other has the same prior, dimensions, etc."""
        if self.warmup != other.warmup:
            raise ValueError("Warmup is different between two MCMCResults")
        if self.thin != other.thin:
            raise ValueError("Thinning is different between two MCMCResults")
        if self.prior != other.prior:
            raise ValueError("Prior is different between two MCMCResults")
        if self.dimensions != other.dimensions:
            raise ValueError("Dimensions are different between two MCMCResults")
        if self.chains.keys() != other.chains.keys():
            raise ValueError("Chains keys are different between two MCMCResults")
        for k, v in self.chains.items():
            # need same dimensions except the first two
            if v.shape[2:] != other.chains[k].shape[2:]:
                raise ValueError(f"Variable {k} has different dimensions between two MCMCResults")

    def drop_warmup(self, warmup: int = 0):
        self.warmup += warmup
        for k, v in self.chains.items():
            self.chains[k] = v[:, warmup:, ...]

    def thin_chains(self, thin: int = 1):
        self.thin *= thin
        for k, v in self.chains.items():
            self.chains[k] = v[:, ::thin, ...]

    def align(self):
        #TODO
        pass

    @property
    def n_chains(self):
        return next(iter(self.chains.values())).shape[0]

    @property
    def n_samples(self):
        return next(iter(self.chains.values())).shape[1]

    def __repr__(self):
        return f"{self.__class__.__name__}({self.n_chains} chains, {self.n_samples} samples)"
        pass

    def to_arviz(self):
        return az.from_dict(posterior={k: v.cpu() for k, v in self.chains.items()})

    def to_predict(self, thin: int | None = None, n_samples: int | None = 1000):
        """Create a Predict object from the MCMCResults."""
        if thin is None:
            thin = int(np.ceil(self.n_samples * self.n_chains / n_samples))
        variables = {
            k: self.chains[k].flatten(0, 1)[::thin, ...]
            for k in self._sufficient_for_prediction
        }
        return BFFMPredict(variables=variables, dimensions=self.dimensions, prior=self.prior)


def _add_transformed_variables(chains):
    if "loadings.inner_products" not in chains and \
            "loadings" in chains:
        L = chains["loadings"]  # (..., E, K)
        chains["loadings.inner_products"] = L @ L.transpose(-1, -2) # (..., E, E)
    if "loadings.rank_one" not in chains and \
            "loadings" in chains:
        L = chains["loadings"].transpose(-1, -2)  # (..., K, E)
        chains["loadings.rank_one"] = L.unsqueeze(-1) @ L.unsqueeze(-2)  # (..., K, E, E)
    if "loadings.projection" not in chains and \
            "loadings" in chains:
        L = chains["loadings"]  # (..., E, K)
        LtL = L.transpose(-1, -2) @ L  # (..., K, K)
        LtLinv = torch.linalg.inv(LtL)  # (..., K, K)
        chains["loadings.projection"] = L @ LtLinv @ L.transpose(-1, -2)  # (..., E, E)
    if "loadings.norm_one" not in chains and \
            "loadings" in chains:
        L = chains["loadings"]
        Lnorm = torch.linalg.norm(L, dim=-2, keepdim=True)
        chains["loadings.norm_one"] = L / Lnorm
    if "loadings.times_shrinkage" not in chains\
            and "loadings" in chains and "shrinkage_factor" in chains:
        L = chains["loadings"]
        s = chains["shrinkage_factor"].unsqueeze(-2)
        chains["loadings.times_shrinkage"] = L / s.sqrt()
    if "smgp_factors.target_signal" not in chains and \
            "smgp_factors.nontarget_process" in chains and \
            "smgp_factors.target_process" in chains and \
            "smgp_factors.mixing_process" in chains:
        chains["smgp_factors.target_signal"] = \
            (1 - chains["smgp_factors.mixing_process"]) * \
            chains["smgp_factors.nontarget_process"] + \
            chains["smgp_factors.mixing_process"] * \
            chains["smgp_factors.target_process"]
    if "smgp_factors.difference_process" not in chains and \
            "smgp_factors.nontarget_process" in chains and \
            "smgp_factors.target_process" in chains and \
            "smgp_factors.mixing_process" in chains:
        chains["smgp_factors.difference_process"] = \
            chains["smgp_factors.target_signal"] - \
            chains["smgp_factors.nontarget_process"]
    if "smgp_scaling.target_signal" not in chains and \
            "smgp_scaling.nontarget_process" in chains and \
            "smgp_scaling.target_process" in chains and \
            "smgp_scaling.mixing_process" in chains:
        chains["smgp_scaling.target_signal"] = \
            (1 - chains["smgp_scaling.mixing_process"]) * \
            chains["smgp_scaling.nontarget_process"] + \
            chains["smgp_scaling.mixing_process"] * \
            chains["smgp_scaling.target_process"]
    if "smgp_scaling.difference_process" not in chains and \
            "smgp_scaling.nontarget_process" in chains and \
            "smgp_scaling.target_process" in chains and \
            "smgp_scaling.mixing_process" in chains:
        chains["smgp_scaling.difference_process"] = \
            chains["smgp_scaling.target_signal"] - \
            chains["smgp_scaling.nontarget_process"]
