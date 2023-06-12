from typing import Any
import torch
import numpy as np
import arviz as az
import pickle
from . import BFFMPredict
from results.metrics import metrics
import warnings
from collections import defaultdict


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


class BFFMResults:
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
        settings: dict[str: Any] | None = None,
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
        self.settings = settings if settings is not None else dict()
        self.chains = chains
        self.warmup = warmed_up
        self.thin = thinned
        self.drop_warmup(warmup)
        self.thin_chains(thin)
        self._validate()
        self._aligned = False
        print(f"{suffix}Created {repr(self)}.")
        print(padding + f"- thinning: {self.thin}")
        print(padding + f"- warmup: {self.warmup}")

    @classmethod
    def from_files(
            cls,
            files: list[list[str]] | list[str],
            **kwargs
    ) -> "BFFMResults":
        """Each element is understood to be a chain. Then, each element can be multiple
        files for a single chain"""
        return cls.concat([cls.from_files_single_chain(f, **kwargs) for f in files], 0)

    @classmethod
    def from_files_single_chain(
            cls,
            files: list[str] | str,
            **kwargs
    ) -> "BFFMResults":
        """Each element is understood to be part of a single chain."""
        if isinstance(files, str):
            files = [files]
        return cls.concat([cls._read_file_single_chain(f, **kwargs) for f in files], 1)

    @classmethod
    def _read_file_single_chain(cls, file: str, **kwargs) -> "BFFMResults":
        """Each element is understood to be part of a single chain."""
        print(suffix + "Reading file")
        print(padding + f"'{file}'")
        with open(file, "rb") as f:
            result = pickle.load(f)
        print(padding + "done!")
        return cls.single_chain(**result, **kwargs)

    @classmethod
    def single_chain(cls, prior, dimensions, chain, log_likelihood=None, **kwargs) -> "BFFMResults":
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
    def concat(cls, results: list["BFFMResults"], dim: int) -> "BFFMResults":
        """dim=0 means multiple chains, dim=1 means concatenate"""
        out = results[0]
        for r in results[1:]:
            out = out.append(r, dim)
        return out

    def append(self, other: "BFFMResults", dim: int) -> "BFFMResults":
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

    def _check_parameters(self, other: "BFFMResults"):
        """Check that the other has the same prior, dimensions, etc."""
        if self.warmup != other.warmup:
            warnings.warn("Warmup is different between two MCMCResults")
        if self.thin != other.thin:
            warnings.warn("Thinning is different between two MCMCResults")
        if self.prior != other.prior:
            warnings.warn("Prior is different between two MCMCResults")
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

    def posterior_mean(self, by_chains=False):
        if not self._aligned:
            warnings.warn("The chains were not aligned. The posterior means may not correct.")
        if by_chains:
            return {k: v._mean(1) for k, v in self.chains.items()}
        return {k: v._mean((0, 1)) for k, v in self.chains.items()}

    def posterior_median(self, by_chains=False):
        if not self._aligned:
            warnings.warn("The chains were not aligned. The posterior medians may not correct.")
        if by_chains:
            return {k: v.median(1) for k, v in self.chains.items()}
        return {k: v.median((0, 1)) for k, v in self.chains.items()}

    def metrics(self, true_values: dict[str, torch.Tensor]):
        means = self.posterior_mean()
        out = defaultdict(dict)
        for k, v in true_values.items():
            out[k] = defaultdict(lambda: float("nan"))
            for m, f in metrics[k].items():
                out[k][m] = f(means[k], v)
        return out

    def add_transformed_variables(self):
        add_transformed_variables(self.chains)

    def align(self, loadings: torch.Tensor | None = None):
        print(f"{suffix}Aligning component order")
        self.align_order_chains(loadings)
        print(f"{suffix}Aligning signs within chains")
        for chain in range(self.n_chains):
            self.align_signs_chain(chain, loadings)
        print(f"{suffix}Aligning signs between chains")
        self.align_signs_chains(loadings)
        self._aligned = True

    def align_signs(self, loadings: torch.Tensor | None = None):
        for chain in range(self.n_chains):
            self.align_signs_chain(chain, loadings)
        self.align_signs_chains(loadings)
        self._aligned = True

    def align_signs_chain(self, chain: int, loadings: torch.Tensor | None = None):
        # choose last entry of first chain as reference
        if loadings is None:  # align to self if not specified
            loadings = self.chains["loadings"][chain, -1, ...]  # E x K
        ips = torch.einsum(
            "nek,ek->nk",
            self.chains["loadings"][chain, ...],
            loadings
        )
        signflips = torch.sign(ips)  # N x K
        print(f"{suffix}Number of sign flips in chain {chain}: {(signflips == -1).sum(0).tolist()}")
        self.chains["loadings"][chain, ...] *= signflips.unsqueeze(1)
        self.chains["smgp_factors.nontarget_process"][chain, ...] *= signflips.unsqueeze(-1)
        self.chains["smgp_factors.target_process"][chain, ...] *= signflips.unsqueeze(-1)

    def align_signs_chains(self, loadings: torch.Tensor | None = None):
        # choose last entry of first chain as reference
        if loadings is None:  # align to self
            loadings = self.chains["loadings"][0, -1, ...]  # E x K
        ips = torch.einsum(
            "bek, ek->bk",
            self.chains["loadings"][:, -1, ...],
            loadings
        )
        signflips = torch.sign(ips)  # B x K
        self.chains["loadings"] *= signflips.unsqueeze(1).unsqueeze(-2)
        self.chains["smgp_factors.nontarget_process"] *= signflips.unsqueeze(1).unsqueeze(-1)
        self.chains["smgp_factors.target_process"] *= signflips.unsqueeze(1).unsqueeze(-1)

    def check_concordance(self, loadings: torch.Tensor | None = None):
        # choose last entry of first chain as reference
        if loadings is None:  # align to self
            loadings = self.chains["loadings"][0, -1, ...]  # E x K
        # first look at the ordering
        ips = torch.einsum(
            "bek, ej->bkj",
            self.chains["loadings"][:, -1, ...],
            loadings
        ).abs()
        norm0 = torch.norm(loadings, dim=0)
        norm1 = torch.norm(self.chains["loadings"][:, -1, ...], dim=1)
        ips /= norm0.unsqueeze(0).unsqueeze(0) * norm1.unsqueeze(-1)
        N, K, _ = ips.shape
        cs = ips.gather(-1, torch.arange(K).reshape(1, K, 1).repeat(N, 1, 1)).squeeze(-1)
        print(f"{suffix}Cosine similarities without alignment:")
        print(cs)

        orders = torch.zeros(N, K) - 1
        best_order = ips.argsort(-1, descending=True)
        # we would like to use best_order[:, :, 0], but we could have repetitions
        for n in range(N):
            chosen = []
            for k in range(K):
                j = best_order[n, k, 0]
                ith = 0
                while j in chosen:
                    ith += 1
                    j = best_order[n, k, ith]
                chosen.append(j)
                orders[n, k] = j
        print(f"{suffix}Proposed alignment:")
        print(orders.long())
        cs = ips.gather(-1, orders.long().unsqueeze(-1)).squeeze(-1)
        print(f"{suffix}Cosine similarities of proposed alignment:")
        print(cs)
        if cs.lt(0.8).any():
            warnings.warn("Some cosine similarities are below 0.8. "
                          "This may indicate a problem with the alignment.")
        return orders

    def align_order_chains(self, loadings: torch.Tensor | None = None, orders: torch.Tensor | None = None):
        if orders is None:
            orders = self.check_concordance(loadings)
        for n in range(self.n_chains):
            self.chains["loadings"][n, ...] = \
                self.chains["loadings"][n, :, :, orders[n, :].long()]
            self.chains["heterogeneities"][n, ...] = \
                self.chains["heterogeneities"][n, :, :, orders[n, :].long()]
            self.chains["shrinkage_factor"][n, ...] = \
                self.chains["shrinkage_factor"][n, :, orders[n, :].long()]

            self.chains["smgp_factors.nontarget_process"][n, ...] = \
                self.chains["smgp_factors.nontarget_process"][n, :, orders[n, :].long(), :]
            self.chains["smgp_factors.target_process"][n, ...] = \
                self.chains["smgp_factors.target_process"][n, :, orders[n, :].long(), :]
            self.chains["smgp_factors.mixing_process"][n, ...] = \
                self.chains["smgp_factors.mixing_process"][n, :, orders[n, :].long(), :]

            self.chains["smgp_scaling.nontarget_process"][n, ...] = \
                self.chains["smgp_scaling.nontarget_process"][n, :, orders[n, :].long(), :]
            self.chains["smgp_scaling.target_process"][n, ...] = \
                self.chains["smgp_scaling.target_process"][n, :, orders[n, :].long(), :]
            self.chains["smgp_scaling.mixing_process"][n, ...] = \
                self.chains["smgp_scaling.mixing_process"][n, :, orders[n, :].long(), :]

    @property
    def n_chains(self) -> int:
        return next(iter(self.chains.values())).shape[0]

    @property
    def n_samples(self) -> int:
        return next(iter(self.chains.values())).shape[1]

    def __repr__(self):
        return f"{self.__class__.__name__}({self.n_chains} chains, {self.n_samples} samples)"
        pass

    def to_arviz(self) -> az.InferenceData:
        return az.from_dict(posterior={k: v.cpu() for k, v in self.chains.items()})

    def to_predict(self, thin: int | None = None, n_samples: int | None = 1000) -> BFFMPredict:
        """Create a Predict object from the MCMCResults."""
        if thin is None:
            thin = int(np.ceil(self.n_samples * self.n_chains / n_samples))
        variables = {
            k: self.chains[k].flatten(0, 1)[::thin, ...]
            for k in self._sufficient_for_prediction
        }
        return BFFMPredict(
            variables=variables,
            dimensions=self.dimensions,
            prior=self.prior,
            settings=self.settings
        )


def add_transformed_variables(chains):
    if "loadings" in chains:
        L = chains["loadings"]  # (..., E, K)
        chains["loadings.inner_products"] = L @ L.transpose(-1, -2) # (..., E, E)
    if "loadings" in chains:
        L = chains["loadings"].transpose(-1, -2)  # (..., K, E)
        chains["loadings.rank_one"] = L.unsqueeze(-1) @ L.unsqueeze(-2)  # (..., K, E, E)
    if "loadings" in chains:
        L = chains["loadings"]  # (..., E, K)
        LtL = L.transpose(-1, -2) @ L  # (..., K, K)
        LtLinv = torch.linalg.inv(LtL)  # (..., K, K)
        chains["loadings.projection"] = L @ LtLinv @ L.transpose(-1, -2)  # (..., E, E)
    if "loadings" in chains:
        L = chains["loadings"]
        Lnorm = torch.linalg.norm(L, dim=-2, keepdim=True)
        chains["loadings.norm_one"] = L / Lnorm
    if "loadings" in chains and "shrinkage_factor" in chains:
        L = chains["loadings"]
        s = chains["shrinkage_factor"].unsqueeze(-2)
        chains["loadings.times_shrinkage"] = L * s.sqrt()
    # Factor process global: target signal and difference process
    if "smgp_factors.nontarget_process" in chains and \
            "smgp_factors.target_process" in chains and \
            "smgp_factors.mixing_process" in chains:
        chains["smgp_factors.target_signal"] = \
            (1 - chains["smgp_factors.mixing_process"]) * \
            chains["smgp_factors.nontarget_process"] + \
            chains["smgp_factors.mixing_process"] * \
            chains["smgp_factors.target_process"]
    if "smgp_factors.nontarget_process" in chains and \
            "smgp_factors.target_process" in chains and \
            "smgp_factors.mixing_process" in chains:
        chains["smgp_factors.difference_process"] = \
            chains["smgp_factors.target_signal"] - \
            chains["smgp_factors.nontarget_process"]
    # Scaling process global: target signal and difference process
    if "smgp_scaling.nontarget_process" in chains and \
            "smgp_scaling.target_process" in chains and \
            "smgp_scaling.mixing_process" in chains:
        chains["smgp_scaling.target_signal"] = \
            (1 - chains["smgp_scaling.mixing_process"]) * \
            chains["smgp_scaling.nontarget_process"] + \
            chains["smgp_scaling.mixing_process"] * \
            chains["smgp_scaling.target_process"]
    if "smgp_scaling.nontarget_process" in chains and \
            "smgp_scaling.target_process" in chains and \
            "smgp_scaling.mixing_process" in chains:
        chains["smgp_scaling.difference_process"] = \
            chains["smgp_scaling.target_signal"] - \
            chains["smgp_scaling.nontarget_process"]
    # Scaling process global: scaled by shrinkage
    # TODO: check that we might be better with *?
    if "smgp_scaling.nontarget_process" in chains and \
            "shrinkage_factor" in chains:
        s = chains["shrinkage_factor"].unsqueeze(-1)
        chains["smgp_scaling.nontarget_process_times_shrinkage"] = \
            chains["smgp_scaling.nontarget_process"] / s.sqrt()
    if "smgp_scaling.target_process" in chains and \
            "shrinkage_factor" in chains:
        s = chains["shrinkage_factor"].unsqueeze(-1)
        chains["smgp_scaling.target_process_times_shrinkage"] = \
            chains["smgp_scaling.target_process"] / s.sqrt()
    if "smgp_scaling.target_signal" in chains and \
            "shrinkage_factor" in chains:
        s = chains["shrinkage_factor"].unsqueeze(-1)
        chains["smgp_scaling.target_signal_times_shrinkage"] = \
            chains["smgp_scaling.target_signal"] / s.sqrt()
    if "smgp_scaling.difference_process" in chains and \
            "shrinkage_factor" in chains:
        s = chains["shrinkage_factor"].unsqueeze(-1)
        chains["smgp_scaling.difference_process_times_shrinkage"] = \
            chains["smgp_scaling.difference_process"] / s.sqrt()
    # Scaling process global: center at 1 using geometric mean
    if "smgp_scaling.nontarget_process" in chains:
        center = chains["smgp_scaling.nontarget_process"].log().mean(-1, keepdim=True).exp()
        chains["smgp_scaling.nontarget_process_centered"] = \
            chains["smgp_scaling.nontarget_process"] / center
    if "smgp_scaling.target_process" in chains:
        center = chains["smgp_scaling.target_process"].log().mean(-1, keepdim=True).exp()
        chains["smgp_scaling.target_process_centered"] = \
            chains["smgp_scaling.target_process"] / center
    if "smgp_scaling.target_signal" in chains:
        center = chains["smgp_scaling.target_signal"].log().mean(-1, keepdim=True).exp()
        chains["smgp_scaling.target_signal_centered"] = \
            chains["smgp_scaling.target_signal"] / center
    if "smgp_scaling.difference_process" in chains:
        center = chains["smgp_scaling.difference_process"].mean(-1, keepdim=True)
        chains["smgp_scaling.difference_process_centered"] = \
            chains["smgp_scaling.difference_process"] - center
    if "smgp_scaling.difference_process" in chains and \
            "smgp_scaling.nontarget_process" in chains:
        chains["smgp_scaling.target_multiplier_process"] = \
            1 + chains["smgp_scaling.difference_process"] / \
                chains["smgp_scaling.nontarget_process"]
    if "smgp_scaling.nontarget_process" in chains and \
        "loadings" in chains:
        Lnorm = torch.linalg.norm(chains["loadings"], dim=-2, keepdim=False)
        pnorm = chains["smgp_scaling.nontarget_process"].log().mean(-1, keepdim=False).exp()
        chains["scaling_factor"] = (pnorm * Lnorm).pow(2.)
