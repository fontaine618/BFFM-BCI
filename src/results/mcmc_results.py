from typing import Tuple, Union, Any
import torch
import numpy as np
import arviz as az
import pickle


class MCMCResults:
    """Results of an MCMC run.

    Indexing:
    - first dimension is the chain index
    - second dimension is the sample index
    - all other dimensions are the dimensions of the variable/llk

    """

    def __init__(
        self,
        prior: dict[str: Any],
        dimensions: dict[str: int],
        chains: dict[str: torch.Tensor],
        log_likelihood: dict[str: torch.Tensor] | None = None,
        warmup: int = 0,
        thin: int = 1
    ):
        self.variables_for_prediction = None
        pass

    @classmethod
    def from_files(
            cls,
            files: list[list[str]] | list[str]
    ):
        pass

    @staticmethod
    def _read_files_single_chain(
            files: list[str] | str,
    ):
        prior = None
        dimensions = None
        chain = None
        log_likelihood = None
        if isinstance(files, str):
            files = [files]
        for file in files:
            with open(file, "rb") as f:
                result = pickle.load(f)

            # check prior
            if prior is None:
                prior = result["prior"]
            else:
                if prior != result["prior"]:
                    raise ValueError("Prior is different across files")

            # check dimensions
            if dimensions is None:
                dimensions = result["dimensions"]
            else:
                if dimensions != result["dimensions"]:
                    raise ValueError("Dimensions are different across files")

            # check log_likelihood
            if log_likelihood is None:
                log_likelihood = result["log_likelihood"]
            else:
                # need exactly the same keys
                if log_likelihood.keys() != result["log_likelihood"].keys():
                    raise ValueError("Log likelihoods are different across files")
                for k, v in log_likelihood.items():
                    result["log_likelihood"][k] = torch.concat(
                        (result["log_likelihood"][k], v),
                        dim=0
                    )

            # check chain
            if chain is None:
                chain = result["chain"]
            else:
                # need exactly the same keys
                if chain.keys() != result["chain"].keys():
                    raise ValueError("Chains are different across files")
                for k, v in chain.items():
                    # need dame dimensions except the first one
                    if v.shape[1:] != result["chain"][k].shape[1:]:
                        raise ValueError(f"Variable {k} has different dimensions across files")
                    result["chain"][k] = torch.concat(
                        (result["chain"][k], v),
                        dim=0
                    )
        return prior, dimensions, chain, log_likelihood

    def _add_transformed_variables(self):
        # TODO
        pass

    def _align(self):
        #TODO
        pass

    def prepare_for_prediction(self, n_sample: int = 1000):
        # TODO
        pass

    def predict(
            self,
            sequences: torch.Tensor,
            stimulus_order: torch.Tensor,
            n_samples: int = 1000,
            **kwargs
    ):
        if self.variables_for_prediction is None:
            self.prepare_for_prediction(n_samples)
        # TODO
        pass

