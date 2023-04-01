import torch
import math
from .utils import Kernel
import scipy
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from . import BFFModel


class BFFMPredict:
    """
    Excpexted:
    loadings (N x E x K)
    observation_variance (N x E)
    smgp_factors.nontarget_process (N x K x T)
    smgp_factors.target_signal (N x K x T)
    smgp_scaling.nontarget_process (N x K x T)
    smgp_scaling.target_signal (N x K x T)

    Note that we index row first then column
    So the char label should run by row first.
    """

    def __init__(
            self,
            variables: dict[str, torch.Tensor],
            dimensions: dict,
            prior: dict,
            character_labels: list[str] | None = None
    ):
        self.variables = variables
        self.dimensions = dimensions
        self.prior = prior
        p = prior["kernel_gp_factor"]
        tmat = scipy.linalg.toeplitz(p[0] ** (np.arange(dimensions["n_timepoints"]) * p[2]))
        self._kernel = Kernel.from_covariance_matrix(torch.Tensor(tmat) * p[1])
        self._dist = MultivariateNormal(
            loc=torch.zeros(self._kernel.shape[0]),
            scale_tril=self._kernel.chol
        )
        if character_labels is None:
            nc = 1
            for i in self.dimensions["n_stimulus"]:
                nc *= i
            character_labels = [f"C{i+1}" for i in range(nc)]
        self.character_labels = character_labels

    @property
    def n_samples(self):
        return self.variables["loadings"].shape[0]

    @property
    def combinations(self) -> torch.Tensor:
        Js = self.dimensions["n_stimulus"]
        # TODO this is hard coded for 6-6
        Js = (6, 6)
        combinations = torch.cartesian_prod(*[torch.arange(J) for J in Js])
        to_add = torch.cumsum(torch.Tensor([0] + list(Js))[0:-1], 0).reshape(1, -1)
        combinations = combinations + to_add
        combinations = torch.nn.functional.one_hot(combinations.long(), sum(Js)).sum(1)
        # this is hard coded for 12 choose 2
        # i0, i1 = torch.tril_indices(12, 12, offset=-1)
        # combinations = torch.nn.functional.one_hot(i0.long(), 12)
        # combinations += torch.nn.functional.one_hot(i1.long(), 12)
        return combinations  # L x J

    def predict(
            self,
            order: torch.Tensor,  # M x J
            sequence: torch.Tensor,  # M x E x T,
            factor_samples: int,
            character_idx: torch.Tensor | None = None,  # M
            factor_processes_method: str = "posterior",
            aggregation_method: str = "product",
            return_cumulative: bool = False,
    ):
        # first step is always to get log-likelihood for all sequences
        # all combinations and all posterior samples
        llk = self.log_likelihood(
            order=order,
            sequence=sequence,
            factor_samples=factor_samples,
            factor_processes_method=factor_processes_method,
        )
        if character_idx is None:
            llk_long = llk.unsqueeze(1)  # M x 1 x L x N
        else:
            character_idx = character_idx.flatten()
            chars = character_idx.unique()
            max_rep = max([(character_idx == char).int().sum().item() for char in chars])
            llk_long = torch.zeros((len(chars), max_rep, llk.shape[1], llk.shape[2]))
            for i, char in enumerate(chars):
                idx = character_idx == char
                llk_long[i, :sum(idx), ...] = llk[idx, ...]
            # should now be nc x nr x L x N
        # Aggregation switch
        if aggregation_method == "product":
            # get log prob for each sequence
            log_prob = torch.logsumexp(llk_long, dim=3) - math.log(self.n_samples)
            # get log prob for each character (this is the product)
            log_prob = log_prob.cumsum(1)  # nc x nr x L
            # map to probabilities
            log_prob = log_prob - torch.logsumexp(log_prob, dim=2, keepdim=True)
        elif aggregation_method == "integral":
            # sum over repetitions
            log_prob = llk_long.cumsum(1)  # nc x nr x L x N
            # get log prob for each character
            log_prob = torch.logsumexp(log_prob, dim=3) - math.log(self.n_samples)
            # map to probabilities
            log_prob = log_prob - torch.logsumexp(log_prob, dim=2, keepdim=True)
        else:
            raise ValueError(f"Unknown aggregation method {aggregation_method}")

        # get predicted class
        wide_pred = log_prob.argmax(2)
        wide_pred_one_hot = self.combinations[wide_pred, :]
        if character_idx is None:
            return log_prob[:, 0, :], wide_pred_one_hot[:, 0, :]
        else:
            if return_cumulative:
                return log_prob, wide_pred_one_hot, chars
            else:
                return log_prob[:, -1, :], wide_pred_one_hot[:, -1, :], chars

    def log_likelihood(
            self,
            order: torch.Tensor,  # M x J
            sequence: torch.Tensor,  # M x E x T,
            factor_samples: int,
            factor_processes_method: str = "posterior",
    ):
        N = self.n_samples
        M, E, T = sequence.shape
        L = self.combinations.shape[0]
        B = factor_samples
        target_repeated = self.combinations.repeat(M, 1)  # (ML) x J
        sequence_repeated = sequence.repeat_interleave(L, 0)  # (ML) x E x T
        order_repeated = order.repeat_interleave(L, 0)  # (ML) x J
        # NB the ordering is
        # seq0 ... seq0, seq1 ... seq1, seq2 ... seq2
        # comb0, comb1, comb2, comb0, comb1, comb2, comb0, comb1, comb2
        self.dimensions["n_sequences"] = M

        bffmodel = BFFModel(
            stimulus_order=order_repeated,
            target_stimulus=target_repeated,
            sequences=sequence_repeated,
            **self.prior,
            **self.dimensions,
        )

        # run through all posterior samples
        llk = torch.zeros(M, L, N)
        for sample_idx in range(N):
            print(f"Sample {sample_idx+1}/{N}")
            # get global variables
            variables = {
                "loadings": self.variables["loadings"][sample_idx, :, :],
                "observation_variance": self.variables["observation_variance"][sample_idx, :],
                "smgp_factors": {
                    "nontarget_process": self.variables["smgp_factors.nontarget_process"][sample_idx, :, :],
                    "target_process": self.variables["smgp_factors.target_process"][sample_idx, :, :],
                    "mixing_process": self.variables["smgp_factors.target_process"][sample_idx, :, :],
                },
                "smgp_scaling": {
                    "nontarget_process": self.variables["smgp_scaling.nontarget_process"][sample_idx, :, :],
                    "target_process": self.variables["smgp_scaling.target_process"][sample_idx, :, :],
                    "mixing_process": self.variables["smgp_scaling.target_process"][sample_idx, :, :],
                }
            }
            bffmodel.set(**variables)
            bffmodel.generate_local_variables()

            if factor_processes_method == "posterior":
                llk_idx = torch.zeros(M*L, B)
                for b in range(B):
                    bffmodel.variables["factor_processes"].sample()
                    llk_idx[:, b] = bffmodel.variables["observations"].log_density_per_sequence
                llk_idx = math.log(B) - torch.logsumexp(-llk_idx, 1)
            elif factor_processes_method == "prior":
                llk_idx = torch.zeros(M*L, B)
                for b in range(B):
                    bffmodel.variables["factor_processes"].generate()
                    llk_idx[:, b] = bffmodel.variables["observations"].log_density_per_sequence
                llk_idx = torch.logsumexp(llk_idx, 1) - math.log(B)
            elif factor_processes_method == "prior_mean":
                bffmodel.variables["factor_processes"].data = \
                    bffmodel.variables["mean_factor_processes"].data
                llk_idx = bffmodel.variables["observations"].log_density_per_sequence
            elif factor_processes_method == "posterior_mean":
                bffmodel.variables["factor_processes"].data = \
                    bffmodel.variables["factor_processes"].posterior_mean
                llk_idx = bffmodel.variables["observations"].log_density_per_sequence
            elif factor_processes_method == "analytical":
                llk_idx = torch.zeros(M*L)
                x = bffmodel.variables["observations"].data  # (ML) x E x T
                xi = bffmodel.variables["loading_processes"].data # (ML) x K x T
                zbar = bffmodel.variables["mean_factor_processes"].data  # (ML) x K x T
                Sigma = bffmodel.variables["observation_variance"].data  # E
                Theta = bffmodel.variables["loadings"].data  # E x K
                Kmat = bffmodel.variables["factor_processes"].kernel.cov  # T x T
                mean = torch.einsum("mkt, ek -> met", xi*zbar, Theta)  # (ML) x E x T
                # TODO need to vectorize this, way too slow currently
                for ml in range(M*L):
                    if ml % 100 == 0:
                        print(f"Sample {sample_idx+1}/{N}, sequence {ml+1}/{M*L}")
                    mean_ml = mean[ml, :, :]  # E x T
                    mean_ml = mean_ml.flatten()  # blocks are per channel [T, ..., T]
                    torch.einsum("ek, fk, kt -> eft", Theta, Theta, xi[ml, :, :].pow(2))
                    cov = torch.einsum(
                        "ek, fk, kt, ts, ks-> efts",
                        Theta, Theta, xi[ml, :, :], Kmat, xi[ml, :, :]
                    )
                    cov = cov.permute(0, 2, 1, 3).flatten(2, 3).flatten(0, 1)
                    cov = cov + torch.kron(torch.diag(Sigma), torch.eye(T))
                    dist = torch.distributions.MultivariateNormal(mean_ml, cov)
                    llk_idx[ml] = dist.log_prob(x[ml, :, :].flatten())
            else:
                raise ValueError(f"Unknown factor_processes_method {factor_processes_method}")
            llk[:, :, sample_idx] = llk_idx.reshape(M, L)
        return llk  # M x L x N

    def one_hot_to_combination_id(self, one_hot: torch.Tensor):
        # one_hot should be ... x n_combinations
        ips = torch.einsum("...i,ji->...j", one_hot.double(), self.combinations.double())
        idx = torch.argmax(ips, -1)
        return idx