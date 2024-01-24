import torch
from typing import Any, Callable
import math
from .utils import Kernel
import scipy
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal
from . import BFFModel

T = torch.Tensor


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
            variables: dict[str: T],
            dimensions: dict[str: Any],
            prior: dict[str: Any],
            settings: dict[str: Any],
            character_labels: list[str] | None = None
    ):
        self.variables = variables
        self.dimensions = dimensions
        self.prior = prior
        self.settings = settings
        p = prior["kernel_gp_factor"]
        tmat = scipy.linalg.toeplitz(p[0] ** (np.arange(dimensions["n_timepoints"]) * p[2]))
        self._kernel = Kernel.from_covariance_matrix(T(tmat) * p[1])
        self._dist = MultivariateNormal(
            loc=torch.zeros(self._kernel.shape[0]),
            scale_tril=self._kernel.chol
        )
        if character_labels is None:
            nc = 1
            for i in self.dimensions["n_stimulus"]:
                nc *= i
            character_labels = [f"C{i + 1}" for i in range(nc)]
        self.character_labels = character_labels

    @property
    def n_samples(self):
        return self.variables["loadings"].shape[0]

    @property
    def combinations(self) -> T:
        Js = self.dimensions["n_stimulus"]
        # TODO this is hard coded for 6-6
        Js = (6, 6)
        combinations = torch.cartesian_prod(*[torch.arange(J) for J in Js])
        to_add = torch.cumsum(T([0] + list(Js))[0:-1], 0).reshape(1, -1)
        combinations = combinations + to_add
        combinations = torch.nn.functional.one_hot(combinations.long(), sum(Js)).sum(1)
        # this is hard coded for 12 choose 2
        # i0, i1 = torch.tril_indices(12, 12, offset=-1)
        # combinations = torch.nn.functional.one_hot(i0.long(), 12)
        # combinations += torch.nn.functional.one_hot(i1.long(), 12)
        return combinations  # L x J

    def predict(
            self,
            order: T,  # M x J
            sequence: T,  # M x E x T,
            factor_samples: int = 10,
            character_idx: T | None = None,  # M
            factor_processes_method: str = "analytic",
            drop_component: int | None = None,
            batchsize: int = 25,
    ):
        # first step is always to get log-likelihood for all sequences
        # all combinations and all posterior samples
        llk = self.log_likelihood(
            order=order,
            sequence=sequence,
            factor_samples=factor_samples,
            factor_processes_method=factor_processes_method,
            drop_component=drop_component,
            batchsize=batchsize
        )
        if character_idx is None:
            llk_long = llk.unsqueeze(1)  # M x 1 x L x N
            chars = None
        else:
            character_idx = character_idx.flatten()
            chars = character_idx.unique()
            max_rep = max([(character_idx == char).int().sum().item() for char in chars])
            llk_long = torch.zeros((len(chars), max_rep, llk.shape[1], llk.shape[2]))
            for i, char in enumerate(chars):
                idx = character_idx == char
                llk_long[i, :sum(idx), ...] = llk[idx, ...]
            # should now be nc x nr x L x N
        return llk_long, chars

    @staticmethod
    def aggregate(
            llk_long, # nc x nr x L x N
            sample_mean: str = "harmonic",
            which_first: str = "sample",
    ) -> T:
        """
        Aggregates log-likelihoods over sequences and samples.
        :param llk_long: nc x nr x L x N
        :param sequence_mean: arithmetic, geometric, harmonic
        :param sample_mean: arithmetic, geometric, harmonic
        :param which_first: sequence, sample
        :return: nc x nr x L
        """
        if which_first == "sequence":
            log_prob = torch.cumsum(llk_long, dim=1)
            if sample_mean == "arithmetic":
                log_prob = torch.logsumexp(log_prob, dim=3) - math.log(llk_long.shape[3])
            elif sample_mean == "geometric":
                log_prob = torch.mean(log_prob, dim=3)
            elif sample_mean == "harmonic":
                log_prob = -torch.logsumexp(-log_prob, dim=3) + math.log(llk_long.shape[3])
            else:
                raise ValueError(f"Unknown sample_mean {sample_mean}")
        elif which_first == "sample":
            if sample_mean == "arithmetic":
                log_prob = torch.logsumexp(llk_long, dim=3) - math.log(llk_long.shape[3])
            elif sample_mean == "geometric":
                log_prob = torch.mean(llk_long, dim=3)
            elif sample_mean == "harmonic":
                log_prob = -torch.logsumexp(-llk_long, dim=3) + math.log(llk_long.shape[3])
            else:
                raise ValueError(f"Unknown sample_mean {sample_mean}")
            log_prob = torch.cumsum(log_prob, dim=1)
        else:
            raise ValueError(f"Unknown which_first {which_first}")
        return log_prob - torch.logsumexp(log_prob, dim=2, keepdim=True)

    def get_predictions(
            self,
            log_prob: T,
            return_cumulative: bool = True,
    ) -> T:
        wide_pred = log_prob.argmax(2)
        wide_pred_one_hot = self.combinations[wide_pred, :]
        if return_cumulative:
            return wide_pred_one_hot
        else:
            return wide_pred_one_hot[:, -1, :]

    def marginal_log_likelihood(
            self,
            order: T,  # M x J
            sequence: T,  # M x E x T,
            target: T,  # M x J
            batch_size: int = 25,
    ):
        # TODO: perhaps abstract thing a bit for the other similar method
        N = self.n_samples
        M, E, nt = sequence.shape
        self.dimensions["n_sequences"] = M

        bffmodel = BFFModel(
            stimulus_order=order,
            target_stimulus=target,
            sequences=sequence,
            **self.settings,
            **self.prior,
            **self.dimensions,
        )

        # run through all posterior samples
        llk = torch.zeros(M, N)
        for sample_idx in range(N):
            print(f"Sample {sample_idx + 1}/{N}")
            # get global variables
            self.update_model(bffmodel, sample_idx, None)
            llk_idx = torch.zeros(M)
            x = bffmodel.variables["observations"].data  # (ML) x E x T
            xi = bffmodel.variables["loading_processes"].data  # (ML) x K x T
            zbar = bffmodel.variables["mean_factor_processes"].data  # (ML) x K x T
            Sigma = bffmodel.variables["observation_variance"].data  # E
            Theta = bffmodel.variables["loadings"].data  # E x K
            Kmat = bffmodel.variables["factor_processes"].kernel.cov  # T x T
            mean = torch.einsum("mkt, ek -> met", xi * zbar, Theta)  # (ML) x E x T
            n_batches = M // batch_size
            if M % batch_size > 0:
                n_batches += 1
            for batch_idx in range(n_batches):
                ml = torch.arange(batch_idx * batch_size, min((batch_idx + 1) * batch_size, M))
                # print(f"Sample {sample_idx + 1}/{N}, batch {batch_idx + 1}/{n_batches}"
                #       f" ({batch_size} sequences per batch)")
                mean_ml = mean[ml, :, :]  # ... x E x T
                mean_ml = mean_ml.flatten(1)  # blocks are per channel [T, ..., T]
                cov = torch.einsum(
                    "ek, fk, bkt, ts, bks-> befts",
                    Theta, Theta, xi[ml, :, :], Kmat, xi[ml, :, :]
                )
                cov = cov.permute(0, 1, 3, 2, 4).flatten(3, 4).flatten(1, 2)
                cov = 0.5 * (cov + cov.transpose(1, 2))
                cov = cov + torch.kron(torch.diag(Sigma), torch.eye(nt)).unsqueeze(0)
                dist = torch.distributions.MultivariateNormal(mean_ml, cov)
                llk_idx[ml] = dist.log_prob(x[ml, :, :].flatten(1))
            llk[:, sample_idx] = llk_idx
        return llk  # M x N

    def maximum_log_likelihood(
            self,
            order: T,  # M x J
            sequence: T,  # M x E x T,
            target: T,  # M x J
    ):
        # TODO: perhaps abstract thing a bit for the other similar method
        N = self.n_samples
        M, E, nt = sequence.shape
        self.dimensions["n_sequences"] = M

        bffmodel = BFFModel(
            stimulus_order=order,
            target_stimulus=target,
            sequences=sequence,
            **self.settings,
            **self.prior,
            **self.dimensions,
        )

        # run through all posterior samples
        llk = torch.zeros(M, N)
        for sample_idx in range(N):
            print(f"Sample {sample_idx + 1}/{N}")
            # get global variables
            self.update_model(bffmodel, sample_idx, None)
            bffmodel.variables["factor_processes"].data = \
                bffmodel.variables["factor_processes"].posterior_mean
            llk_idx = bffmodel.variables["observations"].log_density_per_sequence + \
                      bffmodel.variables["factor_processes"].log_density_per_sequence
            llk[:, sample_idx] = llk_idx
        return llk  # M x N

    def log_likelihood(
            self,
            order: T,  # M x J
            sequence: T,  # M x E x T,
            factor_samples: int,
            factor_processes_method: str = "maximize",
            drop_component: int | None = None,
            batchsize: int = 25,
    ):
        N = self.n_samples
        M, E, nt = sequence.shape
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
            **self.settings,
            **self.prior,
            **self.dimensions,
        )

        # run through all posterior samples
        llk = torch.zeros(M, L, N)
        for sample_idx in range(N):
            print(f"Sample {sample_idx + 1}/{N}")
            # get global variables
            self.update_model(bffmodel, sample_idx, drop_component)

            if factor_processes_method == "posterior":
                llk_idx = torch.zeros(M * L, B)
                for b in range(20):
                    bffmodel.variables["factor_processes"].sample()
                    newllk = bffmodel.variables["observations"].log_density
                    print(f"Burn-in {b}: {newllk}")
                for b in range(B):
                    bffmodel.variables["factor_processes"].sample()
                    newllk = bffmodel.variables["observations"].log_density
                    print(f"Samples {b}: {newllk}")
                    llk_idx[:, b] = bffmodel.variables["observations"].log_density_per_sequence
                llk_idx = math.log(B) - torch.logsumexp(-llk_idx, 1)
            elif factor_processes_method == "prior":
                llk_idx = torch.zeros(M * L, B)
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
                    bffmodel.variables["factor_processes"].posterior_mean_by_conditionals
                llk_idx = bffmodel.variables["observations"].log_density_per_sequence
            elif factor_processes_method == "maximize":
                bffmodel.variables["factor_processes"].data = \
                    bffmodel.variables["factor_processes"].posterior_mean_by_conditionals
                llk_idx = bffmodel.variables["observations"].log_density_per_sequence + \
                            bffmodel.variables["factor_processes"].log_density_per_sequence
            elif factor_processes_method == "analytical":
                llk_idx = torch.zeros(M * L)
                x = bffmodel.variables["observations"].data  # (ML) x E x T
                xi = bffmodel.variables["loading_processes"].data  # (ML) x K x T
                zbar = bffmodel.variables["mean_factor_processes"].data  # (ML) x K x T
                Sigma = bffmodel.variables["observation_variance"].data  # E
                Theta = bffmodel.variables["loadings"].data  # E x K
                Kmat = bffmodel.variables["factor_processes"].kernel.cov  # T x T
                mean = torch.einsum("mkt, ek -> met", xi * zbar, Theta)  # (ML) x E x T
                batch_size = batchsize
                n_batches = M * L // batch_size
                if M * L % batch_size > 0:
                    n_batches += 1
                for batch_idx in range(n_batches):
                    ml = torch.arange(batch_idx * batch_size, min((batch_idx + 1) * batch_size, M*L))
                    # print(f"Sample {sample_idx + 1}/{N}, batch {batch_idx + 1}/{n_batches}"
                    #       f" ({batch_size} sequences per batch)")
                    mean_ml = mean[ml, :, :]  # ... x E x T
                    mean_ml = mean_ml.flatten(1)  # blocks are per channel [T, ..., T]
                    cov = torch.einsum(
                        "ek, fk, bkt, ts, bks-> befts",
                        Theta, Theta, xi[ml, :, :], Kmat, xi[ml, :, :]
                    )
                    cov = cov.permute(0, 1, 3, 2, 4).flatten(3, 4).flatten(1, 2)
                    cov = 0.5 * (cov + cov.transpose(1, 2))
                    cov = cov + torch.kron(torch.diag(Sigma), torch.eye(nt)).unsqueeze(0)
                    dist = torch.distributions.MultivariateNormal(mean_ml, cov)
                    llk_idx[ml] = dist.log_prob(x[ml, :, :].flatten(1))
            else:
                raise ValueError(f"Unknown factor_processes_method {factor_processes_method}")
            llk[:, :, sample_idx] = llk_idx.reshape(M, L)
        return llk  # M x L x N

    def update_model(self, bffmodel, sample_idx, drop_component):
        variables = {
            "loadings": self.variables["loadings"][sample_idx, :, :].clone().detach(),
            "observation_variance": self.variables["observation_variance"][sample_idx, :],
            "smgp_factors": {
                "nontarget_process": self.variables["smgp_factors.nontarget_process"][sample_idx, :, :],
                "target_process": self.variables["smgp_factors.target_process"][sample_idx, :, :],
                "mixing_process": self.variables["smgp_factors.mixing_process"][sample_idx, :, :],
            },
            "smgp_scaling": {
                "nontarget_process": self.variables["smgp_scaling.nontarget_process"][sample_idx, :, :],
                "target_process": self.variables["smgp_scaling.target_process"][sample_idx, :, :],
                "mixing_process": self.variables["smgp_scaling.mixing_process"][sample_idx, :, :],
            }
        }
        if drop_component is not None:
            variables["smgp_scaling.mixing_process"][:, drop_component, :] = 0.
            variables["smgp_factors.mixing_process"][:, drop_component, :] = 0.
        bffmodel.set(**variables)
        bffmodel.generate_local_variables()

    def one_hot_to_combination_id(self, one_hot: T):
        # one_hot should be ... x n_combinations
        ips = torch.einsum("...i,ji->...j", one_hot.double(), self.combinations.double())
        idx = torch.argmax(ips, -1)
        return idx

    def posterior_checks(
            self,
            order: T,  # M x J
            target: T,  # M x J
            sequences: T,  # M x E x T
            **statistics: dict[str: Callable[[BFFModel], float]]
    ):
        bffmodel = BFFModel(
            stimulus_order=order,
            target_stimulus=target,
            sequences=sequences,
            **self.settings,
            **self.prior,
            **self.dimensions,
        )

        observed = {sname: [] for sname in statistics.keys()}
        sampled = {sname: [] for sname in statistics.keys()}

        for sample_idx in range(self.n_samples):
            self.update_model(bffmodel, sample_idx)

            # observed
            bffmodel.set(observations=sequences)
            bffmodel.variables["factor_processes"].data = \
                bffmodel.variables["factor_processes"].posterior_mean

            for sname, sfunc in statistics.items():
                observed[sname].append(sfunc(bffmodel))

            # sampled
            bffmodel.generate_local_variables()
            bffmodel.variables["observations"].generate()
            bffmodel.variables["factor_processes"].data = \
                bffmodel.variables["factor_processes"].posterior_mean

            for sname, sfunc in statistics.items():
                sampled[sname].append(sfunc(bffmodel))

        return observed, sampled
