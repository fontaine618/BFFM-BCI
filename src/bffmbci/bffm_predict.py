import torch
from .utils import Kernel
import scipy
import numpy as np
from torch.distributions.multivariate_normal import MultivariateNormal


class BFFMPredict:
    """
    Excpexted:
    loadings (N x E x K)
    observation_variance (N x E)
    smgp_factors.nontarget_process (N x K x T)
    smgp_factors.target_signal (N x K x T)
    smgp_scaling.nontarget_process (N x K x T)
    smgp_scaling.target_signal (N x K x T)
    """

    def __init__(
            self,
            variables: dict[str, torch.Tensor],
            dimensions: dict,
            prior: dict
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

    @property
    def n_samples(self):
        return self.variables["loadings"].shape[0]

    @property
    def combinations(self) -> torch.Tensor:
        Js = self.dimensions["n_stimulus"]
        combinations = torch.cartesian_prod(*[torch.arange(J) for J in Js])
        to_add = torch.cumsum(torch.Tensor([0] + list(Js))[0:-1], 0).reshape(1, -1)
        combinations = combinations + to_add
        combinations = torch.nn.functional.one_hot(combinations.long(), sum(Js)).sum(1)
        return combinations  # L x J

    def superposition(
            self,
            order: torch.Tensor,  # M x J
            combinations: torch.Tensor,  # L x J
            sample_idx: int,
            b0: torch.Tensor,
            b1: torch.Tensor,
    ):
        # M is the number of sequences to predict, L is the number of combinations
        b0 = b0[sample_idx, :, :]
        b1 = b1[sample_idx, :, :]
        K, _ = b0.shape
        M, J = order.shape
        L, _ = combinations.shape
        d = self.dimensions["stimulus_to_stimulus_interval"]
        W = self.dimensions["stimulus_window"]
        T = (J - 1) * d + W

        w = order.unsqueeze(1).unsqueeze(1).unsqueeze(-1)  # M x 1 x 1 x J x 1
        y = combinations.unsqueeze(0).unsqueeze(2).unsqueeze(-1)  # 1 x L x 1 x J x 1

        b0.unsqueeze_(0).unsqueeze_(0).unsqueeze_(-2)  # 1 x 1 x K x 1 x T
        b1.unsqueeze_(0).unsqueeze_(0).unsqueeze_(-2)  # 1 x 1 x K x 1 x T

        time = torch.arange(0, T).reshape(1, 1, 1, 1, -1)  # 1 x 1 x 1 x 1 x T

        p_in = (1-y) * b0 + y * b1  # M x L x K x J x T
        shift_n = (time - w * d).long()  # M x L x 1 x J x T
        which_n = (shift_n >= 0) * (shift_n < W)
        shift_n = torch.where(which_n, shift_n, W)  # when outside, we label it W
        p_in = torch.cat([p_in, torch.zeros(1, L, K, J, 1)], 4)  # add the 0 bin for when shift_n=W
        shift_n = shift_n.repeat(1, L, K, 1, 1)  # M x L x K x J x T
        p_in = p_in.repeat(M, 1, 1, 1, 1)  # M x L x K x J x T
        value = torch.gather(p_in, 4, shift_n).sum(3)  # M x L x K x T
        return value  # M x L x K x T

    def scaling_processes(
            self,
            order: torch.Tensor,  # M x J
            combinations: torch.Tensor,  # L x J
            sample_idx: int,
    ):
        return self.superposition(
            order,
            combinations,
            sample_idx,
            self.variables["smgp_scaling.nontarget_process"],
            self.variables["smgp_scaling.target_signal"],
        )

    def mean_factor_processes(
            self,
            order: torch.Tensor,  # M x J
            combinations: torch.Tensor,  # L x J
            sample_idx: int,
    ):
        return self.superposition(
            order,
            combinations,
            sample_idx,
            self.variables["smgp_factors.nontarget_process"],
            self.variables["smgp_factors.target_signal"],
        )

    def factor_processes(
            self,
            mean_factor_processes: torch.Tensor,  # M x L x K x T
            factor_samples: int,
    ):
        if factor_samples <= 0:
            return mean_factor_processes.unsqueeze(-2)
        dims = list(mean_factor_processes.shape[:-1]) + [factor_samples]
        # for some reason, CUDA doesn't like this shape?
        z = torch.stack([
            self._dist.sample(dims[1:])
            for _ in range(dims[0])
        ], 0)
        # z = self._dist.sample(dims)
        value = mean_factor_processes.unsqueeze(-2) + z
        return value  # M x L x K x B x T

    def mean(
            self,
            scaling_processes: torch.Tensor,  # M x L x K x T
            factor_processes: torch.Tensor,  # M x L x K x B x T
            sample_idx: int
    ):
        processes = (scaling_processes.unsqueeze(-2) * factor_processes)  # M x L x K x B x T
        L = self.variables["loadings"][sample_idx, :, :]  # E x K
        mean = torch.einsum("mlkbt,ek->mlbet", processes, L)  # M x L x B x E x T
        return mean

    def log_prob(
            self,
            mean: torch.Tensor,  # M x L x B x E x T
            sequence: torch.Tensor,  # M x E x T,
            sample_idx: int
    ):
        dist = MultivariateNormal(
            loc=torch.zeros(sequence.shape[1]),
            covariance_matrix=torch.diag(self.variables["observation_variance"][sample_idx, :])
        )
        B = mean.shape[2]
        diff = sequence.unsqueeze(1).unsqueeze(1) - mean  # M x L x B x E x T
        diff = diff.permute(0, 1, 2, 4, 3)  # M x L x B x T x E
        # for some reason, CUDA is unhappy with doing this in one go
        log_prob = torch.stack([
            dist.log_prob(diff[:, :, :, t, :])  # M x L x B x E
            for t in range(diff.shape[-2])
        ], 3)  # M x L x B x T
        log_prob = torch.logsumexp(log_prob, 2, keepdim=False)  # M x L x T
        log_prob -= torch.log(torch.Tensor([B]))
        return torch.logsumexp(log_prob, 2, keepdim=False)  # M x L

    def _predict_idx(
            self,
            order: torch.Tensor,  # M x J
            sequence: torch.Tensor,  # M x E x T,
            factor_samples: int,
            sample_idx: int,
    ):
        scaling_processes = self.scaling_processes(order, self.combinations, sample_idx)
        mean_factor_processes = self.mean_factor_processes(order, self.combinations, sample_idx)
        factor_processes = self.factor_processes(mean_factor_processes, factor_samples)
        mean = self.mean(scaling_processes, factor_processes, sample_idx)
        log_prob = self.log_prob(mean, sequence, sample_idx)
        return log_prob  # M x L

    def predict(
            self,
            order: torch.Tensor,  # M x J
            sequence: torch.Tensor,  # M x E x T,
            factor_samples: int,
    ):
        N = self.n_samples
        log_probs = [
            self._predict_idx(order, sequence, factor_samples, sample_idx)
            for sample_idx in range(N)
        ]
        log_probs = torch.stack(log_probs, 2)  # M x L x N
        log_probs -= log_probs.logsumexp(1, keepdim=True)  # center == map to probabilities
        log_probs = torch.logsumexp(log_probs, 2, keepdim=False)  # M x L
        log_probs -= torch.log(torch.Tensor([N]))
        pred = torch.argmax(log_probs, 1)
        pred_one_hot = self.combinations[pred, :]
        return log_probs, pred_one_hot

    def one_hot_to_combination_id(self, one_hot: torch.Tensor):
        # TODO
        pass