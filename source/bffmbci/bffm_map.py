from typing import Tuple, Union
import torch
import pickle
import numpy as np
import scipy.linalg
import torch.nn.functional as F
import math
import arviz as az

from .utils import Kernel
from .bffm_init import bffm_initializer
from .utils.inverse_gamma import InverseGamma


class BFFModelMAP:

    def __init__(
            self,
            stimulus_order: torch.Tensor,
            target_stimulus: torch.Tensor,
            stimulus_window: int,
            stimulus_to_stimulus_interval: int,
            latent_dim: int,
            sequences: Union[torch.Tensor, None] = None,
            n_stimulus: Tuple[int, int] = (12, 2),
            n_sequences: int = 15 * 19,
            n_channels: int = 15,
            sparse: bool = False,
            shrinkage: str = "none",
            covariance: str = "dynamic_regression",
            mean_regression: bool = True,
            **kwargs
    ):
        self._dimensions = {
            "n_sequences": n_sequences,
            "n_timepoints": (n_stimulus[0] - 1) * stimulus_to_stimulus_interval + stimulus_window,
            "n_channels": n_channels,
            "n_stimulus": n_stimulus,
            "stimulus_window": stimulus_window,
            "stimulus_to_stimulus_interval": stimulus_to_stimulus_interval,
            "latent_dim": latent_dim
        }
        self.prior_parameters = {}
        self._initialize_prior_parameters(**kwargs)
        self.variables = {}
        self._settings = {}
        self._prepare_model(
            sequences=sequences,
            stimulus_order=stimulus_order,
            target_stimulus=target_stimulus,
            sparse=sparse,
            shrinkage=shrinkage,
            covariance=covariance,
            mean_regression=mean_regression
        )
        self._eval = False

    def _initialize_prior_parameters(self, **kwargs):
        prior_parameters = {
            "observation_variance": (1., 10.),
            "heterogeneities": 3.,
            "shrinkage_factor": (1., 3.),
            "kernel_gp_factor_processes": (0.7, 1., 1.),
            "kernel_tgp_factor_processes": (0.7, 0.5, 1.),
            "kernel_gp_loading_processes": (0.7, 1., 1.),
            "kernel_tgp_loading_processes": (0.7, 0.5, 1.),
            "kernel_gp_factor": (0.7, 1., 1.)
        }
        for k in prior_parameters.keys():
            if k in kwargs:
                prior_parameters[k] = kwargs[k]
        self.prior_parameters = prior_parameters

    def _prepare_model(
            self,
            sequences: Union[torch.Tensor, None],
            stimulus_order: torch.Tensor,
            target_stimulus: torch.Tensor,
            sparse: bool = False,
            shrinkage: str = "none",
            covariance: str = "dynamic_regression",
            mean_regression: bool = True
    ):
        self._settings = {
            "sparse": sparse,
            "shrinkage": shrinkage,
            "covariance": covariance,
            "mean_regression": mean_regression
        }
        self.variables = dict()
        self.data = {
            "sequences": sequences,
            "stimulus_order": stimulus_order,
            "target_stimulus": target_stimulus
        }
        self._kernels = dict()

        # local variables for dimensions
        N = self._dimensions["n_sequences"]
        T = self._dimensions["n_timepoints"]
        K = self._dimensions["latent_dim"]
        E = self._dimensions["n_channels"]
        w = self._dimensions["stimulus_window"]
        d = self._dimensions["stimulus_to_stimulus_interval"]

        # observation variance (log_scale)
        self.variables["log_observation_variance"] = torch.nn.Parameter(
            torch.zeros(E),
            requires_grad=True
        )

        # loadings
        self.variables["loadings"] = torch.nn.Parameter(
            torch.randn(E, K),
            requires_grad=True
        )
        self.variables["heterogeneities"] = torch.nn.Parameter(
            torch.ones(E, K),
            requires_grad=True
        )

        # scaling processes
        self.variables["scaling_process.nontarget_signal"] = torch.nn.Parameter(
            torch.zeros(K, w),
            requires_grad=True
        )
        self.variables["scaling_process.target_signal"] = torch.nn.Parameter(
            torch.zeros(K, w),
            requires_grad=True
        )
        self.variables["scaling_process.logit_mixing_signal"] = torch.nn.Parameter(
            torch.zeros(K, w),
            requires_grad=True
        )

        # factor processes
        self.variables["factor_process.nontarget_signal"] = torch.nn.Parameter(
            torch.zeros(K, w),
            requires_grad=True
        )
        self.variables["factor_process.target_signal"] = torch.nn.Parameter(
            torch.zeros(K, w),
            requires_grad=True
        )
        self.variables["factor_process.logit_mixing_signal"] = torch.nn.Parameter(
            torch.zeros(K, w),
            requires_grad=True
        )

        # mean factor processes
        self.variables["mean_factor_processes"] = torch.nn.Parameter(
            torch.zeros(N, K, T),
            requires_grad=False
        )

        # factor processes
        self.variables["factor_processes"] = torch.nn.Parameter(
            torch.zeros(N, K, T),
            requires_grad=False
        )

        # kernels
        p = self.prior_parameters["kernel_gp_factor"]
        tmat = _build_kernel_matrix(T, p[1], p[2], p[0])
        self._kernels["factors"] = Kernel.from_covariance_matrix(tmat)
        p = self.prior_parameters["kernel_gp_factor_processes"]
        tmat = _build_kernel_matrix(w, p[1], p[2], p[0])
        self._kernels["factor_signals"] = Kernel.from_covariance_matrix(tmat)
        p = self.prior_parameters["kernel_tgp_factor_processes"]
        tmat = _build_kernel_matrix(w, p[1], p[2], p[0])
        self._kernels["factor_mixing"] = Kernel.from_covariance_matrix(tmat)
        p = self.prior_parameters["kernel_gp_loading_processes"]
        tmat = _build_kernel_matrix(w, p[1], p[2], p[0])
        self._kernels["scaling_signals"] = Kernel.from_covariance_matrix(tmat)
        p = self.prior_parameters["kernel_tgp_loading_processes"]
        tmat = _build_kernel_matrix(w, p[1], p[2], p[0])
        self._kernels["scaling_mixing"] = Kernel.from_covariance_matrix(tmat)

        # variants
        if covariance == "dynamic_regression":
            pass
        elif covariance == "dynamic":
            self.variables["scaling_process.target_signal"].requires_grad = False
            self.variables["scaling_process.target_signal"].data.zero_()
            self.variables["scaling_process.logit_mixing_signal"].requires_grad = False
            self.variables["scaling_process.logit_mixing_signal"].data.fill_(-100.)
        elif covariance == "static":
            self.variables["scaling_process.nontarget_signal"].requires_grad = False
            self.variables["scaling_process.nontarget_signal"].data.zero_()
            self.variables["scaling_process.target_signal"].requires_grad = False
            self.variables["scaling_process.target_signal"].data.zero_()
            self.variables["scaling_process.logit_mixing_signal"].requires_grad = False
            self.variables["scaling_process.logit_mixing_signal"].data.fill_(-100.)
        else:
            raise ValueError(f"Unknown covariance type: {covariance}")
        if mean_regression:
            pass
        else:
            self.variables["factor_process.target_signal"].requires_grad = False
            self.variables["factor_process.target_signal"].data.zeros_()
            self.variables["factor_process.logit_mixing_signal"].requires_grad = False
            self.variables["factor_process.logit_mixing_signal"].data.fill_(-100.)

    def _processes_from_signals(self, nontarget, target, logit_mixing):
        mixing = torch.sigmoid(logit_mixing)
        target = nontarget + mixing * (target - nontarget)
        return nontarget, target

    def _superposition(self, nontarget, target):
        N = self._dimensions["n_sequences"]
        T = self._dimensions["n_timepoints"]
        K = self._dimensions["latent_dim"]
        d = self._dimensions["stimulus_to_stimulus_interval"]
        J = self._dimensions["n_stimulus"][0]
        w = self._dimensions["stimulus_window"]
        time = torch.arange(0, T)
        W = self.data["stimulus_order"]
        Y = self.data["target_stimulus"]
        nontarget = nontarget.unsqueeze(-1).unsqueeze(0)
        target = target.unsqueeze(-1).unsqueeze(0)
        Y = Y.unsqueeze(1).unsqueeze(1)
        p_in = ((1 - Y) * nontarget + Y * target)  # N x K x w x J
        shift_n = (time - W.unsqueeze(-1) * d).unsqueeze(1).long()  # N x 1 x J x T
        which_n = (shift_n >= 0) * (shift_n < w)  # N x 1 x J x T
        shift_n = torch.where(which_n, shift_n, w).expand(N, K, J, T)  # w is a bin value
        p_in = torch.cat([p_in, torch.zeros(N, K, 1, J)], 2)  # N x K x w+1 x J
        value = torch.gather(p_in.movedim(3, 2), 3, shift_n).sum(2)
        return value

    def _scaling_processes(self):
        nontarget = self.variables["scaling_process.nontarget_signal"]
        target = self.variables["scaling_process.target_signal"]
        logit_mixing = self.variables["scaling_process.logit_mixing_signal"]
        nontarget, target = self._processes_from_signals(nontarget, target, logit_mixing)
        return nontarget, target

    def _factor_processes(self):
        nontarget = self.variables["factor_process.nontarget_signal"]
        target = self.variables["factor_process.target_signal"]
        logit_mixing = self.variables["factor_process.logit_mixing_signal"]
        nontarget, target = self._processes_from_signals(nontarget, target, logit_mixing)
        return nontarget, target

    def _scaling_superposition(self):
        nontarget, target = self._scaling_processes()
        value = self._superposition(nontarget, target)
        return value.exp()

    def _factor_superposition(self):
        nontarget, target = self._factor_processes()
        value = self._superposition(nontarget, target)
        return value

    def _mean(self):
        N = self._dimensions["n_sequences"]
        T = self._dimensions["n_timepoints"]
        K = self._dimensions["latent_dim"]

        L = self.variables["loadings"]
        scaling = self._scaling_superposition()
        self.variables["mean_factor_processes"] = self._factor_superposition()
        if not self._eval:
            z = torch.randn((N, K, T))
            chol = self._kernels["factors"].chol
            z = z @ chol.T
            self.variables["factor_processes"] = self.variables["mean_factor_processes"] + z
        mean = torch.einsum(
            "ek, nkt, nkt -> net",
            L, scaling, self.variables["factor_processes"]
        )

        return mean

    def _log_likelihood_per_sequence(self):
        mean = self._mean()
        sig2 = self.variables["log_observation_variance"].exp()
        y = self.data["sequences"]
        mvn = torch.distributions.MultivariateNormal(mean.movedim(2, 1), sig2.diag_embed())
        return mvn.log_prob(y.movedim(2, 1)).sum(-1)

    def _log_likelihood(self):
        return self._log_likelihood_per_sequence().sum()

    def _factor_log_proba_per_sequence(self):
        mean = self.variables["mean_factor_processes"]
        factor = self.variables["factor_processes"]
        cov = self._kernels["factors"].cov
        mvn = torch.distributions.MultivariateNormal(mean, cov)
        return mvn.log_prob(factor).sum(-1)

    def _factor_log_proba(self):
        return self._factor_log_proba_per_sequence().sum()

    def _prior_log_proba(self):
        log_proba = 0.
        log_proba += self._loadings_prior_log_proba()
        log_proba += self._scaling_process_prior_log_proba()
        log_proba += self._factor_process_prior_log_proba()
        log_proba += self._observation_variance_prior_log_proba()
        return log_proba

    def _loadings_prior_log_proba(self):
        log_proba = 0.
        L = self.variables["loadings"]
        H = self.variables["heterogeneities"]
        log_proba += torch.distributions.Normal(0., H).log_prob(L).sum()
        gamma = self.prior_parameters["heterogeneities"]
        log_proba += InverseGamma(gamma / 2, gamma / 2).log_prob(H).sum()
        return log_proba

    def _scaling_process_prior_log_proba(self):
        log_proba = 0.
        # nontarget
        X = self.variables["scaling_process.nontarget_signal"]
        cov = self._kernels["scaling_signals"].cov
        mvn = torch.distributions.MultivariateNormal(torch.zeros_like(X), cov)
        log_proba += mvn.log_prob(X).sum()
        # target
        X = self.variables["scaling_process.target_signal"]
        cov = self._kernels["scaling_signals"].cov
        mvn = torch.distributions.MultivariateNormal(torch.zeros_like(X), cov)
        log_proba += mvn.log_prob(X).sum()
        # mixing
        X = torch.sigmoid(self.variables["scaling_process.logit_mixing_signal"])
        cov = self._kernels["scaling_mixing"].cov
        mvn = torch.distributions.MultivariateNormal(torch.zeros_like(X), cov)
        log_proba += mvn.log_prob(X).sum()
        return log_proba

    def _factor_process_prior_log_proba(self):
        log_proba = 0.
        # nontarget
        X = self.variables["factor_process.nontarget_signal"]
        cov = self._kernels["factor_signals"].cov
        mvn = torch.distributions.MultivariateNormal(torch.zeros_like(X), cov)
        log_proba += mvn.log_prob(X).sum()
        # target
        X = self.variables["factor_process.target_signal"]
        cov = self._kernels["factor_signals"].cov
        mvn = torch.distributions.MultivariateNormal(torch.zeros_like(X), cov)
        log_proba += mvn.log_prob(X).sum()
        # mixing
        X = torch.sigmoid(self.variables["factor_process.logit_mixing_signal"])
        cov = self._kernels["factor_mixing"].cov
        mvn = torch.distributions.MultivariateNormal(torch.zeros_like(X), cov)
        log_proba += mvn.log_prob(X).sum()
        return log_proba

    def _observation_variance_prior_log_proba(self):
        log_proba = 0.
        X = self.variables["log_observation_variance"]
        a, b = self.prior_parameters["observation_variance"]
        log_proba += InverseGamma(a, b).log_prob(X.exp()).sum()
        return log_proba

    def _joint_log_proba(self):
        llk = self._log_likelihood()
        prior = self._prior_log_proba()
        factor = self._factor_log_proba()
        return llk + prior + factor

    def fit(self, lr=0.1, max_iter=1000, tol=1e-6):
        optimizer = torch.optim.Adam(self.variables.values(), lr=lr)
        prevllk = self._log_likelihood().item()
        for i in range(max_iter):
            optimizer.zero_grad()
            newllk = self._joint_log_proba()
            (-newllk).backward()
            optimizer.step()
            if i % 10 == 0:
                print(f"iter {i}: {newllk.item()}")
            if abs(newllk - prevllk) / abs(prevllk) < tol:
                break
            prevllk = newllk.item()

    def initialize(self, reverse=False, weighted=False):
        # use WFA to find loadings and variance
        # the estimated factors will be used to initialize the processes below
        loadings, observation_variance, factors = bffm_initializer(
            target_stimulus=self.data["target_stimulus"],
            stimulus_order=self.data["stimulus_order"],
            sequences=self.data["sequences"],
            latent_dim=self._dimensions["latent_dim"],
            stimulus_window=self._dimensions["stimulus_window"],
            stimulus_to_stimulus_interval=self._dimensions["stimulus_to_stimulus_interval"],
            weighted=weighted
        )
        if reverse:
            loadings = loadings.flip(1)
            factors = factors.flip(1)
        # sparsify loadings
        max_loadings = loadings.abs().max(0).values
        threshold = 0.25 * max_loadings
        loadings = torch.where(loadings.abs() > threshold, loadings, torch.zeros_like(loadings))
        # smooth out the factors
        smat = scipy.linalg.toeplitz(0.5 ** np.arange(factors.shape[2]))
        smat = torch.Tensor(smat)
        smat = smat / smat.sum(0)
        sfactors = factors @ smat
        # put values into variables
        self.variables["loadings"].data = loadings
        self.variables["log_observation_variance"].data = observation_variance.log()
        self.variables["factor_processes"].data = sfactors.clone()

    def eval(self):
        # self._eval = True
        # reset computation graph,
        # no gradients for all variables except factor processes
        for k, v in self.variables.items():
            if k != "factor_processes":
                v.detach_()
        # self.variables["factor_processes"] = torch.nn.Parameter(
        #     self.variables["factor_processes"].data,
        #     requires_grad=True
        # )

    def update_data(
            self,
            stimulus_order: torch.Tensor,
            target_stimulus: torch.Tensor,
            sequences: torch.Tensor
    ):
        self.data = {
            "sequences": sequences,
            "stimulus_order": stimulus_order,
            "target_stimulus": target_stimulus
        }
        self._dimensions["n_sequences"] = sequences.shape[0]

        self._prepare_local_variables()

    def _prepare_local_variables(self):
        N = self._dimensions["n_sequences"]
        T = self._dimensions["n_timepoints"]
        K = self._dimensions["latent_dim"]
        # mean factor processes
        self.variables["mean_factor_processes"] = torch.nn.Parameter(
            torch.zeros(N, K, T),
            requires_grad=False
        )
        # factor processes
        self.variables["factor_processes"] = torch.nn.Parameter(
            torch.zeros(N, K, T),
            requires_grad=False
        )

    @property
    def combinations(self):
        Js = (6, 6)
        combinations = torch.cartesian_prod(*[torch.arange(J) for J in Js])
        to_add = torch.cumsum(torch.Tensor([0] + list(Js))[0:-1], 0).reshape(1, -1)
        combinations = combinations + to_add
        combinations = torch.nn.functional.one_hot(combinations.long(), sum(Js)).sum(1)
        return combinations  # L x J

    def _expand_data_for_prediction(self):
        N = self._dimensions["n_sequences"]
        T = self._dimensions["n_timepoints"]
        E = self._dimensions["n_channels"]
        L = 36
        J = 12
        # store data
        self.original_data = {
            "sequences": self.data["sequences"].clone(),
            "stimulus_order": self.data["stimulus_order"].clone(),
            "target_stimulus": self.data["target_stimulus"].clone()
        }
        # sequences
        sequences = self.data["sequences"]  # N x E x T
        sequences = sequences.unsqueeze(1).expand(N, L, E, T)  # N x L x E x T
        sequences = sequences.reshape(N * L, E, T)
        self.data["sequences"] = sequences
        # stimulus order
        stimulus_order = self.data["stimulus_order"]  # N x J
        stimulus_order = stimulus_order.unsqueeze(1).expand(N, L, J)  # N x L x J
        stimulus_order = stimulus_order.reshape(N * L, J)
        self.data["stimulus_order"] = stimulus_order
        # target stimulus
        self.data["target_stimulus"] = self.combinations.repeat(N, 1)  # N x L x J
        self._dimensions["n_sequences"] = N * L

        self._prepare_local_variables()

    def _reset_original_data(self):
        self.data = {
            "sequences": self.original_data["sequences"].clone(),
            "stimulus_order": self.original_data["stimulus_order"].clone(),
            "target_stimulus": self.original_data["target_stimulus"].clone()
        }
        self._dimensions["n_sequences"] = self.original_data["sequences"].shape[0]

        self._prepare_local_variables()

    # def predict(self):
    #     self._expand_data_for_prediction()
    #     self.fit()
    #     log_proba = self._log_likelihood_per_sequence()
    #     log_proba += self._factor_log_proba_per_sequence()
    #     N = self.original_data["sequences"].shape[0]
    #     L = 36
    #     log_proba = log_proba.reshape(N, L)
    #     self._reset_original_data()
    #     return log_proba

    def export_variables(self):
        variables = {
            "observation_variance": self.variables["log_observation_variance"].exp().detach().clone(),
            "loadings": self.variables["loadings"].detach().clone(),
            "heterogeneities": self.variables["heterogeneities"].detach().clone(),
            "smgp_factors.nontarget_process": self.variables["factor_process.nontarget_signal"].detach().clone(),
            "smgp_factors.target_process": self.variables["factor_process.target_signal"].detach().clone(),
            "smgp_factors.mixing_process": torch.sigmoid(
                self.variables["factor_process.logit_mixing_signal"].detach().clone()),
            "smgp_scaling.nontarget_process": self.variables["scaling_process.nontarget_signal"].detach().clone(),
            "smgp_scaling.target_process": self.variables["scaling_process.target_signal"].detach().clone(),
            "smgp_scaling.mixing_process": torch.sigmoid(
                self.variables["scaling_process.logit_mixing_signal"].detach().clone())
        }
        return variables

    def predict(self, n_samples=100):
        with torch.no_grad():
            combinations = self.combinations
            N = self._dimensions["n_sequences"]
            L = combinations.shape[0]
            J = 12
            log_proba = torch.zeros(N, L, n_samples)
            self.eval()
            for l in range(L):
                print(f"predicting combination {l+1}/{L}")
                self.data["target_stimulus"] = combinations[l, :].expand(N, J)
                for n in range(n_samples):
                    log_proba[:, l, n] = self._log_likelihood_per_sequence()
                # self.fit()
                # log_proba[:, l] = self._log_likelihood_per_sequence() + self._factor_log_proba_per_sequence()
            # log_prob = torch.zeros(N, L)
            # for l in range(L):
            #     for i in range(N):
            #         log_prob_ni = log_proba[i, l, :]
            #         log_weights = torch.Tensor(az.psislw(-log_prob_ni.cpu().numpy(), reff=1.)[0])
            #         log_weights += log_prob_ni
            #         log_prob[i, l] = torch.logsumexp(log_weights, dim=0)
            # log_proba = log_prob
            log_proba = -torch.logsumexp(-log_proba, dim=-1) + math.log(log_proba.shape[-1])
            # log_proba = torch.logsumexp(log_proba, dim=-1) - math.log(log_proba.shape[-1])
            return log_proba



class DynamicRegressionCovarianceRegressionMeanMAP(BFFModelMAP):

    def __init__(self, **kwargs):
        kwargs["covariance"] = "dynamic_regression"
        kwargs["mean_regression"] = True
        super().__init__(**kwargs)


class DynamicCovarianceRegressionMeanMAP(BFFModelMAP):

    def __init__(self, **kwargs):
        kwargs["covariance"] = "dynamic"
        kwargs["mean_regression"] = True
        super().__init__(**kwargs)


class StaticCovarianceRegressionMeanMAP(BFFModelMAP):

    def __init__(self, **kwargs):
        kwargs["covariance"] = "static"
        kwargs["mean_regression"] = True
        super().__init__(**kwargs)


class DynamicRegressionCovarianceStaticMeanMAP(BFFModelMAP):

    def __init__(self, **kwargs):
        kwargs["covariance"] = "dynamic_regression"
        kwargs["mean_regression"] = False
        super().__init__(**kwargs)


def _build_kernel_matrix(dim, scale=1., power=1., correlation=0.8):
    t = torch.arange(dim).float()
    diff = t.unsqueeze(0) - t.unsqueeze(1)
    diff = diff.abs() ** power
    kernel = correlation ** diff
    kernel = kernel * scale
    return kernel
