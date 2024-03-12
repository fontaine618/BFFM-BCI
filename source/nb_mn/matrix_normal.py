import torch
import math


class MatrixNormal:
    """TxE matrix normal distribution"""

    def __init__(self, T, E):
        self.T = T
        self.E = E
        self.mean = torch.zeros(E, T)
        self.time_cov = torch.eye(T)
        self.time_icov = torch.eye(T)
        self.channel_cov = torch.eye(E)
        self.channel_icov = torch.eye(E)

    def log_prob(self, X):
        """Expects [N, T, E]"""
        R = X - self.mean.unsqueeze(0)
        R1 = R @ self.channel_icov
        R2 = self.time_icov @ R
        trace = (R1 * R2).sum((1, 2))
        log_det = self.time_cov.logdet() * self.E + self.channel_cov.logdet() * self.T
        log_pi = math.log(2 * math.pi) * self.E * self.T
        log_prob = - 0.5 * (trace + log_det + log_pi)
        return log_prob

    def fit(self, X):
        """Expects [N, T, E]"""
        self.mean = X.mean(0)
        log_prob = self.log_prob(X).sum()
        R = X - self.mean.unsqueeze(0)
        for i in range(1000):
            self._update_time_cov(R)
            self._update_channel_col(R)
            log_prob_new = self.log_prob(X).sum()
            print(f"iter {i}, log_prob {log_prob_new}")
            if abs(log_prob_new - log_prob) < 1e-3:
                break
            log_prob = log_prob_new
        return self

    def _update_time_cov(self, R):
        """Expects [N, T, E]"""
        time_cov = torch.einsum(
            "nse, ef, ntf -> st",
            R, self.channel_icov, R
        )
        time_cov /= R.shape[0] * self.E
        time_cov += 1e-4 * torch.eye(self.T)
        # standardize
        time_cov /= time_cov.diag().max()
        self.time_cov = time_cov
        self.time_icov = time_cov.inverse()
        return self

    def _update_channel_col(self, R):
        """Expects [N, T, E]"""
        channel_cov = torch.einsum(
            "nse, st, ntf -> ef",
            R, self.time_icov, R
        )
        channel_cov /= R.shape[0] * self.T
        channel_cov += 1e-4 * torch.eye(self.E)
        self.channel_cov = channel_cov
        self.channel_icov = channel_cov.inverse()
        return self

