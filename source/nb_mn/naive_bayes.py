import torch
from .matrix_normal import MatrixNormal


class NaiveBayesMatrixNormal:

    def __init__(self, T, E):
        self.T = T
        self.E = E
        self.models = [MatrixNormal(T, E) for _ in range(2)]
        self.prior = torch.zeros(2)

    def fit(self, X, y):
        for i in range(2):
            idx = y == i
            self.models[i].fit(X[idx])
            self.prior[i] = idx.float().mean()
        return self

    def log_prob(self, X):
        log_probs = torch.vstack([
            self.models[i].log_prob(X) + self.prior.log()[i]
            for i in range(2)
        ]).T
        return log_probs

    def predict(self, X):
        log_prob = self.log_prob(X)
        return torch.nn.functional.log_softmax(log_prob, dim=1)[:, 1]