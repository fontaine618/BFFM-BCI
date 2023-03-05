import torch
from collections import defaultdict
"""List of all metrics that can be computed from the MCMC results.

Format: dict[variable: dict[metric: function]
"""
metrics: dict[str, dict[str, callable]] = defaultdict(
    dict,
    {
        "loadings": {
            "columnwise_2norm":
                lambda x, y: torch.norm(x - y, dim=0),
            "columnwise_2norm_rel":
                lambda x, y: torch.norm(x - y, dim=0) / torch.norm(y, dim=0),
            "frobenius":
                lambda x, y: torch.norm(x - y, "fro"),
            "frobenius_rel":
                lambda x, y: torch.norm(x - y, "fro") / torch.norm(y, "fro"),
            "columnwise_cosine_similarity":
                lambda x, y: torch.cosine_similarity(x, y, dim=0),
        },
        "loadings.inner_products": {
            "frobenius":
                lambda x, y: torch.norm(x - y, "fro"),
            "frobenius_rel":
                lambda x, y: torch.norm(x - y, "fro") / torch.norm(y, "fro"),
        },
        "loadings.rank_one": {
            "columnwise_frobenius":
                lambda x, y: torch.norm(x - y, "fro", dim=[1, 2]),
            "columnwise_frobenius_rel":
                lambda x, y: torch.norm(x - y, "fro", dim=[1, 2]) /
                             torch.norm(y, "fro", dim=[1, 2]),
        },
        "loadings.projection": {
            "frobenius":
                lambda x, y: torch.norm(x - y, "fro"),
            "frobenius_rel":
                lambda x, y: torch.norm(x - y, "fro") / torch.norm(y, "fro"),
        },
        "loadings.times_shrinkage": {
            "columnwise_2norm":
                lambda x, y: torch.norm(x - y, dim=0),
            "columnwise_2norm_rel":
                lambda x, y: torch.norm(x - y, dim=0) / torch.norm(y, dim=0),
            "frobenius":
                lambda x, y: torch.norm(x - y, "fro"),
            "frobenius_rel":
                lambda x, y: torch.norm(x - y, "fro") / torch.norm(y, "fro"),
        },
        "loadings.norm_one": {
            "columnwise_2norm":
                lambda x, y: torch.norm(x - y, dim=0),
            "columnwise_2norm_rel":
                lambda x, y: torch.norm(x - y, dim=0) / torch.norm(y, dim=0),
            "frobenius":
                lambda x, y: torch.norm(x - y, "fro"),
            "frobenius_rel":
                lambda x, y: torch.norm(x - y, "fro") / torch.norm(y, "fro"),
        }
    })