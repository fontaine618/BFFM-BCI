import sys
import os
import torch
import time
import pickle

sys.path.insert(1, '/home/simon/Documents/BCI/src')
torch.set_default_tensor_type(torch.cuda.FloatTensor)

from source.bffmbci import BFFMResults
import matplotlib.pyplot as plt
import pandas as pd
from torch.distributions import Categorical
from source.bffmbci.bffm import BFFModel

plt.style.use("seaborn-v0_8-whitegrid")

from source.data.k_protocol import KProtocol

# =============================================================================
# SETUP
dir = "/home/simon/Documents/BCI/experiments/tuning/"
dir_data = "/home/simon/Documents/BCI/K_Protocol/"
dir_chains = dir + "chains/"
dir_results = dir + "predict/"
dir_figures = dir + "figures/"

type = "TRN"
subject = "114"
session = "001"
name = f"K{subject}_{session}_BCI_{type}"
filename = dir_data + name + ".mat"
window = 800.0
bandpass_window = (0.1, 15.0)
bandpass_order = 2
downsample = 8
factor_samples = 10
factor_processes_method = "analytical"
sample_mean = "harmonic"
which_first = "sample"
return_cumulative = True
n_samples = 15
nchars = 19

K = 8
nreps = 7
seed = 0
cor = 0.5
shrinkage = 5.
file = f"seed{seed}_nreps{nreps}_cor{cor}_shrinkage{shrinkage}.chain"


out = []
# -----------------------------------------------------------------------------



# =============================================================================
# testing accuracy
eeg = KProtocol(
    filename=filename,
    type=type,
    subject=subject,
    session=session,
    window=window,
    bandpass_window=bandpass_window,
    bandpass_order=bandpass_order,
    downsample=downsample,
)
eeg = eeg.repetitions(range(nreps + 1, 16))
nreps = 15 - nreps

order = eeg.stimulus_order
sequence = eeg.sequence
target = eeg.target
character_idx = eeg.character_idx

for c in [None, 0, 1, 2, 3, 4, 5, 6, 7, ]:
    torch.cuda.empty_cache()
    results = BFFMResults.from_files(
        [dir_chains + file],
        warmup=10_000,
        thin=1
    )
    self = results.to_predict(n_samples=n_samples)

    llk_long, chars = self.predict(
        order=order,
        sequence=sequence,
        factor_samples=factor_samples,
        character_idx=character_idx,
        factor_processes_method=factor_processes_method,
        drop_component=c
    )

    log_prob = self.aggregate(
        llk_long,
        sample_mean=sample_mean,
        which_first=which_first
    )

    wide_pred_one_hot = self.get_predictions(log_prob, True)

    entropy = Categorical(logits=log_prob).entropy()

    target_ = target[::nreps, :].unsqueeze(1).repeat(1, nreps, 1)
    hamming = (wide_pred_one_hot != target_).double().sum(2).sum(0) / 2
    acc = (wide_pred_one_hot == target_).all(2).double().sum(0)

    target36 = torch.nn.functional.one_hot(self.one_hot_to_combination_id(target_), 36)
    bce = (target36 * log_prob).sum(-1).mean(0)

    df = pd.DataFrame({
        "hamming": hamming.cpu(),
        "acc": acc.cpu(),
        "max_entropy": entropy.max(0)[0].abs().cpu(),
        "mean_entropy": entropy.mean(0).abs().cpu(),
        "min_max_proba": log_prob.max(2)[0].min(0)[0].cpu().exp(),
        "bce": bce.cpu(),
        "dataset": "testing",
        "training_reps": nreps,
        "repetition": range(1, nreps + 1),
        "cor": cor,
        "sample_mean": sample_mean,
        "which_first": which_first,
        "drop_component": c
    }, index=range(1, nreps + 1))
    print(df)

    out.append(df)

df = pd.concat(out)
os.makedirs(dir_results, exist_ok=True)
df.to_csv(dir_results + f"drop_component.csv")
print(df)
# -----------------------------------------------------------------------------


# =============================================================================
# Plot results
# figures: acc, bce, hamming
# xaxis: repetition
# yaxis: acc, bce, hamming
# subset to testing set
# curves (color) are the drop_component
df = pd.read_csv(dir_results + f"drop_component.csv", index_col=0)
df = df[df["dataset"] == "testing"]
df["drop_component"] = df["drop_component"].astype(str)
# nan replace by None
df["drop_component"] = df["drop_component"].replace("nan", "None")

# select the metrics and convert to long format
metrics = ["acc", "bce"]
df = df.melt(
    id_vars=["repetition", "drop_component"],
    value_vars=metrics,
    var_name="metric",
    value_name="value"
)

import seaborn as sns
sns.relplot(
    data=df,
    x="repetition",
    y="value",
    hue="drop_component",
    style="drop_component",
    row="metric",
    kind="line",
    legend="full",
    palette="viridis",
    facet_kws={"legend_out": True, "sharey": False},
    height=4,
)
plt.savefig(dir_figures + "drop_component.pdf", bbox_inches="tight")
# -----------------------------------------------------------------------------



