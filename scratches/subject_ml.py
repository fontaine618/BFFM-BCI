import sys
import os
import torch
import time
import pickle
import pandas as pd
import numpy as np
import scipy.special
import itertools as it
import torchmetrics
sys.path.insert(1, '/home/simon/Documents/BCI/src')
torch.set_default_tensor_type(torch.cuda.FloatTensor)
from source.data.k_protocol import KProtocol
from torch.distributions import Categorical
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')

# =============================================================================
# SETUP
dir_data = "/home/simon/Documents/BCI/K_Protocol/"
dir_chains = "/home/simon/Documents/BCI/experiments/subject/chains/"
dir_results = "/home/simon/Documents/BCI/experiments/subject/results/"
os.makedirs(dir_chains, exist_ok=True)
os.makedirs(dir_results, exist_ok=True)

# file
type = "TRN"
subject = "178" #str(sys.argv[1])
session = "001"
name = f"K{subject}_{session}_BCI_{type}"
filename = dir_data + name + ".mat"

# preprocessing
window = 800.0
bandpass_window = (0.1, 15.0)
bandpass_order = 2
downsample = 8


# experiment
seed = 0
train_reps = 15
# -----------------------------------------------------------------------------


# =============================================================================
# LOAD DATA
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
# -----------------------------------------------------------------------------




# =============================================================================
# TRAIN RF
X = eeg.stimulus.cpu().numpy()
y = eeg.stimulus_data["type"].values
# rf = RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0)
# rf.fit(X.reshape(X.shape[0], -1), y)
# rf.feature_importances_.reshape(25, 16).sum(0)
gb = GradientBoostingClassifier(n_estimators=100, max_depth=2, random_state=0, subsample=0.2)
gb.fit(X.reshape(X.shape[0], -1), y)
gb.feature_importances_.reshape(25, 16).sum(1)
# svc = SVC(kernel="linear", probability=True)
# svc.fit(X.reshape(X.shape[0], -1), y)
# -----------------------------------------------------------------------------



# =============================================================================
# TEST
trnstim = eeg.stimulus_data

nchars = eeg.stimulus_data["character"].nunique()
nreps = eeg.stimulus_data["repetition"].nunique()


# get prediction
# log_proba = rf.predict_log_proba(X.reshape(X.shape[0], -1))[:, 1]
log_proba = gb.predict_log_proba(X.reshape(X.shape[0], -1))[:, 1]
# log_proba = svc.predict_log_proba(X.reshape(X.shape[0], -1))[:, 1]
trnstim["log_proba"] = log_proba

# to key probabilities
log_prob = np.zeros((nchars, nreps, 36))
for c in trnstim["character"].unique():
    cum_log_proba = np.zeros((6, 6))
    for j, r in enumerate(trnstim["repetition"].unique()):
        idx = (trnstim["character"] == c) & (trnstim["repetition"] == r)
        log_proba = trnstim.loc[idx, "log_proba"].values
        stim = trnstim.loc[idx, "source"].values
        log_proba_mat = np.zeros((6, 6))
        for i, s in enumerate(stim):
            if s < 7:
                log_proba_mat[s-1, :] += log_proba[i]
            else:
                log_proba_mat[:, s-7] += log_proba[i]
        log_proba_mat -= scipy.special.logsumexp(log_proba_mat)
        cum_log_proba += log_proba_mat
        log_prob[c-1, j, :] = cum_log_proba.flatten().copy()

log_prob = torch.Tensor(log_prob)
log_prob -= torch.logsumexp(log_prob, dim=-1, keepdim=True)

Js = (6, 6)
combinations = torch.cartesian_prod(*[torch.arange(J) for J in Js])
to_add = torch.cumsum(torch.Tensor([0] + list(Js))[0:-1], 0).reshape(1, -1)
combinations = combinations + to_add
combinations = torch.nn.functional.one_hot(combinations.long(), sum(Js)).sum(1)

wide_pred = log_prob.argmax(2)
eeg.keyboard.flatten()[wide_pred.cpu()]
wide_pred_one_hot = combinations[wide_pred, :]
# -----------------------------------------------------------------------------






# =============================================================================
# METRICS

# entropy
entropy = Categorical(logits=log_prob).entropy()
mean_entropy = entropy.mean(0)

# accuracy & hamming
target_wide = eeg.target.view(nchars, nreps, -1)
accuracy = (wide_pred_one_hot == target_wide).all(2).double().mean(0)
hamming = (wide_pred_one_hot != target_wide).double().sum(2).mean(0) / 2

# binary cross-entropy
ips = torch.einsum("...i,ji->...j", target_wide.double(), combinations.double())
idx = torch.argmax(ips, -1)

target_char = torch.nn.functional.one_hot(idx, 36)
bce = - (target_char * log_prob).sum(2).mean(0)

# auc
target_char_int = torch.argmax(target_char, -1)
auc = torch.Tensor([
    torchmetrics.functional.classification.multiclass_auroc(
        preds=log_prob[:, c, :],
        target=target_char_int[:, c],
        num_classes=36,
        average="weighted"
    ) for c in range(nreps)
])

# save
df = pd.DataFrame({
    "hamming": hamming.cpu(),
    "acc": accuracy.cpu(),
    "mean_entropy": entropy.mean(0).abs().cpu(),
    "bce": bce.cpu(),
    "auroc": auc.cpu(),
    "dataset": name + "_test",
    "repetition": range(1, nreps + 1),
    "training_reps": train_reps,
    # "method": "RF",
    "method": "GB",
    # "method": "SVM",
}, index=range(1, nreps + 1))
print(df)
# df.to_csv(dir_results + f"K{subject}_trn{train_reps}_seed{seed}_rf.test")
# df.to_csv(dir_results + f"K{subject}_trn{train_reps}_seed{seed}_gb.test")
# df.to_csv(dir_results + f"K{subject}_trn{train_reps}_seed{seed}_svm.test")
# -----------------------------------------------------------------------------

del eeg


channels = ['F3', 'Fz', 'F4', 'T7', 'C3', 'Cz', 'C4', 'T8',
                       'CP3', 'CP4', 'P3', 'Pz', 'P4', 'PO7', 'PO8', 'Oz']

plt.cla()
plt.imshow(gb.feature_importances_.reshape(25, 16).T, cmap="Blues")
plt.yticks(range(16), channels)
plt.xticks([0, 6, 12, 18, 24], [0, 200, 400, 600, 800])
plt.grid(visible=False, axis="y")
plt.colorbar()
plt.tight_layout()
plt.title(f"K{subject}")
plt.show()