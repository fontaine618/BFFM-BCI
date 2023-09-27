import sys
import torch
import numpy as np
import pandas as pd
sys.path.insert(1, '/home/simfont/Documents/BCI/src')
torch.set_default_tensor_type(torch.cuda.FloatTensor)
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-whitegrid")
from source.data.k_protocol import KProtocol
from source.swlda.swlda import swlda, swlda_predict

# =============================================================================
# SETUP
dir_data = "/home/simon/Documents/BCI/K_Protocol/"

type = "TRN"
subject = "114"
session = "001"
name = f"K{subject}_{session}_BCI_{type}"
filename = dir_data + name + ".mat"
window = 800.0
bandpass_window = (0.1, 15.0)
bandpass_order = 2
downsample = 8
nchars = 19
# -----------------------------------------------------------------------------


# =============================================================================
# load data
self = KProtocol(
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
# load data
response = self.stimulus.cpu().numpy()
type = self.stimulus_data["type"].values
trn = self.stimulus_data["repetition"] < 8
trn = self.stimulus_data.index[trn]
trnX = response[trn, ...]
trny = type[trn]
trnstim = self.stimulus_data.loc[trn]
tst = self.stimulus_data["repetition"] > 7
tst = self.stimulus_data.index[tst]
tstX = response[tst, ...]
tsty = type[tst]
tststim = self.stimulus_data.loc[tst]

whichchannels, restored_weights, bias = swlda(
    responses=trnX,
    type=trny,
    sampling_rate=1000,
    response_window=[0, response.shape[1]-1],
    decimation_frequency=1000,
    max_model_features=150,
    penter=0.1,
    premove=0.15
)

Bmat = torch.zeros((16, 25))
Bmat[restored_weights[:, 0]-1, restored_weights[:, 1]-1] = torch.Tensor(restored_weights[:, 3])
Bmat.cpu()


trn_pred, trn_agg_pred, trn_cum_pred = swlda_predict(Bmat.cpu(), trnX, trnstim, self.keyboard)
tst_pred, tst_agg_pred, tst_cum_pred = swlda_predict(Bmat.cpu(), tstX, tststim, self.keyboard)

rowtrue = self.stimulus_data.loc[(self.stimulus_data["source"]<7) & (self.stimulus_data["type"]==1)]
rowtrue = rowtrue.groupby(["character"]).head(1).sort_values(["character"]).reset_index()
rowtrue = rowtrue[["character", "source"]].rename(columns={"source": "row"})
coltrue = self.stimulus_data.loc[(self.stimulus_data["source"]>6) & (self.stimulus_data["type"]==1)]
coltrue = coltrue.groupby(["character"]).head(1).sort_values(["character"]).reset_index()
coltrue = coltrue[["character", "source"]].rename(columns={"source": "col"})
true = rowtrue.merge(coltrue, on=["character"], how="outer").reset_index()
true["char"] = self.keyboard[true["row"]-1, true["col"]-7]

# test metrics
tstdf = tst_cum_pred.join(true.set_index("character"), on="character", how="left", rsuffix="_true", lsuffix="_pred")
tstdf["hamming"] = (tstdf["row_pred"] == tstdf["row_true"]).astype(int) \
                    + (tstdf["col_pred"] == tstdf["col_true"]).astype(int)
tstdf["acc"] = (tstdf["char_pred"] == tstdf["char_true"]).astype(int)
tstdf = tstdf.groupby("repetition").agg({"hamming": "sum", "acc": "sum"})
tstdf["dataset"] = "testing"
tstdf.reset_index(inplace=True)
tstdf["repetition"] -= 7

# train metrics
trndf = trn_cum_pred.join(true.set_index("character"), on="character", how="left", rsuffix="_true", lsuffix="_pred")
trndf["hamming"] = (trndf["row_pred"] == trndf["row_true"]).astype(int) \
        + (trndf["col_pred"] == trndf["col_true"]).astype(int)
trndf["acc"] = (trndf["char_pred"] == trndf["char_true"]).astype(int)
trndf = trndf.groupby("repetition").agg({"hamming": "sum", "acc": "sum"})
trndf["dataset"] = "training"
trndf.reset_index(inplace=True)

# merge and save
df = pd.concat([trndf, tstdf], axis=0)
df["training_reps"] = 7
swlda_df = df
# -----------------------------------------------------------------------------