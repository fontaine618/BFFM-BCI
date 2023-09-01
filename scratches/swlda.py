import sys
import torch
sys.path.insert(1, '/home/simfont/Documents/BCI/src')
torch.set_default_tensor_type(torch.cuda.FloatTensor)
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-whitegrid")
from source.data.k_protocol import KProtocol
from source.swlda.swlda import swlda

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

whichchannels, restored_weights = swlda(
    responses=response,
    type=type,
    sampling_rate=1000,
    response_window=[0, response.shape[1]-1],
    decimation_frequency=1000,
    max_model_features=60,
    penter=0.1,
    premove=0.2
)

plt.plot(signal[10000:10100, 0])
plt.plot(filtered[10000:10100, 0].cpu())
plt.plot(smoothed[10000:10100, 0].cpu())
plt.show()

import mne
from mne.datasets import eegbci

eegbci.load_data(1, [6, 10, 14], "~/datasets")
# -----------------------------------------------------------------------------