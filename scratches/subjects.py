import sys
import torch
import scipy.io
from source.data.k_protocol import KProtocol, _array_to_dict

# =============================================================================
# SETUP
type = "TRN"
subject = "111"
session = "001"
name = f"K{subject}_{session}_BCI_{type}"
dir_data = "/home/simon/Documents/BCI/K_Protocol/"

# preprocessing
window = 800.0
bandpass_window = (0.1, 15.0)
bandpass_order = 2
downsample = 8
# -----------------------------------------------------------------------------


# =============================================================================
# LOAD DATA
filename = dir_data + name + ".mat"
mat_obj = scipy.io.loadmat(filename)["Data"][0][0][0][0][0]
signal = mat_obj[0]
states = _array_to_dict(mat_obj[1][0][0])
parameters = _array_to_dict(mat_obj[2][0][0])
parameters = {k: _array_to_dict(v[0][0]) for k, v in parameters.items()}

mat_obj = scipy.io.loadmat(filename)
# -----------------------------------------------------------------------------