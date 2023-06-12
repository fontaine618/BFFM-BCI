import torch
from source.bffmbci.bffm import BFFModel
import matplotlib.pyplot as plt
plt.style.use("seaborn-v0_8-whitegrid")

torch.set_default_tensor_type(torch.cuda.FloatTensor)

latent_dim = 3
stimulus_window = 25
stimulus_to_stimulus_interval = 5
n_channels = 16
n_characters = 17
n_repetitions = 15
n_timepoints = 11 * stimulus_to_stimulus_interval + stimulus_window

torch.manual_seed(0)
model = BFFModel.generate_from_dimensions(
	latent_dim=latent_dim,
	n_channels=n_channels,
	stimulus_to_stimulus_interval=stimulus_to_stimulus_interval,
	stimulus_window=stimulus_window,
	n_stimulus=(12, 2),
	n_characters=n_characters,
	n_repetitions=n_repetitions,
	heterogeneities=3.,
	shrinkage_factor=(2., 5.),
	nonnegative_smgp=False,
	scaling_activation="e",
	kernel_gp_loading_processes=(0.99, 0.1, 1.)
)

Xtensor = model.variables["observations"].data  # (n_sequences, n_channels, n_timepoints)
X = torch.cat([
    Xtensor[i, :, :] for i in range(Xtensor.shape[0])
], 1).T
order = model.variables["sequence_data"].order.data
target = model.variables["sequence_data"].target.data
ylist = []
for i in range(order.shape[0]):
    oi = order[i, :]
    ti = target[i, :]
    yi = torch.zeros(n_timepoints)
    onset = oi[ti==1]*stimulus_to_stimulus_interval
    yi[onset] = 1
    ylist.append(yi)
y = torch.cat(ylist)

D = torch.vstack([
    torch.roll(y, i) for i in range(stimulus_window)
]).T

Xq, Xr = torch.linalg.qr(X)
Dq, Dr = torch.linalg.qr(D)

U, S, V = torch.linalg.svd(Dq.T @ Xq)

I = latent_dim

U = U[:, :I]
S = S[:I]
V = V[:I, :]

Uhat = torch.inverse(Xr) @ V.T
a = torch.inverse(Dr) @ U @ torch.diag(S)

L = model.variables["loadings"].data
Lstd = L / L.norm(2, dim=0).reshape(1, -1)
Uhatstd = Uhat / Uhat.norm(2, dim=0).reshape(1, -1)

Lstd.T @ Uhatstd

Shat = X @ Uhat

plt.plot(a.cpu())
plt.show()