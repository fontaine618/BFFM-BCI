import matplotlib.pyplot as plt
import torch
import pandas as pd
import numpy as np
from source.bffmbci.bffm import BFFModel
from source.bffmbci.bffm_init import bffm_initializer

plt.style.use("seaborn-v0_8-whitegrid")
pd.set_option('display.max_rows', 500)

# generate some data
torch.manual_seed(0)
K = 3
N = 15*19
n_stimulus = (6, 6)
J = sum(n_stimulus)
w = 55
d = 10
T = w + (J-1) * d
E = 15
self = BFFModel.generate_from_dimensions(
	latent_dim=K,
	n_sequences=N,
	n_stimulus=n_stimulus,
	stimulus_window=w,
	n_channels=E,
	stimulus_to_stimulus_interval=d,
	observation_variance=(3., 1.),
	kernel_gp_factor_processes=(0.99, 1., 1.),
	kernel_tgp_factor_processes=(0.99, 0.5, 1.),
	kernel_gp_loading_processes=(0.99, 1., 1.),
	kernel_tgp_loading_processes=(0.99, 0.5, 1.),
	kernel_gp_factor=(0.99, 0.1, 1.)
)
sequences = self.variables["observations"].data  # N x E x T in R
stimulus_order = self.variables["sequence_data"].order.data  # N x J in 0:J
target_stimulus = self.variables["sequence_data"].target.data  # N x J in {0,1}
stimulus_window = w
stimulus_to_stimulus_interval = d
latent_dim = K

from source.initialization.wfa import WFA


self = bffm_initializer(
	sequences=sequences,
	stimulus_order=stimulus_order,
	target_stimulus=target_stimulus,
	stimulus_window=w,
	stimulus_to_stimulus_interval=d,
	latent_dim=K,
)


l = self.variables["loadings"].data
v = self.variables["observation_variance"].data

l @ l.T
loadings @ loadings.T

l @ l.T + torch.diag(v)
loadings @ loadings.T + torch.diag(observation_variance)





fitted_mean = (factors.permute(0, 2, 1) @ loadings.T).permute(0, 2, 1)
observations = self.variables["observations"].data

i = 0
e = 1

plt.scatter(
	observations[i, e, :].cpu().numpy(),
	fitted_mean[i, e, :].cpu().numpy(),
	c=variance[i,:].cpu().numpy(),
	cmap="viridis"
)
plt.colorbar()
plt.tight_layout()
plt.show()


plt.plot(factors[0, :, :].cpu().numpy().T)
plt.tight_layout()
plt.show()


i = 1
e = 2
y = self.variables["sequence_data"].target.data[i, :]
o = self.variables["sequence_data"].order.data[i, :]

plt.cla()
# plt.plot(factors[i, e, :].cpu().numpy().T, label="factors")
plt.plot(sfactors[i, e, :].cpu().numpy().T, label="sfactors")
# plt.plot(self.variables["mean_factor_processes"]._value[i, e, :].cpu().numpy().T, label="mean_factor_processes")
plt.plot(self.variables["factor_processes"]._value[i, e, :].cpu().numpy().T, label="factor_processes")
plt.plot(lprocesses[i, e, :].cpu().numpy().T, label="lprocesses")
plt.plot(lprocesses[i, e, :].cpu().numpy().T *
         self.variables["factor_processes"]._value[i, e, :].cpu().numpy().T,
         label="product")
plt.plot(self.variables["loading_processes"]._value[i, e, :].cpu().numpy().T, label="loading_processes")
for j in range(12):
	plt.axvline(j*d, color="black", linestyle="--", alpha=(0.2 if y[o[j]]==0 else 1.))
plt.tight_layout()
plt.legend()
plt.show()



self.variables["shrinkage_factor"]._value
self.variables["heterogeneities"]._value






# construct the table of where are the previous target
target_index = target * torch.arange(0, J).reshape((1, -1)) * d
target_index = target_index.sort(dim=1)[0][:, -len(n_stimulus):]
time_since_target = torch.zeros([N, T, T])

# everywhere we need to put a 1. (N x 2 x T)
where = torch.arange(0., T).unsqueeze(0).unsqueeze(0) - target_index.unsqueeze(-1) - 1
where = where.type(torch.int64).clamp(-1, T-1)
where = torch.where(where==-1, T-1, where)

for j in range(len(n_stimulus)):
	indices = (
		torch.arange(0, N).reshape((-1, 1)).expand((-1, T)),
		torch.arange(0, T).reshape((1, -1)).expand((N, -1)),
		where[:, j, :]
	)
	time_since_target[indices] = 1.

time_since_target = time_since_target[:, :, :w].type(torch.int64)

pattern = [
	[
		"".join([str(x) for x in time_since_target[i, t, :].tolist()])
		for t in range(T)
	]
	for i in range(N)
]

# to long format
observations_long = observations.permute(2, 0, 1).reshape((-1, E))
# the order is such that the first T corresponds to the first sequence
observations_long[0:3, :]
observations[0, :, 0:3].T
# time
sequence_long = torch.arange(0., N).expand(T, -1).T.reshape((-1, )).type(torch.int64)
timepoint = torch.arange(0., T).expand(N, -1).type(torch.int64)
timepoint_long = timepoint.reshape((-1, )) % w
pattern_long = sum(pattern, [])

# in df
meta = pd.DataFrame({
	"sequence": sequence_long,
	"timepoint": timepoint_long,
	"pattern": pattern_long
})

X = pd.DataFrame(observations_long)
df = pd.concat([meta, X], axis=1)
df["pattern"].value_counts()
# drop those with multiple 1 since this will be tricky to deal with
to_drop = df["pattern"].str.count("1") > 1
df = df.loc[~to_drop]
df["pattern"].value_counts()
# compute new pattern names for the
df["target"] = (df["pattern"].str.count("1") > 0).astype(int)
# check
df[["sequence", "timepoint", "pattern", "target"]]
covs = df.drop(columns=["sequence", "pattern"]).groupby(["target", "timepoint"]).agg("cov")
patterns = covs.index.droplevel(2)
patterns = set(patterns.values)
cov_tensor = np.array([
	covs.loc[(*pattern, slice(None))].values
	for pattern in patterns
])

weights = df.groupby(["target", "timepoint"]).size().values

from qndiag import qndiag

D, _ = qndiag(cov_tensor, weights=weights)
Dnorm = (D**2).sum(1)
which = Dnorm.argsort()[-K:][::-1]
DK = D[:, which]
inv = np.linalg.inv(DK @ DK.T)




# FA
import torch
torch.manual_seed(0)
N, p, K = 1000, 7, 3
w = 5 * torch.ones(N)
Theta = torch.randn(p, K)
sig2 = torch.randn((p, )).abs()
z = torch.randn((N, K)) * w.reshape((-1, 1)).sqrt()
X = z @ Theta.T + torch.randn((N, p)) * sig2.sqrt()


from source.initialization.wfa import WFA
self = WFA(K)
self.fit(X, w)
Theta1 = self._loadings
sig21 = self._observation_variance
factors = self._m1

torch.cov(X.T)
Theta @ Theta.T
Theta1 @ Theta1.T

mean = factors @ Theta1.T

l1 = self._loadings.clone()
f1 = self._m1.clone()

l2 = self._loadings.clone()
f2 = self._m1.clone()

m1 = f1 @ l1.T
m2 = f2 @ l2.T