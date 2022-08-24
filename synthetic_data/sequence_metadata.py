import numpy as np


def sequence_metadata(
		n_character: int = 19,
		n_sequences_per_character: int = 15,
		n_stimuli: tuple[int] = (6, 6)
) -> tuple[list[list[list[int]]], list[list[list[int]]], list[list[list[int]]]]:
	"""
	Generates the sequence information.

	:param n_character:
	:param n_sequences_per_character:
	:param n_stimuli:
	:param n_regions:
	:return:
		w[l][i] of length sum(n_stimuli) and contains the order of stimuli
		y[l][i] of length sum(n_stimuli) and contains 0/1 indicators of whether it is a target stimuli
		yy[l] is of length len(n_stimuli) and contains the target stimuli (this is redundant given w and y, but it's nice to have)
	"""
	L = n_character
	I = n_sequences_per_character
	J = sum(n_stimuli)
	l_stimuli = np.cumsum([0, *n_stimuli[:-1]])
	u_stimuli = np.cumsum(n_stimuli)
	# generate ordering of stimuli
	w = [[
		np.concatenate([np.random.permutation(range(l, u)) for l, u in zip(l_stimuli, u_stimuli)]).tolist()
		for _ in range(I)] for _ in range(L)]
	yy = [
		[np.random.randint(l, u) for l, u in zip(l_stimuli, u_stimuli)]
		for _ in range(L)
	]
	y = [[[0 for _ in range(J)] for _ in range(I)] for _ in range(L)]
	for l in range(L):
		for i in range(I):
			for j in yy[l]:
				y[l][i][w[l][i][j]] = 1
	return w, y, yy


def sequence_metadata_long(
		n_character: int = 19,
		n_sequences_per_character: int = 15,
		n_stimuli: tuple[int] = (6, 6)
) -> tuple[list[int], list[int]]:
	wwide, ywide, yy = sequence_metadata(n_character, n_sequences_per_character, n_stimuli)
	seq = []
	char = []
	wlong = []
	ylong = []
	for l, (wl, yl) in enumerate(zip(wwide, ywide)):
		for i, (wli, yli) in enumerate(zip(wl, yl)):
			seq.append(i)
			char.append(l)
			wlong.append(wli)
			ylong.append(yli)
	return char, seq, wlong, ylong, yy
