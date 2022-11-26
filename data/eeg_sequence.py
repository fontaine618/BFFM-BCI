from dataclasses import dataclass
import pandas as pd
import numpy as np
import scipy.io
import scipy.signal


class KProtocol:
	#TODO this could be a dataclass, use __post_init__

	def __init__(
			self,
			filename: str,
			type: str,
			subject: str,
			session: str,
			window: float = 800.,
			bandpass_window: tuple = (0.1, 15.),
			bandpass_order: int = 5,
			downsample: int = 8
	):
		self.session = session
		self.subject = subject
		self.type = type
		self.window = window
		mat_obj = scipy.io.loadmat(filename)["Data"][0][0][0][0][0]
		signal = mat_obj[0]
		states = array_to_dict(mat_obj[1][0][0])
		parameters = array_to_dict(mat_obj[2][0][0])
		parameters = {k: array_to_dict(v[0][0]) for k, v in parameters.items()}
		self.sampling_rate: int = parameters["SamplingRate"]["NumericValue"][0][0]
		self.bandpass_window = bandpass_window
		self.bandpass_order = bandpass_order
		self.downsample = downsample

		stimulus_data, sequence_data = process_data(states, window, self.sampling_rate)

		# bandpass filter
		bandpass = scipy.signal.butter(
			N=bandpass_order,
			Wn=bandpass_window,
			btype="bandpass",
			fs=self.sampling_rate
		)
		filtered = scipy.signal.lfilter(*bandpass, signal.T).T

		# build signal tensor
		max_window = (stimulus_data["end"] - stimulus_data["begin"]).max()
		n_stimulus = len(stimulus_data)
		n_channels = filtered.shape[1]
		stimulus_signal = np.ones((n_stimulus, max_window, n_channels)) * np.nan
		for i, (start, end) in enumerate(zip(stimulus_data["begin"], stimulus_data["end"])):
			n = end - start
			stimulus_signal[i, 0:n, :] = filtered[start:end, :]

		# downsample
		stimulus_signal = stimulus_signal[:, ::downsample, :]

		# store
		self.stimulus_data = stimulus_data
		self.sequence_data = sequence_data
		self.signal = stimulus_signal


def process_data(states: dict, window: float = 800., sampling_rate: float = 256.):
	# identify sequences
	stimulus_code = states["StimulusCode"].squeeze()
	stimulus_type = states["StimulusType"].squeeze()
	stimulus_begin = states["StimulusBegin"].squeeze()
	phase_in_sequence = states["PhaseInSequence"].squeeze()
	sequence_data = pd.DataFrame({
		"stimulus_code": stimulus_code,
		"stimulus_type": stimulus_type,
		"stimulus_begin": stimulus_begin,
		"phase_in_sequence": phase_in_sequence,
	})
	sequence_data["time_ms"] = sequence_data.index * 1000 / sampling_rate
	sequence_data["previous_phase"] = sequence_data["phase_in_sequence"].shift(1)
	sequence_data["next_phase"] = sequence_data["phase_in_sequence"].shift(-1)
	sequence_data["new_sequence"] = (sequence_data["phase_in_sequence"] == 1) & \
	                                (sequence_data["previous_phase"] != 1)
	sequence_data["end_sequence"] = (sequence_data["phase_in_sequence"] == 3) & \
	                                (sequence_data["next_phase"] != 3)
	sequence_data["min_sequence"] = sequence_data["new_sequence"].cumsum()
	sequence_data["max_sequence"] = sequence_data["end_sequence"].shift(1).cumsum() + 1
	sequence_data["sequence"] = sequence_data["min_sequence"] * \
	                            (sequence_data["min_sequence"] == sequence_data["max_sequence"])
	# identify stimulus
	sequence_data["previous_stimulus"] = sequence_data["stimulus_code"].shift(1)
	sequence_data["new_stimulus"] = (sequence_data["stimulus_code"] > 0) & \
	                                (sequence_data["previous_stimulus"] == 0)
	# identify start and end of window after stimulus
	tmp = (sequence_data["new_stimulus"] == 1)
	stimulus_begin = tmp[tmp].index.values
	stimulus_sequence = sequence_data["sequence"][stimulus_begin]
	sequence_end = sequence_data.groupby("sequence").apply(lambda df: df.index.max())
	# restrict to sequence
	stimulus_sequence_end = sequence_end[stimulus_sequence].values.astype(int)
	# maximum length
	stimulus_max = np.ceil(stimulus_begin + window * sampling_rate / 1000).astype(int)
	# min between the two conditions
	stimulus_end = np.fmin(stimulus_sequence_end, stimulus_max) + 1
	stimulus_code = sequence_data["stimulus_code"][stimulus_begin]
	stimulus_type = sequence_data["stimulus_type"][stimulus_begin]
	# store all info
	stimulus_data = pd.DataFrame({
		"begin": stimulus_begin,
		"end": stimulus_end,
		"sequence": stimulus_sequence,
		"src": stimulus_code,
		"type": stimulus_type,
		"length": stimulus_end - stimulus_begin,
	}).reset_index(drop=True)
	return stimulus_data, sequence_end


def array_to_dict(mat_array):
	return {
		k: v for k, v in zip(mat_array.dtype.names, mat_array)
	}
