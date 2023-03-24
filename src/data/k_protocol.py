import pandas as pd
import torch
import numpy as np
import scipy.io
import scipy.signal


def _array_to_dict(mat_array):
    return {k: v for k, v in zip(mat_array.dtype.names, mat_array)}


def _identify_sequences_and_stimuli(states: dict, window: float, sampling_rate: int) -> tuple:
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
        "character": stimulus_sequence,
        "src": stimulus_code,
        "type": stimulus_type,
        "length": stimulus_end - stimulus_begin,
    }).reset_index(drop=True)
    # identify repetitions
    stimulus_data["new_character"] = stimulus_data["character"].shift(1) != stimulus_data["character"]
    stimulus_data["sequence_end"] = ((stimulus_data["src"].cumsum() % 78) == 0).astype(int)
    stimulus_data["sequence_begin"] = stimulus_data["sequence_end"].shift(-11).replace(np.nan, 0).astype(int)
    stimulus_data["sequence"] = stimulus_data["sequence_begin"].cumsum().astype(int)

    return stimulus_data


class KProtocol:

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
        mat_obj = scipy.io.loadmat(filename)["Data"][0][0][0][0][0]
        signal = mat_obj[0]
        states = _array_to_dict(mat_obj[1][0][0])
        parameters = _array_to_dict(mat_obj[2][0][0])
        parameters = {k: _array_to_dict(v[0][0]) for k, v in parameters.items()}
        sampling_rate: int = parameters["SamplingRate"]["NumericValue"][0][0]

        # bandpass filter
        bandpass = scipy.signal.butter(
            N=bandpass_order,
            Wn=bandpass_window,
            btype="bandpass",
            fs=sampling_rate
        )
        filtered = scipy.signal.lfilter(*bandpass, signal.T).T

        # get sequences and stimuli
        stimulus_data = _identify_sequences_and_stimuli(states, window, sampling_rate)

        # construct sequence tensor
        seq_ids = stimulus_data["sequence"].unique()
        begin = [stimulus_data["begin"][
            (stimulus_data["sequence_begin"] == 1) *
            (stimulus_data["sequence"] == seq_id)
        ].values[0] for seq_id in seq_ids]
        end = [stimulus_data["end"][
            (stimulus_data["sequence_end"] == 1) *
            (stimulus_data["sequence"] == seq_id)
        ].values[0] for seq_id in seq_ids]
        length = [end[i] - begin[i] for i in range(len(begin))]
        max_length = np.max(length)
        sequence = torch.zeros((len(seq_ids), filtered.shape[1], max_length))
        for i, (b, e) in enumerate(zip(begin, end)):
            sequence[i, :, :e - b] = torch.tensor(filtered[b:e, :]).T

        # construct stimulus tensor
        stimulus = torch.zeros(len(seq_ids), 12, dtype=int)
        for i, seq_id in enumerate(seq_ids):
            stimulus[i, :] = torch.tensor(
                stimulus_data["src"].loc[stimulus_data["sequence"] == seq_id].values.astype(int)
            )

        # construct target tensor
        target = torch.zeros(len(seq_ids), 12, dtype=int)
        for i, seq_id in enumerate(seq_ids):
            which = stimulus_data["sequence"] == seq_id
            is_target = stimulus_data["type"].loc[which].values.astype(int)
            order = stimulus_data["src"].loc[which].values.astype(int)
            target[i, :] = torch.tensor(is_target[order.argsort()])

        # downsample
        sequence = sequence[:, :, ::downsample]

        # compute window in interval
        stimulus_window = int(window * sampling_rate / 1000 / downsample)
        sts_interval = int(40 / downsample)
        total_length = sts_interval * 11 + stimulus_window
        sequence = sequence[:, :, :total_length]

        self.session = session
        self.subject = subject
        self.type = type
        self.window = window
        self.sampling_rate = sampling_rate
        self.bandpass_window = bandpass_window
        self.bandpass_order = bandpass_order
        self.downsample = downsample
        self.sequence = sequence
        self.stimulus_order = stimulus
        self.target = target
        self.stimulus_data = stimulus_data
        self.stimulus_to_stimulus_interval = sts_interval
        self.stimulus_window = stimulus_window