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
        "source": stimulus_code,
        "type": stimulus_type,
        "length": stimulus_end - stimulus_begin,
    }).reset_index(drop=True)
    # identify repetitions
    stimulus_data["new_character"] = stimulus_data["character"].shift(1) != stimulus_data["character"]
    stimulus_data["sequence_end"] = ((stimulus_data["source"].cumsum() % 78) == 0).astype(int)
    stimulus_data["sequence_begin"] = stimulus_data["sequence_end"].shift(-11).replace(np.nan, 0).astype(int)
    stimulus_data["sequence"] = stimulus_data["sequence_begin"].cumsum().astype(int)
    stimulus_data["active"] = (stimulus_data["sequence_begin"] - \
                              stimulus_data["sequence_end"]).cumsum() + \
                            stimulus_data["sequence_end"]
    stimulus_data = stimulus_data[stimulus_data["active"] == 1]
    stimulus_data["repetition"] = stimulus_data.groupby("character")["sequence_begin"].cumsum()
    return stimulus_data


def _patch_frt(filename, stimulus_data):
    # Data.DBIData.DBI_EXP_Info.Stimulus_Type
    intended_text = scipy.io.loadmat(filename)["Data"][0][0][1][0][0][4][0][0]
    row_col = intended_text[4]
    row = row_col[:, 0]
    col = row_col[:, 1]
    n_chars = row.shape[0]
    # need to drop the last ones; it seems the session may include an extra one sometimes
    stimulus_data = stimulus_data[stimulus_data["character"] < n_chars + 1]
    row = pd.DataFrame({
        "character": np.arange(1, n_chars + 1),
        "source": row,
        "type_raw": 1
    })
    col = pd.DataFrame({
        "character": np.arange(1, n_chars + 1),
        "source": col,
        "type_raw": 1
    })
    df = pd.concat([row, col])
    # join df into stimulus_data on character and source
    stimulus_data = pd.merge(stimulus_data, df, on=["character", "source"], how="left")
    # now, type_raw has 1 or NaN, we need to replace NaN with 0 and then convert to int
    stimulus_data["type"] = stimulus_data["type_raw"].replace(np.nan, 0).astype(int)
    # drop type_raw
    stimulus_data = stimulus_data.drop(columns=["type_raw"])
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
            downsample: int = 8,
            center: bool = True
    ):
        mat_obj = scipy.io.loadmat(filename)["Data"][0][0][0][0][0]
        signal = mat_obj[0]
        states = _array_to_dict(mat_obj[1][0][0])
        parameters = _array_to_dict(mat_obj[2][0][0])
        parameters = {k: _array_to_dict(v[0][0]) for k, v in parameters.items()}
        sampling_rate: int = parameters["SamplingRate"]["NumericValue"][0][0]

        characters = [x[0][0] for x in parameters["TargetDefinitions"]["Value"]]
        rows = np.arange(1, 7).repeat(6)
        columns = np.tile(np.arange(1, 7), 6)
        d = downsample

        design = pd.DataFrame({
            "character": characters,
            "row": rows,
            "column": columns,
            "stimulus_code": [(row-1, col+5) for row, col in zip(rows, columns)]
        })

        # bandpass filter
        bandpass = scipy.signal.butter(
            N=bandpass_order,
            Wn=bandpass_window,
            btype="bandpass",
            fs=sampling_rate
        )
        filtered = torch.Tensor(scipy.signal.lfilter(*bandpass, signal.T).T)

        # MA smoothing
        if downsample > 1:
            smoothed = torch.nn.functional.avg_pool1d(
                filtered.T,
                downsample-1, 1,
                padding=(downsample-2)//2, count_include_pad=False
            ).T
        else:
            smoothed = filtered

        # get sequences and stimuli
        stimulus_data = _identify_sequences_and_stimuli(states, window, sampling_rate)

        # patch FRT target
        if type == "FRT":
            stimulus_data = _patch_frt(filename, stimulus_data)

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
        max_length = np.max(length) + d - 1
        sequence = torch.zeros((len(seq_ids), filtered.shape[1], max_length))
        for i, (b, e) in enumerate(zip(begin, end)):
            sequence[i, :, :(e-b+d-1)] = filtered[b:(e+d-1), :].T

        # construct order and target tensor
        target = torch.zeros(len(seq_ids), 12, dtype=int)
        stimulus = torch.zeros(len(seq_ids), 12, dtype=int)
        character_idx = torch.zeros(len(seq_ids), dtype=int)
        for i, seq_id in enumerate(seq_ids):
            which = stimulus_data["sequence"] == seq_id
            # this is the "source", i.e., the row/col identifier
            order = stimulus_data["source"].loc[which].values.astype(int)
            # we rather need the reverse: when a row/column is presented
            inv_order = order.argsort()
            stimulus[i, :] = torch.tensor(inv_order)
            # this is "type", i.e., whether the active sitmulus is a target
            is_target = stimulus_data["type"].loc[which].values.astype(int)
            # we rather want the reverse: which row/column is a target
            target[i, :] = torch.tensor(is_target[inv_order])
            character_idx[i] = stimulus_data["character"].loc[which].values[0]

        # downsample
        sequence = sequence[:, :, ::downsample]

        # compute window in interval
        stimulus_window = int(window * sampling_rate / 1000 / downsample)
        sts_interval = int(40 / downsample)
        total_length = sts_interval * 11 + stimulus_window
        sequence = sequence[:, :, :total_length]

        # center each sequence
        if center:
            sequence = sequence - sequence.mean(2, keepdim=True)

        # data
        self.filtered = filtered
        self.smoothed = smoothed
        # parameters
        self.session = session
        self.subject = subject
        self.type = type
        self.window = window
        self.sampling_rate = sampling_rate
        self.bandpass_window = bandpass_window
        self.bandpass_order = bandpass_order
        self.downsample = downsample
        self.stimulus_to_stimulus_interval = sts_interval
        self.stimulus_window = stimulus_window
        self.design = design

        # I'm not sure if this is correct, but this is what Tianwen used,
        # and it looks fine when plotted
        self.channel_names = ['F3', 'Fz', 'F4', 'T7', 'C3', 'Cz', 'C4', 'T8',
                       'CP3', 'CP4', 'P3', 'Pz', 'P4', 'PO7', 'PO8', 'Oz']
        self.keyboard = np.array([
            ["A", "B", "C", "D", "E", "F"],
            ["G", "H", "I", "J", "K", "L"],
            ["M", "N", "O", "P", "Q", "R"],
            ["S", "T", "U", "V", "W", "X"],
            ["Y", "Z", "1", "2", "3", "4"],
            ["5", "SPK", ".", "BS", "!", "_"]
        ])

        # data
        self.sequence = sequence
        self.stimulus_order = stimulus
        self.target = target
        self.stimulus_data = stimulus_data
        self.character_idx = character_idx - 1

        # prepare stimulus format
        max_length = stimulus_window
        stimulus = torch.zeros((len(self.stimulus_data), max_length*d, self.filtered.shape[1]))
        for i, (b, e) in enumerate(zip(self.stimulus_data["begin"], self.stimulus_data["end"])):
            stimulus[i, :(max_length*d), :] = self.filtered[b:(b+max_length*d), :]

        # downsample
        stimulus = stimulus[:, ::downsample, :]

        # center
        if center:
            stimulus = stimulus - stimulus.mean(1, keepdim=True)

        self.stimulus = stimulus

    def repetitions(self, reps: list[int], reverse: bool = False) -> "KProtocol":
        seqs = self.stimulus_data.groupby("sequence").head(1).loc[:, ("sequence", "character", "repetition")]
        keep = seqs["repetition"].isin(reps).values
        tkeep = torch.BoolTensor(keep)
        self.sequence = self.sequence[tkeep, :, :]
        self.stimulus_order = self.stimulus_order[tkeep, :]
        self.target = self.target[tkeep, :]
        self.character_idx = self.character_idx[tkeep]
        stim_keep = self.stimulus_data["repetition"].isin(reps).values
        self.stimulus_data = self.stimulus_data[stim_keep]
        self.stimulus = self.stimulus[stim_keep, :, :]
        if reverse:
            self.sequence = torch.flip(self.sequence, dims=(0,))
            self.stimulus_order = torch.flip(self.stimulus_order, dims=(0,))
            self.target = torch.flip(self.target, dims=(0,))
            self.character_idx = self.character_idx.flip(dims=(0,))
            self.stimulus_data = self.stimulus_data.iloc[::-1]
            self.stimulus = torch.flip(self.stimulus, dims=(0,))
        return self