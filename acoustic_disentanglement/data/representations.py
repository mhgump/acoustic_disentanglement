"""
"""
import json
import os
import h5py
import numpy as np

from speech_representations.data.data_reader import get_jagged_indices
from speech_representations.utils.math import gaussian_normalization


class NumpyRepresentationDatastore:

    def __init__(self, filename, mode='a'):
        self.filename = filename
        os.makedirs(os.path.dirname(self.filename), exist_ok=True)
        self.file = h5py.File(self.filename, mode=mode)

    def close(self):
        self.file.close()

    def num_items(self, feature_name, model_name, trial_index, chkpt_index):
        return self.file[self.data_path(feature_name, model_name, trial_index, chkpt_index)].shape[0]

    @staticmethod
    def data_path(feature_name, model_name, trial_index, chkpt_index):
        return '{}/{}/{}/{}/data'.format(feature_name, model_name, trial_index, chkpt_index)

    @staticmethod
    def seq_pcts_path(feature_name, model_name, trial_index, chkpt_index):
        return '{}/{}/{}/{}/seg_pcts'.format(feature_name, model_name, trial_index, chkpt_index)

    @staticmethod
    def length_path(feature_name, model_name, trial_index, chkpt_index):
        return '{}/{}/{}/{}/length'.format(feature_name, model_name, trial_index, chkpt_index)

    def add_representation(self, feature_name, model_name, trial_index, chkpt_index, num_items, z_dim):
        self.file.create_dataset(self.data_path(feature_name, model_name, trial_index, chkpt_index),
            (num_items, z_dim,), dtype=h5py.special_dtype(vlen=np.float32))
        self.file.create_dataset(self.seq_pcts_path(feature_name, model_name, trial_index, chkpt_index),
            (num_items, 1), dtype=np.float32)
        self.file.create_dataset(self.length_path(feature_name, model_name, trial_index, chkpt_index),
            (num_items, 1), dtype=np.float32)

    def add_value(self, feature_name, model_name, trial_index, chkpt_index, index, value, seq_pct):
        self.file[self.data_path(feature_name, model_name, trial_index, chkpt_index)][index, :] = value.transpose(1, 0)
        self.file[self.seq_pcts_path(feature_name, model_name, trial_index, chkpt_index)][index, :] = seq_pct
        self.file[self.length_path(feature_name, model_name, trial_index, chkpt_index)][index, :] = value.shape[0]

    def generate_latents(self, feature_name):
        for model_name in sorted(self.file[feature_name].keys()):
            for trial_ind in sorted(self.file['{}/{}'.format(feature_name, model_name)].keys()):
                for chkpt_ind in sorted(self.file['{}/{}/{}'.format(feature_name, model_name, trial_ind)].keys()):
                    yield model_name, trial_ind, chkpt_ind


class NumpyRepresentationDatareader(NumpyRepresentationDatastore):

    def __init__(self, filename):
        super().__init__(filename, mode='r')

    def set_feature(self, feature_name, model_name, trial_index, chkpt_index):
        self._data_path = self.data_path(feature_name, model_name, trial_index, chkpt_index)
        self._seg_pcts_path = self.seq_pcts_path(feature_name, model_name, trial_index, chkpt_index)
        self._length_path = self.length_path(feature_name, model_name, trial_index, chkpt_index)

    def get_data(self, indices):
        unique, unique_inverse = np.unique(indices, return_inverse=True)
        data = self.file[self._data_path][unique, :][unique_inverse]
        seg_pcts = self.file[self._seg_pcts_path][unique, :][unique_inverse]
        lengths = self.file[self._length_path][unique, :][unique_inverse]
        max_length = np.max(lengths)
        N, d = data.shape
        data = np.concatenate(data.transpose().reshape(-1)).reshape(-1, d, order='F')
        # We could compute seq_inds without this fancy indexing, but the indices are necessary to compute seg_start_pcts quickly
        _inds = get_jagged_indices(lengths, max_length=max_length, N=N)
        seq_inds = np.repeat(np.arange(N), max_length)[_inds]
        seg_start_pcts = np.repeat(np.arange(max_length).reshape(1, -1), N, axis=0) * seg_pcts
        seg_start_pcts = seg_start_pcts.reshape(-1)[_inds]
        seg_length_pcts = np.repeat(seg_pcts, max_length)[_inds]
        return data, seq_inds, seg_start_pcts, seg_length_pcts
