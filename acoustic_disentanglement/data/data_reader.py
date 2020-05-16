import numpy as np
import pandas as pd
import os
import h5py

from speech_representations.data.datasource import SplitDatasourceBase
from speech_representations.features import load_normalization


def aligned_batch_from_variable_length_data(data, sequence_lengths, ends, \
        sequence_indices, pct_window_start, pct_window_length, window_length):
    N = sequence_indices.shape[0]
    d = data.shape[2]
    segment_indices = np.arange(0.5, window_length, 1.0) * np.reciprocal(window_length * 1.)
    segment_indices = np.repeat(segment_indices.reshape(1, -1), N, axis=0)
    segment_indices *= (pct_window_length * sequence_lengths).reshape(-1, 1)
    segment_indices += (pct_window_start * sequence_lengths).reshape(-1, 1)
    max_num_segments = ends.shape[1]
    ends = ends.reshape(N, 1, max_num_segments)
    segment_indices = segment_indices.reshape(N, window_length, 1)
    segment_indices = np.argmax(segment_indices < ends, axis=2) # shape (N, window_length)
    sequence_indices = np.repeat(sequence_indices.reshape(-1, 1), window_length, axis=1)
    return data[sequence_indices, segment_indices, :].reshape(N, window_length, d)


def aligned_batch_from_data(data, sequence_lengths, \
        sequence_indices, pct_window_start, pct_window_length, window_length):
    N = sequence_indices.shape[0]
    d = data.shape[2]
    segment_indices = np.arange(0.5, window_length, 1.0) * np.reciprocal(window_length * 1.)
    segment_indices = np.repeat(segment_indices.reshape(1, -1), N, axis=0)
    segment_indices *= (pct_window_length * sequence_lengths).reshape(-1, 1)
    segment_indices += (pct_window_start * sequence_lengths).reshape(-1, 1)
    segment_indices = np.floor(segment_indices).astype(np.int32)
    sequence_indices = np.repeat(sequence_indices.reshape(-1, 1), window_length, axis=1)
    return data[sequence_indices, segment_indices, :].reshape(N, window_length, d)


def get_jagged_indices(lengths, max_length=None, N=None):
    N = N or lengths.shape[0]
    max_length = max_length or np.max(lengths)
    P = np.repeat(np.arange(max_length).reshape(1, -1), N, axis=0)
    Q = np.repeat(lengths.reshape(-1, 1), max_length, axis=1)
    M = np.where(P.reshape(-1) < Q.reshape(-1))[0]
    return M


def pad_jagged_data(data):
    N, d = data.shape
    lengths = np.array(list(map(len, data[:,0])))
    max_length = np.max(lengths)
    M = get_jagged_indices(lengths, max_length=max_length, N=N)
    out_data = np.zeros((N * max_length, d))
    out_data[M, :] = np.concatenate(data.transpose().reshape(-1)).reshape(-1, d, order='F')[:]
    out_data = out_data.reshape(N, max_length, d)
    return out_data


class DataReader:

    def get_num_items(self):
        pass

    def get_data(self, feature_name, indices):
        raise NotImplementedError

    def get_num_segments(self, feature_name, indices):
        raise NotImplementedError

    def get_sequence_length(self, feature_name, indices):
        raise NotImplementedError

    def get_segment_ends(self, feature_name, indices):
        raise NotImplementedError

    def get_length(self, feature_name):
        raise NotImplementedError

    def is_jagged_length(self, feature_name):
        raise NotImplementedError

    def is_variable_length_segments(self, feature_name):
        raise NotImplementedError

    def get_windowing_length_func(self, feature_name):
        def _func(indices):
            if self.is_variable_length_segments(feature_name):
                return self.get_sequence_length(feature_name, indices)
            else:
                return self.get_num_segments(feature_name, indices)
        return _func

    def get_padded_mask(self, feature_name, indices):
        assert self.is_jagged_length(feature_name), 'Can only compute padded batches for jagged data'
        N = indices.shape[0]
        num_segments = self.get_num_segments(feature_name, indices).reshape(-1, 1)
        max_length = np.max(num_segments)
        mask = np.repeat([[1], [0]], N, axis=1).transpose(1,0).flatten()
        mask_lengths = np.concatenate([num_segments, max_length - num_segments], axis=1).flatten()
        mask = np.repeat(mask, mask_lengths).reshape(N, max_length)
        return mask.astype(np.bool)

    def get_padded_batch(self, feature_name, indices, return_mask=True):    
        assert self.is_jagged_length(feature_name), 'Can only compute padded batches for jagged data'
        padded_data = pad_jagged_data(self.get_data(feature_name, indices))
        if return_mask:
            return padded_data, self.get_padded_mask(feature_name, indices)
        else:
            return padded_data

    def get_padded_segment_ends(self, feature_name, indices, return_mask=False):
        assert self.is_jagged_length(feature_name), 'Can only compute padded batches for jagged data'
        padded_data = pad_jagged_data(self.get_segment_ends(feature_name, indices))
        if return_mask:
            return padded_data, self.get_padded_mask(feature_name, indices)
        else:
            return padded_data

    def get_aligned_batch(self, feature_name, sequence_indices, pct_window_start, pct_window_length, window_length):
        assert self.is_jagged_length(feature_name), 'Can only compute aligned batches for jagged data'
        data, _ = self.get_padded_batch(feature_name, sequence_indices)
        if self.is_variable_length_segments(feature_name):
            sequence_lengths = self.get_sequence_length(feature_name, sequence_indices).reshape(-1)
            ends = self.get_padded_segment_ends(feature_name, sequence_indices)
            return aligned_batch_from_variable_length_data(data, sequence_lengths, ends, 
                np.arange(sequence_indices.shape[0]),  pct_window_start, pct_window_length, window_length)
        else:
            sequence_lengths = self.get_num_segments(feature_name, sequence_indices).reshape(-1)
            return aligned_batch_from_data(data, sequence_lengths, np.arange(sequence_indices.shape[0]),
                pct_window_start, pct_window_length, window_length)


class DatasourceReader(SplitDatasourceBase, DataReader):

    def __init__(self, filename, partition_name):
        SplitDatasourceBase.__init__(self, filename)
        self.open(mode='r')
        self.partition_name = partition_name

    def get_num_items(self):
        return self.ptn_num_items[self.partition_name]

    def get_data(self, feature_name, indices):
        unique, unique_inverse = np.unique(indices, return_inverse=True)
        data = self.data[self.partition_name, feature_name][unique, :]
        return data[unique_inverse].reshape((indices.shape[0], -1))

    def get_num_segments(self, feature_name, indices):
        unique, unique_inverse = np.unique(indices, return_inverse=True)
        data = self.num_segments[self.partition_name, feature_name][unique, :]
        return data[unique_inverse].reshape((indices.shape[0], -1))

    def get_sequence_length(self, feature_name, indices):
        unique, unique_inverse = np.unique(indices, return_inverse=True)
        data = self.sequence_length[self.partition_name, feature_name][unique, :]
        return data[unique_inverse].reshape((indices.shape[0], -1))

    def get_segment_ends(self, feature_name, indices):
        unique, unique_inverse = np.unique(indices, return_inverse=True)
        data = self.segment_ends[self.partition_name, feature_name][unique, :]
        return data[unique_inverse].reshape((indices.shape[0], -1))

    def get_normalization(self, feature_name):
        return load_normalization(self.normalization[self.partition_name, feature_name])

    def get_length(self, feature_name):
        return self.length[feature_name]

    def is_jagged_length(self, feature_name):
        return self.jagged_length[feature_name]

    def is_variable_length_segments(self, feature_name):
        return self.variable_length_segments[feature_name]
