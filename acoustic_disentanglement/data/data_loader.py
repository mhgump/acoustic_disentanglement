""" speech_representations/data/data_loader.py

Uses BatchLoader to access through primitive data and feature registry to apply
transformations to these raw features. Used for all non model data access.
"""
import numpy as np

from speech_representations.features import transformer, load_normalization, FeatureSpec
from speech_representations.data.data_reader import DataReader
from speech_representations.data.epoch_loader import get_epoch_loader, NonSequentialEpochLoader


class SequenceEpoch:
    def __init__(self, num_sequences, batch_size, seed=None):
        self.num_sequences = num_sequences
        self.batch_size = batch_size

    def __iter__(self):
        total_batches = int(np.ceil(self.num_sequences / self.batch_size))
        for i in range(total_batches):
            inds = np.arange(i * self.batch_size, min((i + 1) * self.batch_size, self.num_sequences)) 
            self.current_indices = inds
            yield inds

class RandomSequenceEpoch:
    def __init__(self, num_sequences, batch_size, seed=None):
        self.num_sequences = num_sequences
        self.batch_size = batch_size
        random = np.random.RandomState(seed)
        self._random_state = random.get_state()

    def __iter__(self):
        random = np.random.RandomState()
        random.set_state(self._random_state)
        all_indices = random.permutation(self.num_sequences)
        current_batch_pointer = 0
        while current_batch_pointer <= self.num_sequences:
            inds = all_indices[current_batch_pointer:current_batch_pointer+self.batch_size]
            self.current_indices = inds
            yield inds
            current_batch_pointer += self.batch_size


def strided_segment_batches(sequence_indices, sequence_lengths, seq_batch_size, window_length, stride_length,
        include_overhang=True):
    sequence_lengths = 1. * sequence_lengths
    N = sequence_lengths.shape[0]
    avg_length = np.mean(sequence_lengths)
    max_length = np.max(sequence_lengths)
    last_e_ind = 0
    for e_ind in range(seq_batch_size, N + 1, seq_batch_size):
        last_e_ind = e_ind
        s_ind = e_ind - seq_batch_size
        batch_indices = np.arange(s_ind, e_ind)
        subsequence_indices = np.arange(0, max_length, stride_length).reshape(1, -1)
        max_strides = subsequence_indices.shape[1]
        subsequence_indices = np.repeat(subsequence_indices, batch_indices.shape[0], axis=0).flatten()
        _sequence_indices = sequence_indices[batch_indices]
        _sequence_indices = np.repeat(_sequence_indices, max_strides)
        _sequence_lengths = np.repeat(sequence_lengths[batch_indices], max_strides)
        valid_indices = np.where((subsequence_indices + window_length) <= _sequence_lengths)[0]
        _sequence_indices = _sequence_indices[valid_indices]
        subsequence_start_pcts = subsequence_indices[valid_indices] * np.reciprocal(_sequence_lengths[valid_indices])
        subsequence_length_pcts = window_length * np.reciprocal(_sequence_lengths[valid_indices])
        yield _sequence_indices, subsequence_start_pcts, subsequence_length_pcts
    if include_overhang and last_e_ind != N:
        batch_indices = np.arange(last_e_ind, N)
        subsequence_indices = np.arange(0, max_length, stride_length).reshape(1, -1)
        max_strides = subsequence_indices.shape[1]
        subsequence_indices = np.repeat(subsequence_indices, batch_indices.shape[0], axis=0).flatten()
        _sequence_indices = sequence_indices[batch_indices]
        _sequence_indices = np.repeat(_sequence_indices, max_strides)
        _sequence_lengths = np.repeat(sequence_lengths[batch_indices], max_strides)
        valid_mask = (subsequence_indices + window_length) <= _sequence_lengths
        _sequence_indices = _sequence_indices[valid_mask]
        subsequence_start_pcts = subsequence_indices[valid_mask] * np.reciprocal(_sequence_lengths[valid_mask])
        subsequence_length_pcts = window_length * np.reciprocal(_sequence_lengths[valid_mask])
        yield _sequence_indices, subsequence_start_pcts, subsequence_length_pcts


def random_segment_batches(sequence_indices, sequence_lengths, batch_size, window_length, K, 
        include_overhang=True, random=None):
    assert batch_size % K == 0
    sequence_lengths = 1. * sequence_lengths
    if random is None:
        random = np.random.RandomState()
    M = sequence_indices.shape[0]
    batch_M = batch_size // K
    last_e_ind = 0
    for e_ind in range(batch_M, M + 1, batch_M):
        last_e_ind = e_ind
        s_ind = e_ind - batch_M
        batch_indices = np.arange(s_ind, e_ind)
        _sequence_indices = np.repeat(sequence_indices[batch_indices], K)
        _sequence_lengths = np.repeat(sequence_lengths[batch_indices], K)
        start_pct_norm = (_sequence_lengths - window_length) * np.reciprocal(_sequence_lengths * 1.)
        subsequence_start_pcts = random.rand(_sequence_indices.shape[0]) * start_pct_norm
        subsequence_length_pcts = window_length * np.reciprocal(_sequence_lengths * 1.)
        yield _sequence_indices, subsequence_start_pcts, subsequence_length_pcts
    if include_overhang and last_e_ind != M:
        batch_indices = np.arange(last_e_ind, M)
        _sequence_indices = np.repeat(sequence_indices[batch_indices], K)
        _sequence_lengths = np.repeat(sequence_lengths[batch_indices], K)
        start_pct_norm = (_sequence_lengths - window_length) * np.reciprocal(_sequence_lengths * 1.)
        subsequence_start_pcts = random.rand(_sequence_indices.shape[0]) * start_pct_norm
        subsequence_length_pcts = window_length * np.reciprocal(_sequence_lengths * 1.)
        yield _sequence_indices, subsequence_start_pcts, subsequence_length_pcts
