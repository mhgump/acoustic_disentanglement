"""

"""
import numpy as np
from enum import Enum


class EpochLoader:

    def __init__(self, num_items, batch_size, sequence_length_func=None, seed=0):
        """ Simple data access to jagged and variable length data along shuffled epochs.

        :seed int: Random seed used for non-determinism in data sampling.
        :window_length int:
        :stride_length int: In the units of window_length, distance between successive subsequences of a single element.
        :K int: Number of random subsequences pulled from each element, one of K and stride_length must be None.
        :chunk_size int: To reduce read cost, chunk_size can be increased so that elements are shuffled in larger chunks.
        """
        super().__init__
        self.num_items = num_items
        self.batch_size = batch_size
        self.sequence_length_func = sequence_length_func
        self.random = np.random.RandomState(seed)
        self.reset()

    def reset(self):
        raise NotImplementedError('Abstract class not implemented.')

    def next(self):
        raise NotImplementedError('Abstract class not implemented.')

    @property
    def sequence_indices(self):
        if not hasattr(self, '_sequence_indices'):
            return None
        return self._sequence_indices

    @property
    def sequence_lengths(self):
        if not hasattr(self, '_sequence_lengths'):
            return None
        return self._sequence_lengths

    @property
    def subsequence_indices(self):
        if not hasattr(self, '_subsequence_indices'):
            return None
        return self._subsequence_indices

    @property
    def subsequence_lengths(self):
        if not hasattr(self, '_subsequence_lengths'):
            return None
        return self._subsequence_lengths


class NonSequentialEpochLoader(EpochLoader):

    def __init__(self, num_items, batch_size, seed=0):
        super().__init__(num_items, batch_size, seed=seed)

    def reset(self):
        self.all_indices = self.random.permutation(self.num_items)

    def next(self, batch_size=None):
        batch_size = batch_size or self.batch_size
        if len(self.all_indices) == 0:
            return False
        self._sequence_indices = self.all_indices[:batch_size]
        self.all_indices = self.all_indices[batch_size:]
        return True


class RandomSequentialEpochLoader(EpochLoader):

    def __init__(self, num_items, batch_size, sequence_length_func, window_length, K, seed=0):
        self.K = K
        self.window_length = window_length
        super().__init__(num_items, batch_size, sequence_length_func, seed)
        assert (self.batch_size % self.K) == 0

    def reset(self):
        self.all_indices = self.random.permutation(self.num_items)
        self.all_sequence_lengths = self.sequence_length_func(self.all_indices).reshape(-1)
        valid_indices = self.all_sequence_lengths > self.window_length
        self.all_indices = self.all_indices[valid_indices]
        self.all_sequence_lengths = self.all_sequence_lengths[valid_indices]
        assert self.all_indices.shape  == self.all_sequence_lengths.shape

    def next(self, batch_size=None):
        batch_size = batch_size or self.batch_size
        if len(self.all_indices) == 0:
            return False
        num_sequences = batch_size // self.K
        num_sequences = num_sequences if self.all_indices.shape[0] >= num_sequences else self.all_indices.shape[0]
        indices = np.repeat(np.arange(0, num_sequences), self.K)
        self._sequence_indices = self.all_indices[indices]
        self._sequence_lengths = self.all_sequence_lengths[indices]
        feasible_window = 1 - (self.window_length * np.reciprocal(self._sequence_lengths * 1.))
        self._subsequence_indices = self.random.rand(indices.shape[0]) * feasible_window
        self._subsequence_lengths = self.window_length * np.reciprocal(self._sequence_lengths * 1.)
        self.all_indices = self.all_indices[num_sequences:]
        self.all_sequence_lengths = self.all_sequence_lengths[num_sequences:]
        assert self._sequence_indices.shape  == self._sequence_lengths.shape
        assert self.all_indices.shape  == self.all_sequence_lengths.shape
        return True


class WindowedSequentialEpochLoader(EpochLoader):

    def __init__(self, num_items, batch_size, sequence_length_func, window_length, stride_length, seed=0):
        self.stride_length = stride_length
        self.window_length = window_length
        super().__init__(num_items, batch_size, sequence_length_func,  seed)

    def reset(self):
        self.all_indices = self.random.permutation(self.num_items)
        self.all_sequence_lengths = self.sequence_length_func(self.all_indices)
        self.num_subsequences = 1 + (self.all_sequence_lengths - self.window_length) // self.stride_length
        self.num_subsequences = self.num_subsequences.reshape(-1)
        valid_indices = self.num_subsequences > 0
        self.all_indices = self.all_indices[valid_indices]
        self.all_sequence_lengths = self.all_sequence_lengths[valid_indices]
        self.num_subsequences = self.num_subsequences[valid_indices]
        self.cumulative_subsequence_count = np.cumsum(self.num_subsequences)
        self.start_index = 0
        assert self.all_indices.shape  == self.all_sequence_lengths.shape
        assert self.all_sequence_lengths.shape  == self.num_subsequences.shape
        assert self.num_subsequences.shape  == self.cumulative_subsequence_count.shape

    def next(self, batch_size=None):
        batch_size = batch_size or self.batch_size
        if len(self.all_indices) == 0:
            return False

        if not np.any((batch_size + self.start_index) <= self.cumulative_subsequence_count):
            M = self.all_indices.shape[0]
        else: 
            M = 1 + np.argmax((batch_size + self.start_index) <= self.cumulative_subsequence_count)
        num_reserved_subsequences = self.cumulative_subsequence_count[M - 1] - (batch_size + self.start_index)
        max_length = np.max(self.all_sequence_lengths[:M])
        base_indices = np.arange(0, max_length, self.stride_length)
        max_strides = base_indices.shape[0]
        subsequence_indices = np.repeat(base_indices.reshape(1, -1), M, axis=0).flatten()
        sequence_lengths = np.repeat(self.all_sequence_lengths[:M], max_strides)
        sequence_indices = np.repeat(np.arange(M), max_strides)

        indices = np.where((subsequence_indices + self.window_length) <= sequence_lengths)[0][self.start_index:]
        if 0 < num_reserved_subsequences:
            indices = indices[:-num_reserved_subsequences]
        self._sequence_indices = self.all_indices[sequence_indices[indices]]
        self._sequence_lengths = self.all_sequence_lengths[sequence_indices[indices]]
        self._subsequence_indices = subsequence_indices[indices] * np.reciprocal(self._sequence_lengths * 1.)
        self._subsequence_lengths = self.window_length * np.reciprocal(self._sequence_lengths * 1.)

        if num_reserved_subsequences == 0:
            self.start_index = 0                    
        elif 0 < num_reserved_subsequences:
            M -= 1
            self.start_index = self.num_subsequences[M] - num_reserved_subsequences
        last_cumulative = self.cumulative_subsequence_count[M - 1] if M - 1 >= 0 else 0
        self.cumulative_subsequence_count = self.cumulative_subsequence_count[M:]
        self.cumulative_subsequence_count -= last_cumulative
        self.all_indices = self.all_indices[M:]
        self.all_sequence_lengths = self.all_sequence_lengths[M:]
        self.num_subsequences = self.num_subsequences[M:]
        if num_reserved_subsequences < 0:
            assert self.all_indices.shape[0] == 0, 'All sequences should be used but \
                {} remain'.format(self.all_indices.shape[0])
        assert self._sequence_indices.shape  == self._sequence_lengths.shape
        assert self._sequence_lengths.shape  == self._subsequence_indices.shape
        assert self._subsequence_indices.shape  == self._subsequence_lengths.shape
        assert self.all_indices.shape  == self.all_sequence_lengths.shape
        assert self.all_sequence_lengths.shape  == self.num_subsequences.shape
        assert self.num_subsequences.shape  == self.cumulative_subsequence_count.shape
        return True


def get_epoch_loader(data_reader, partition_name, feature_name, batch_size, seed,
        window_length, K, stride_length):
    num_items = data_reader.ptn_num_items[partition_name]
    jagged = data_reader.jagged_length[feature_name]
    variable = data_reader.variable_length_segments[feature_name]
    windowed = stride_length is not None
    if not jagged:
        return NonSequentialEpochLoader(num_items, batch_size, seed)
    if variable:
        sequence_length_func = lambda indices: \
                data_reader.sequence_length_batch(partition_name, feature_name, indices)
        if windowed: 
            return WindowedSequentialEpochLoader(num_items, batch_size, sequence_length_func,
                window_length, stride_length, seed)
        else:
            return RandomSequentialEpochLoader(num_items, batch_size, sequence_length_func,
                window_length, K, seed)
    else:
        sequence_length_func = lambda indices: \
                data_reader.num_segments_batch(partition_name, feature_name, indices)
        if windowed:
            return WindowedSequentialEpochLoader(num_items, batch_size, sequence_length_func,
                window_length, stride_length, seed)
        else:
            return RandomSequentialEpochLoader(num_items, batch_size, sequence_length_func,
                window_length, K, seed)
