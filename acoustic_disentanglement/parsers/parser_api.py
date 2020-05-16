import argparse
import numpy as np
import types

from speech_representations.utils import SubscriptableDescriptor
from speech_representations.features import FeatureSpec, transformer, AudioArguments


@transformer(tags=['raw_audio'])
class ParserBase:

    partitions = None
    arguments_class = AudioArguments

    def __init__(self, source_directory, **kwargs):
        self.source_directory = source_directory
        self.feature_list = []
        for kwarg_name, kwarg in kwargs.items():
            setattr(self, kwarg_name, kwarg)

    def length(self, feature_name):
        return getattr(self, feature_name).length

    def jagged_length(self, feature_name):
        return getattr(self, feature_name).jagged_length

    def variable_length_segments(self, feature_name):
        return getattr(self, feature_name).variable_length_segments

    def dtype(self, feature_name):
        return getattr(self, feature_name).dtype

    def valueset(self, feature_name):
        return getattr(self, feature_name).valueset

    def list_features(self, partition_name):
        raise NotImplementedError('Abstract class not implemented.')

    def num_items(self, partition_name):
        raise NotImplementedError('Abstract class not implemented.')

    def get_partition_size(self, partition_name):
        raise NotImplementedError('Abstract class not implemented.')

    def generate_data(self, partition_name, start_index=None, end_index=None):
        """ Generate all available items present within a partition of that dataset

        Must always return the same order across all versions/systems
        """
        raise NotImplementedError('Abstract class not implemented.')
