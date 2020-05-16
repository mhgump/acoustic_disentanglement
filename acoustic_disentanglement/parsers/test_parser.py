import numpy as np
import librosa

from speech_representations.parsers.parser_api import ParserBase
from speech_representations.features import list_features_by_tags, register_feature


class TestParser(ParserBase):

    partitions = ['0', '1']

    def list_features(self, partition_name):
        if partition_name == 0:
            return list_features_by_tags(tags=['raw_audio'])
        else:
            return list_features_by_tags(tags=['raw_audio']) + ['speaker', 'phoneme']

    def list_filesets(self, partition_name):
        return [{} for i in range(self.num_items(partition_name))]

    def num_items(self, partition_name):
        return int(partition_name) + 4

    def get_partition_size(self, partition_name):
        return self.num_items(partition_name)

    _speaker_valueset = [0, 1]

    @register_feature(normalize=False, save=True, length=1, dtype=np.int32, valueset=_speaker_valueset)
    def speaker(self, sample):
        return np.random.choice(self._speaker_valueset)

    _phoneme_valueset = ['a', 'b']

    @register_feature(normalize=False, save=True, length=1, dtype=np.int32, 
        variable_length_segments=True, jagged_length=True, valueset=_phoneme_valueset)
    def phoneme(self, sample):
        sequence_length = np.random.randint(2, 8)
        lengths = np.random.randint(2, 8, size=(sequence_length,))
        ends = np.cumsum(lengths)
        value = np.arange(sequence_length).reshape(1, -1) * np.reciprocal(sequence_length * 1.) * 100
        value = np.repeat(value, self.length('phoneme'), axis=0).astype(self.dtype('phoneme'))
        return value, ends

    @register_feature
    def wav(self, sample):
        wav_filename = librosa.util.example_audio_file()
        wav, _ = librosa.load(wav_filename, self.sr, mono=True, duration=0.25)
        return wav

    def generate_data(self, partition_name, start_index=None, end_index=None):
        """ Generate all available items present within a partition of that dataset

        Must always return the same order across all versions/systems
        """
        start_index = start_index if start_index is not None else 0
        end_index = end_index if end_index is not None else self.num_items(partition_name)
        for i in range(start_index, end_index):
            self.set({'sample': i})
            yield True
