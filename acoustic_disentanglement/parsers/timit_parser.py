import sphfile
import os
import glob
import subprocess
import numpy as np
import tempfile
import librosa

from speech_representations.parsers.parser_api import ParserBase
from speech_representations.features import list_features_by_tags, register_feature


class TimitParser(ParserBase):

    partitions = ['train', 'test']

    def list_features(self, partition_name):
        return list_features_by_tags(tags=['raw_audio']) + ['gender', 'dialect', 'speaker', 'phoneme']

    def get_partition_size(self, partition_name):
        directory = os.path.join(self.source_directory, partition_name)
        return int(subprocess.check_output(['du','-shm', directory]).split()[0].decode('utf-8'))

    _gender_values = ['m', 'f']
    _gender_map  = { k: i for i, k in enumerate(_gender_values) }

    @register_feature(normalize=False, save=True, length=1, dtype=np.int32, valueset=_gender_values)
    def gender(self, filename):
        return self._gender_map[filename.split('/')[-2][0]]

    _dialect_values = ['dr1', 'dr2', 'dr3', 'dr4', 'dr5', 'dr6', 'dr7', 'dr8']
    _dialect_map  = { k: i for i, k in enumerate(_dialect_values) }

    @register_feature(normalize=False, save=True, length=1, dtype=np.int32, valueset=_dialect_values)
    def dialect(self, filename):
        return self._dialect_map[filename.split('/')[-3]]

    @property
    def _speaker_values(self):
        if hasattr(self, '_speaker_values_store'):
            return self._speaker_values_store
        filename_list = sorted(glob.glob('{}/*/*/*/*.wav'.format(self.source_directory)))
        speaker_values = list(set([filename.split('/')[-2][1:] for filename in filename_list]))
        self._speaker_values_store = speaker_values
        return speaker_values

    @property
    def _speaker_map(self):
        if hasattr(self, '_speaker_map_store'):
            return self._speaker_map_store
        speaker_map =  { k: i for i, k in enumerate(self._speaker_values) }
        self._speaker_map_store = speaker_map
        return speaker_map

    @register_feature(normalize=False, save=True, length=1, dtype=np.int32, valueset=lambda obj: obj._speaker_values)
    def speaker(self, filename):
        return self._speaker_map[filename.split('/')[-2][1:]]

    _phoneme_values = ['aa', 'ae', 'ah', 'ao', 'aw', 'ax', 'ax-h', 'axr', 'ay', 'b', 'bcl', 'ch', 'd', 'dcl',
                       'dh', 'dx', 'eh', 'el', 'em', 'en', 'eng', 'epi', 'er', 'ey', 'f', 'g', 'gcl', 'h#',
                       'hh', 'hv', 'ih', 'ix', 'iy', 'jh', 'k', 'kcl', 'l', 'm', 'n', 'ng', 'nx', 'ow', 'oy',
                       'p', 'pau', 'pcl', 'q', 'r', 's', 'sh', 't', 'tcl', 'th', 'uh', 'uw', 'ux', 'v', 'w', 
                       'y', 'z', 'zh']
    _phoneme_map  = { k: i for i, k in enumerate(_phoneme_values) }

    @register_feature(normalize=False, save=True, length=1, dtype=np.int32, variable_length_segments=True, jagged_length=True,
        valueset=_phoneme_values)
    def phoneme(self, filename):
        phn_filename = filename.replace('.wav', '.phn')
        values = []
        ends = []
        for line in open(phn_filename).readlines():
            start, end, phone = line.split()
            values += [self._phoneme_map[phone]]
            ends += [int(end)]
        return np.array(values).reshape(1, -1), np.array(ends)

    @register_feature
    def wav(self, filename):
        sph = sphfile.SPHFile(filename)
        with tempfile.NamedTemporaryFile() as wav_file:
            sph.write_wav(wav_file.name)
            wav, _ = librosa.load(wav_file.name, self.sr, mono=True)
        return wav

    def generate_data(self, partition_name, start_index=None, end_index=None):
        """ Generate all available items present within a partition of that dataset

        Must always return the same order across all versions/systems
        """
        start_index = start_index if start_index is not None else 0
        end_index = end_index if end_index is not None else self.num_items(partition_name)
        directory = os.path.join(self.source_directory, partition_name)
        filelist = sorted(glob.glob('{}/*/*/*.wav'.format(directory)))
        for filename in filelist[start_index:end_index]:
            self.set({'filename': filename})
            yield True

    def num_items(self, partition_name):
        directory = os.path.join(self.source_directory, partition_name)
        return len(glob.glob('{}/*/*/*.wav'.format(directory)))

