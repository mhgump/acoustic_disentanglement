import os
import glob
import subprocess
import numpy as np
import tempfile
import librosa
import sox

from speech_representations.parsers.parser_api import ParserBase
from speech_representations.features import list_features_by_tags, register_feature


class LibrispeechParser(ParserBase):

    partitions = ['dev-clean', 'dev-other', 'test-clean', 'test-other', 'train-clean-100', 'train-clean-360', 'train-other-500']
    
    def list_features(self, partition_name):
        return list_features_by_tags(tags=['raw_audio']) + ['gender', 'speaker']

    def get_partition_size(self, partition_name):
        directory = os.path.join(self.source_directory, partition_name)
        return int(subprocess.check_output(['du','-shm', directory]).split()[0].decode('utf-8'))

    def _load_speaker_file(self):
        num_speakers = 0
        self._speaker_map_store = {}
        self._gender_map_store = {}
        with open(os.path.join(self.source_directory, 'SPEAKERS.TXT')) as f:
            for line in f.readlines():
                if line[0] == ';':
                    continue
                speaker_id, gender_str, _, _, _ = line.split(' | ')
                speaker_id = int(speaker_id.strip())
                gender_str = gender_str.strip()
                self._speaker_map_store[speaker_id] = num_speakers
                self._gender_map_store[speaker_id] = gender_str == 'F'
                num_speakers += 1

    _gender_values = ['m', 'f']

    @register_feature(normalize=False, save=True, length=1, dtype=np.int32, valueset=_gender_values)
    def gender(self, filename):
        if not hasattr(self, '_gender_map_store'):
            self._load_speaker_file()
        speaker_id = int(filename.split('/')[-3])
        return self._gender_map_store[speaker_id]

    @property
    def _speaker_values(self):
        if not hasattr(self, '_speaker_map_store'):
            self._load_speaker_file()
        reverse_map = { v: k for k, v in self._speaker_map_store.items() }
        return [reverse_map[i] for i in range(len(self._speaker_map_store))]

    @register_feature(normalize=False, save=True, length=1, dtype=np.int32, valueset=lambda obj: obj._speaker_values)
    def speaker(self, filename):
        if not hasattr(self, '_speaker_map_store'):
            self._load_speaker_file()
        speaker_id = int(filename.split('/')[-3])
        return self._speaker_map_store[speaker_id]

    @register_feature
    def wav(self, filename):
        with tempfile.NamedTemporaryFile(suffix='.wav') as wav_file:
            sox.Transformer().build(filename, wav_file.name)
            wav, _ = librosa.load(wav_file.name, self.sr, mono=True)
        return wav

    def generate_data(self, partition_name, start_index=None, end_index=None):
        """ Generate all available items present within a partition of that dataset

        Must always return the same order across all versions/systems
        """
        start_index = start_index if start_index is not None else 0
        directory = os.path.join(self.source_directory, partition_name)
        filelist = sorted(glob.glob('{}/*/*/*.flac'.format(directory)))
        filelist = filelist[start_index:end_index] if end_index is not None else filelist[start_index:]
        for filename in filelist:
            self.set({'filename': filename})
            yield True

    def num_items(self, partition_name):
        directory = os.path.join(self.source_directory, partition_name)
        return len(glob.glob('{}/*/*/*.flac'.format(directory)))

